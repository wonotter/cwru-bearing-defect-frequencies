"""학습 없이 외륜·내륜·정상을 분류하고 공통 12 kHz 조건에서 평가한다.

실행: python main.py --rule-classification --segment-seconds 1
정답 라벨은 평가에만 사용한다. classify_segment에는 신호, fs, RPM만 전달한다.
"""

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from config import (
    DATA_DIR, DATASETS, DEFECT_FREQ_MULTIPLIERS,
    ENVELOPE_BANDPASS_HIGH, ENVELOPE_BANDPASS_LOW, PAPER_FILTER_ORDER,
    RESULTS_DIR,
)
from data_loader import load_mat_file
from signal_analysis import compute_paper_envelope_spectrum


# 점수 배열, CSV, 혼동 행렬에서 사용하는 순서를 고정한다.
CLASS_NAMES = ("Outer Race", "Inner Race", "Normal")
CLASS_FREQS = ("BPFO", "BPFI")
CLASS_LABELS = ("외륜", "내륜", "정상")


@dataclass(frozen=True)
class FrequencyRule:
    """평가 결과에 맞춰 조정하지 않는, 모든 파일에 공통인 분석 규칙."""

    band_low_hz: float = ENVELOPE_BANDPASS_LOW
    band_high_hz: float = ENVELOPE_BANDPASS_HIGH
    n_harmonics: int = 3
    relative_tolerance: float = 0.02
    tolerance_bins: float = 2.0
    noise_low_hz: float = 20.0
    noise_high_hz: float = 1000.0
    # 초기 휴리스틱: 두 결함의 평균 피크/배경 비율이 모두 10 미만이면 정상.
    # 평가 데이터의 정답을 보고 자동으로 최적화하지 않는다.
    normal_threshold: float = 10.0


DEFAULT_RULE = FrequencyRule()


def score_spectrum(freqs, amplitudes, rpm, rule=DEFAULT_RULE):
    """외륜/내륜 점수 = 1~3배 주파수 주변 최대 진폭 / 배경 진폭의 평균.

    검색 반폭은 max(주파수 분해능 × 2, 목표 주파수 × 2%)이다.
    배경은 20~1000 Hz에서 두 결함의 검색 영역을 모두 제외한 진폭의 중앙값.
    볼 주파수는 검색/점수/배경 제외 영역 모두에서 제거한다.
    """
    freqs = np.asarray(freqs, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    if (freqs.ndim != 1 or amplitudes.shape != freqs.shape or len(freqs) < 2
            or not np.all(np.isfinite(freqs))
            or not np.all(np.isfinite(amplitudes))
            or np.any(np.diff(freqs) <= 0) or np.any(amplitudes < 0)):
        raise ValueError("스펙트럼은 증가하는 주파수와 유한한 비음수 진폭이어야 합니다.")
    if not np.isfinite(rpm) or rpm <= 0:
        raise ValueError("RPM은 양수여야 합니다.")
    if (rule.n_harmonics < 1 or rule.relative_tolerance < 0
            or rule.tolerance_bins < 0
            or not 0 <= rule.noise_low_hz < rule.noise_high_hz):
        raise ValueError("결함 주파수 규칙의 범위가 유효하지 않습니다.")

    resolution = float(np.median(np.diff(freqs)))
    noise_mask = ((freqs >= rule.noise_low_hz)
                  & (freqs <= rule.noise_high_hz))
    details = []
    for freq_name in CLASS_FREQS:
        base_hz = DEFECT_FREQ_MULTIPLIERS[freq_name] * rpm / 60.0
        class_details = []
        for harmonic in range(1, rule.n_harmonics + 1):
            target = harmonic * base_hz
            tolerance = max(rule.tolerance_bins * resolution,
                            rule.relative_tolerance * target)
            if target + tolerance > min(freqs[-1], rule.noise_high_hz):
                raise ValueError("분석 범위가 모든 결함 고조파를 포함하지 못합니다.")
            mask = np.abs(freqs - target) <= tolerance
            if not np.any(mask):
                raise ValueError("검색 구간에 FFT 주파수 점이 없습니다.")
            indices = np.flatnonzero(mask)
            peak_index = indices[np.argmax(amplitudes[indices])]
            class_details.append({
                "harmonic": harmonic,
                "target_hz": float(target),
                "tolerance_hz": float(tolerance),
                "peak_hz": float(freqs[peak_index]),
                "peak_amplitude": float(amplitudes[peak_index]),
            })
            noise_mask &= ~mask
        details.append(class_details)
    if not np.any(noise_mask):
        raise ValueError("배경 진폭을 계산할 주파수 점이 부족합니다.")

    # 0 배경에서도 계산 가능하게 하되 신호 전체의 진폭 배율에는 불변이다.
    background = float(np.median(amplitudes[noise_mask]))
    denominator = max(background, float(amplitudes.max()) * 1e-12,
                      np.finfo(float).tiny)
    scores = np.array([
        np.mean([peak["peak_amplitude"] / denominator for peak in class_details])
        for class_details in details
    ])
    for class_details in details:
        for peak in class_details:
            peak["peak_to_background"] = peak["peak_amplitude"] / denominator
    return scores, details, background


def predict_fault_or_normal(fault_scores, threshold=DEFAULT_RULE.normal_threshold):
    """두 결함 점수가 임계값 미만이면 정상, 아니면 더 큰 점수의 결함.

    정확히 임계값과 같으면 결함으로 판정한다. 결함 점수 동점이면 외륜 우선.
    """
    fault_scores = np.asarray(fault_scores, dtype=float)
    if (fault_scores.shape != (2,) or not np.all(np.isfinite(fault_scores))
            or np.any(fault_scores < 0) or not np.isfinite(threshold) or threshold <= 0):
        raise ValueError("두 결함 점수는 비음수이고 정상 임계값은 양수여야 합니다.")
    if fault_scores.max() < threshold:
        return "Normal"
    return CLASS_NAMES[int(np.argmax(fault_scores))]


def resample_to_12k(signal, original_fs):
    """48 kHz 정상 신호를 안티앨리어싱 필터와 함께 12 kHz로 낮춘다.

    단순히 fs 표기만 바꾸거나 네 샘플 중 하나만 뽑지 않는다.
    12 kHz 결함 신호는 그대로 사용한다.
    """
    if original_fs == 12000:
        return np.asarray(signal)
    if original_fs == 48000:
        return resample_poly(signal, up=1, down=4)
    raise ValueError(f"지원하지 않는 원본 샘플링 레이트: {original_fs} Hz")


def classify_segment(signal, fs, rpm, rule=DEFAULT_RULE):
    """정답, 파일 ID, 결함 종류를 받지 않는 외륜/내륜/정상 분류기.

    밴드패스와 엔벨로프는 기존 논문 양식 그래프와 같은 함수를 사용한다.
    정상 ROC 점수는 -max(외륜 점수, 내륜 점수)이며 확률이 아니다.
    결함 증거가 약할수록 정상 점수가 높아진다. 최종 예측은 별도 임계값 규칙.
    """
    if not 0 < rule.band_low_hz < rule.band_high_hz < fs / 2:
        raise ValueError("공통 밴드패스 대역이 샘플링 레이트에 맞지 않습니다.")
    freqs, amplitudes, _ = compute_paper_envelope_spectrum(
        np.asarray(signal), fs, band=(rule.band_low_hz, rule.band_high_hz),
    )
    fault_scores, peaks, background = score_spectrum(freqs, amplitudes, rpm, rule)
    prediction = predict_fault_or_normal(fault_scores, rule.normal_threshold)
    scores = np.append(fault_scores, -float(fault_scores.max()))
    return {
        "predicted_label": prediction,
        "scores": scores,
        "fault_scores": fault_scores,
        "freqs": freqs,
        "amplitudes": amplitudes,
        "peaks": peaks,
        "background_amplitude": background,
    }


def evaluate_predictions(true_labels, predicted_labels, scores):
    """sklearn은 학습에 사용하지 않고 F1/ROC 등의 계산기로만 사용한다.

    ROC는 각 결함 대 나머지 결함의 이진 평가(OvR)이다.
    확률로 변환하지 않은 연속 점수를 각 이진 roc_auc_score에 전달한다.
    """
    truth = np.asarray(true_labels)
    predictions = np.asarray(predicted_labels)
    scores = np.asarray(scores, dtype=float)
    if (truth.ndim != 1 or predictions.shape != truth.shape
            or scores.shape != (len(truth), len(CLASS_NAMES))
            or not np.all(np.isfinite(scores))
            or not set(truth).issubset(CLASS_NAMES)
            or not set(predictions).issubset(CLASS_NAMES)):
        raise ValueError("정답/예측/외륜·내륜·정상 점수의 형태 또는 라벨이 유효하지 않습니다.")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=list(CLASS_NAMES), zero_division=0,
    )
    per_class, curves = {}, {}
    for index, label in enumerate(CLASS_NAMES):
        binary_truth = (truth == label).astype(int)
        if len(np.unique(binary_truth)) != 2:
            raise ValueError("ROC 평가에는 각 결함의 양성과 음성 구간이 모두 필요합니다.")
        fpr, tpr, thresholds = roc_curve(binary_truth, scores[:, index])
        auc = float(roc_auc_score(binary_truth, scores[:, index]))
        per_class[label] = {
            "precision": float(precision[index]), "recall": float(recall[index]),
            "f1": float(f1[index]), "support": int(support[index]), "auc_ovr": auc,
        }
        curves[label] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
    metrics = {
        "n_segments": len(truth),
        "class_order": list(CLASS_NAMES),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1.mean()),
        "macro_auc_ovr": float(np.mean([v["auc_ovr"] for v in per_class.values()])),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            truth, predictions, labels=list(CLASS_NAMES),
        ).tolist(),
    }
    return metrics, curves


def _dataset_keys():
    """12 kHz 외륜/내륜과 같은 0 HP의 정상 97을 선택하고 볼은 제외한다."""
    root = Path(DATA_DIR, "12k_drive_end_fault").resolve()
    files = set(root.rglob("*.mat"))
    registered_keys = [key for key, info in DATASETS.items()
                       if Path(info["file"]).resolve() in files and info["fs"] == 12000]
    registered = {Path(DATASETS[key]["file"]).resolve() for key in registered_keys}
    if files - registered:
        raise ValueError("config.DATASETS에 등록되지 않은 12 kHz 파일: "
                         + ", ".join(str(p) for p in sorted(files - registered)))
    keys = [key for key in registered_keys
            if DATASETS[key]["fault_type"] in ("Outer Race", "Inner Race")]
    if not keys:
        raise ValueError("12k_drive_end_fault 폴더에 외륜/내륜 .mat 파일이 없습니다.")
    if not Path(DATASETS["Normal_0"]["file"]).is_file():
        raise ValueError("0 HP 정상 데이터인 normal_baseline/97.mat가 없습니다.")
    keys.append("Normal_0")
    return sorted(keys, key=lambda key: (
        CLASS_NAMES.index(DATASETS[key]["fault_type"]), DATASETS[key]["file_id"],
    ))


def run_rule_classification(segment_seconds=1.0, rule=DEFAULT_RULE):
    """외륜/내륜/정상을 공통 12 kHz의 겹치지 않는 구간으로 평가한다."""
    if not np.isfinite(segment_seconds) or segment_seconds <= 0:
        raise ValueError("구간 길이는 양수여야 합니다.")
    if not np.isfinite(rule.normal_threshold) or rule.normal_threshold <= 0:
        raise ValueError("정상 판정 임계값은 양수여야 합니다.")
    keys = _dataset_keys()
    # 파일명도 실제 표본 길이에 맞춰서 재현 가능하게 만든다.
    segment_samples = int(round(segment_seconds * 12000))
    if segment_samples < 1200:
        raise ValueError("결함 주파수 분석에는 최소 0.1초 구간을 사용하세요.")
    actual_seconds = segment_samples / 12000
    output = Path(RESULTS_DIR, "rule_classification", f"12k_outer_inner_normal_{actual_seconds:g}s")
    output.mkdir(parents=True, exist_ok=True)

    # 그래프 함수는 실험 실행 때에만 불러온다.
    from rule_classifier_plots import plot_evaluation, plot_segment_spectrum
    records, rows, details, spectra = [], [], [], []
    for key in keys:
        data = load_mat_file(key)
        original_fs = data["fs"]
        signal = resample_to_12k(data["signal"], original_fs)
        fs, rpm = 12000, data["rpm"]
        n_segments = len(signal) // segment_samples
        if n_segments == 0:
            raise ValueError(f"{key}: 신호 길이보다 평가 구간이 깁니다.")
        records.append({
            "dataset": key, "file_id": data["info"]["file_id"],
            "source_file": str(Path(data["info"]["file"]).relative_to(DATA_DIR)),
            "true_label": data["info"]["fault_type"],
            "original_fs_hz": original_fs, "fs_hz": fs,
            "resampling": "resample_poly(up=1, down=4)" if original_fs == 48000 else "none",
            "rpm": rpm, "rpm_source": data["rpm_source"],
            "original_duration_seconds": data["duration_s"],
            "duration_seconds": len(signal) / fs, "n_segments": n_segments,
            "unused_tail_seconds": (len(signal) % segment_samples) / fs,
        })
        for index in range(n_segments):
            start = index * segment_samples
            result = classify_segment(signal[start:start + segment_samples], fs, rpm, rule)
            # 정답은 점수 계산이 끝난 뒤에 평가 표에만 붙인다.
            row = {
                "dataset": key, "file_id": data["info"]["file_id"],
                "segment": index + 1, "start_s": start / fs,
                "end_s": (start + segment_samples) / fs, "rpm": rpm,
                "true_label": data["info"]["fault_type"],
                "predicted_label": result["predicted_label"],
                "score_BPFO": float(result["scores"][0]),
                "score_BPFI": float(result["scores"][1]),
                "score_Normal": float(result["scores"][2]),
                "max_fault_score": float(result["fault_scores"].max()),
                "normal_threshold": rule.normal_threshold,
                "background_amplitude": result["background_amplitude"],
            }
            rows.append(row)
            details.append({"dataset": key, "segment": index + 1,
                            "peaks": dict(zip(CLASS_FREQS, result["peaks"]))})
            spectra.append((row, result))
        print(f"{key}: {n_segments} segments, RPM={rpm:g}, "
              f"unused tail={records[-1]['unused_tail_seconds']:.3f}s")

    scores = np.array([[row[f"score_{name}"] for name in ("BPFO", "BPFI", "Normal")]
                       for row in rows])
    metrics, curves = evaluate_predictions(
        [row["true_label"] for row in rows],
        [row["predicted_label"] for row in rows], scores,
    )
    with (output / "segment_predictions.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "experiment": "Common 12 kHz / outer race, inner race, normal / no training",
        "excluded_fault_types": ["Ball"],
        "segment_seconds": actual_seconds, "overlap_seconds": 0,
        "rule": asdict(rule), "filter_order": PAPER_FILTER_ORDER,
        "score_definition": "mean(peak amplitude / common background) for harmonics "
                            f"1..{rule.n_harmonics}",
        "prediction_rule": "Normal if max(BPFO, BPFI) < normal_threshold; "
                           "otherwise argmax of BPFO and BPFI",
        "normal_score_definition": "-max(BPFO score, BPFI score); used for Normal OvR ROC only",
        "threshold_origin": "Fixed initial heuristic of 10x mean peak/background ratio. "
                            "Not optimized on evaluation labels; not a validated universal cutoff.",
        "roc_definition": "binary one-vs-rest per class, using raw continuous scores",
        "evaluation_scope": "All complete windows; descriptive evaluation of these recordings. "
                            "Windows from the same recording are not independent test recordings.",
        "records": records, "metrics": metrics,
    }
    for name, content in (("metrics.json", report), ("peak_details.json", details)):
        (output / name).write_text(json.dumps(content, ensure_ascii=False, indent=2,
                                             allow_nan=False), encoding="utf-8")
    plot_evaluation(rows, metrics, curves, output, actual_seconds, rule.normal_threshold)
    for row, result in spectra:
        plot_segment_spectrum(row, result, rule, output / "spectra")

    print(f"Accuracy={metrics['accuracy']:.3f}, Macro F1={metrics['macro_f1']:.3f}, "
          f"Macro OvR AUC={metrics['macro_auc_ovr']:.3f}")
    for label, values in metrics["per_class"].items():
        print(f"{label}: F1={values['f1']:.3f}, AUC={values['auc_ovr']:.3f}")
    print(f"Results: {output}")
    return report
