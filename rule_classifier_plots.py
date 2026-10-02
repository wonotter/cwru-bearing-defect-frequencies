"""기존 프로젝트의 폰트·결함 색상·엔벨로프 양식을 사용한 분류 결과 그래프."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import DEFECT_FREQ_COLORS, FIGURE_DPI, PAPER_PLOT_FREQ_MAX
from rule_classifier import CLASS_FREQS, CLASS_LABELS, CLASS_NAMES


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
COLORS = [DEFECT_FREQ_COLORS[name]["color"] for name in CLASS_FREQS] + ["#7F8C8D"]
SCORE_FIELDS = ("score_BPFO", "score_BPFI")
FREQ_LABELS = ("BPFO (외륜)", "BPFI (내륜)")


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_evaluation(rows, metrics, curves, output, segment_seconds, normal_threshold):
    n_records = len({row["dataset"] for row in rows})
    caption = (f"12 kHz · {segment_seconds:g}초 구간 {len(rows)}개 · "
               f"원본 파일 {n_records}개 내 구간 평가")
    # 혼동 행렬: 실제 라벨이 행, 규칙 분류기의 예측이 열이다.
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    fontsize=17, color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set_xticks(range(3), CLASS_LABELS)
    ax.set_yticks(range(3), CLASS_LABELS)
    ax.set_xlabel("예측 상태", fontsize=11)
    ax.set_ylabel("실제 상태", fontsize=11)
    ax.set_title("외륜·내륜·정상 규칙 분류 — 혼동 행렬\n" + caption,
                 fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, label="구간 수")
    _save(fig, output / "01_confusion_matrix.png")

    # 클래스별 F1과 세 클래스의 단순 평균(Macro F1).
    fig, ax = plt.subplots(figsize=(8, 5))
    f1 = [metrics["per_class"][name]["f1"] for name in CLASS_NAMES]
    bars = ax.bar(CLASS_LABELS, f1, color=COLORS, width=0.55)
    for bar, value in zip(bars, f1):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025,
                f"{value:.3f}", ha="center", fontsize=12)
    ax.axhline(metrics["macro_f1"], color="#555555", linestyle="--", linewidth=1,
               label=f"Macro F1 = {metrics['macro_f1']:.3f}")
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("F1 score")
    ax.set_title("외륜·내륜·정상 F1 score\n" + caption, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), fontsize=9)
    _save(fig, output / "02_f1_scores.png")

    # 정답 종류별 연속 점수의 임계값을 변화시킨 이진 OvR ROC.
    fig, ax = plt.subplots(figsize=(7.5, 6))
    # 완벽한 ROC들이 겹쳐도 각 색을 확인할 수 있게 선 모양을 구분한다.
    for index, (name, label, color) in enumerate(zip(CLASS_NAMES, CLASS_LABELS, COLORS)):
        curve = curves[name]
        ax.plot(curve["fpr"], curve["tpr"], color=color,
                linewidth=3.0 - index * 0.8,
                linestyle=("-", "--", ":")[index],
                label=f"{label} vs 나머지 (AUC = {metrics['per_class'][name]['auc_ovr']:.3f})")
    ax.plot([0, 1], [0, 1], color="#95A5A6", linestyle="--", linewidth=1,
            label="무작위 기준 (AUC = 0.500)")
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.05),
           xlabel="False Positive Rate (다른 상태를 해당 상태로 판정한 비율)",
           ylabel="True Positive Rate (해당 상태를 찾아낸 비율)")
    ax.set_title("외륜·내륜·정상 ROC — 연속 주파수 점수 사용\n" + caption,
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, output / "03_roc_curves.png")

    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    fig, axes = plt.subplots(len(datasets), 1, figsize=(11, 3.2 * len(datasets)),
                             squeeze=False)
    for ax, dataset in zip(axes[:, 0], datasets):
        subset = [row for row in rows if row["dataset"] == dataset]
        times = [(row["start_s"] + row["end_s"]) / 2 for row in subset]
        for field, label, color in zip(SCORE_FIELDS, CLASS_LABELS, COLORS):
            ax.plot(times, [row[field] for row in subset], "o-", color=color,
                    linewidth=1.1, markersize=4, label=f"{label} 점수")
        ax.axhline(normal_threshold, color=COLORS[2], linestyle="--", linewidth=1,
                   label=f"정상 기준 = {normal_threshold:g}")
        truth = CLASS_LABELS[CLASS_NAMES.index(subset[0]["true_label"])]
        correct = sum(row["true_label"] == row["predicted_label"] for row in subset)
        ax.set_title(f"{subset[0]['file_id']}DE · 실제 {truth} · 정답 {correct}/{len(subset)}구간",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("피크 / 배경 진폭 평균")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    axes[-1, 0].set_xlabel("구간 중심 시간 (s)")
    fig.suptitle(f"두 결함 점수가 모두 {normal_threshold:g} 미만이면 정상, "
                 "그 외에는 더 높은 점수의 결함", fontsize=12, fontweight="bold")
    _save(fig, output / "04_segment_scores.png")


def plot_segment_spectrum(row, result, rule, output):
    """각 구간의 실제 분류에 쓴 스펙트럼, 검색 범위와 선택된 피크를 표시한다."""
    fig, ax = plt.subplots(figsize=(10, 5))
    mask = result["freqs"] <= PAPER_PLOT_FREQ_MAX
    ax.plot(result["freqs"][mask], result["amplitudes"][mask],
            color="#1F77B4", linewidth=0.7, label="Envelope spectrum")
    for name, label, color, peaks in zip(CLASS_FREQS, FREQ_LABELS, COLORS, result["peaks"]):
        for index, peak in enumerate(peaks):
            target, tolerance = peak["target_hz"], peak["tolerance_hz"]
            ax.axvspan(target - tolerance, target + tolerance, color=color, alpha=0.07)
            ax.axvline(target, color=color, linestyle="--", linewidth=1, alpha=0.8,
                       label=label if index == 0 else None)
            ax.plot(peak["peak_hz"], peak["peak_amplitude"], "o", color=color,
                    markersize=4)
    truth = CLASS_LABELS[CLASS_NAMES.index(row["true_label"])]
    prediction = CLASS_LABELS[CLASS_NAMES.index(row["predicted_label"])]
    ax.set_title(
        f"{row['file_id']}DE · {row['start_s']:.3f}–{row['end_s']:.3f} s · "
        f"실제: {truth} / 예측: {prediction}\n"
        f"Bandpass {rule.band_low_hz:g}–{rule.band_high_hz:g} Hz · RPM {row['rpm']:g} · "
        "점선: 이론 주파수 / 음영: 검색 범위 / 원: 선택한 피크",
        fontsize=11, fontweight="bold",
    )
    score_text = " · ".join(f"{label} {score:.2f}" for label, score
                            in zip(CLASS_LABELS[:2], result["fault_scores"]))
    ax.text(0.02, 0.95, f"점수: {score_text} · 둘 다 {rule.normal_threshold:g} 미만이면 정상",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    ax.set(xlim=(0, PAPER_PLOT_FREQ_MAX), ylim=(0, None),
           xlabel="주파수 (Hz)", ylabel="엔벨로프 진폭 (g)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    filename = (f"{row['file_id']}DE_{row['segment']:02d}_"
                f"{row['start_s']:.3f}-{row['end_s']:.3f}s.png")
    _save(fig, output / filename)
