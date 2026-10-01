"""
CWRU 베어링 결함 주파수 검증 프로젝트 - 메인 실행 파이프라인

48kHz, 0.021", 1HP(1772 RPM) 조건에서 수집된 6개 데이터셋에 대해
FFT 및 엔벨로프 스펙트럼 분석을 수행하고, 결함 주파수 마커를 표시하여
BPFO/BPFI/BSF/FTF가 실제로 관측되는지 검증한다.

실행 방법:
    python main.py
    python main.py --base-kurtogram-envelope  # 48 kHz / 1 HP 결과만 생성
    python main.py --ball-segments           # 227DE의 구간별 엔벨로프
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import BASE_KEYS, PAPER_KEYS, RESULTS_DIR, FIGURE_DPI
from data_loader import load_all_datasets, load_mat_file, print_dataset_summary
from signal_analysis import compute_kurtogram, print_defect_frequencies
from visualization import (
    plot_analysis,
    plot_comparison,
    plot_all_envelope_overview,
    plot_method1_analysis,
    plot_method1_all_overview,
    plot_method1_comparison,
    plot_method2_analysis,
    plot_method2_all_overview,
    plot_method2_comparison,
    plot_method_comparison,
    plot_paper_envelope_spectrum,
    print_harmonic_amplitudes,
    plot_kurtogram,
)

# 논문 Fig.5~8과 대응하는 데이터셋 및 표시할 결함 주파수
PAPER_FIGURES = {
    "IR007_0": ("Fig.5", ["BPFI"]),
    "OR007@6_0": ("Fig.6", ["BPFO"]),
    "Normal_0": ("Fig.7", ["BPFI", "BPFO", "BSF"]),
    "B007_0": ("Fig.8", ["BSF"]),
}


def run_paper_reproduction():
    """
    논문(Alonso-González et al., 2023) Fig.5~8을 재현한다.

    논문이 사용한 조건을 그대로 맞추는 것이 핵심이다.
        - 데이터: 12kHz Drive End, 결함 직경 0.007", 부하 0 HP, 1797 RPM
        - 처리  : Kurtogram이 고른 구간을 Butterworth 밴드패스 → 일반 엔벨로프 → FFT
        - 표시  : x축 0~1000 Hz, 해당 결함 주파수 고조파만 점선 표시
    """
    print("\n" + "=" * 60)
    print("  논문 Fig.5~8 재현")
    print("  12kHz | 0.007\" fault | 0HP (1797 RPM)")
    print("  Kurtogram 구간 Butterworth 밴드패스 + 일반 엔벨로프")
    print("=" * 60)

    paper_data = load_all_datasets(PAPER_KEYS)
    for data in paper_data.values():
        print_dataset_summary(data)

    for key, (figure_no, freq_names) in PAPER_FIGURES.items():
        data = paper_data[key]
        print(f"\n  [{figure_no}] {key} — {data['info']['description']}")
        plot_paper_envelope_spectrum(data, freq_names=freq_names, save=True)
        for freq_name in freq_names:
            print_harmonic_amplitudes(data, freq_name)
        plt.close("all")


def run_kurtograms():
    """
    논문 데이터 4개의 Kurtogram을 저장한다.

    각 그림은 주파수 구간마다 충격이 얼마나 강한지 보여 준다.
    검은 테두리가 엔벨로프 밴드패스로 고른 구간이다.
    """
    print("\n" + "=" * 60)
    print("  Kurtogram")
    print("  논문 데이터 (12 kHz / 0.007\" / 0 HP, 정상은 48 kHz)")
    print("=" * 60)

    paper_data = load_all_datasets(PAPER_KEYS)
    for key, data in paper_data.items():
        print(f"\n  Kurtogram 계산 중: {key}")
        plot_kurtogram(data, save=True)
        plt.close("all")


def run_base_kurtogram_envelopes(all_data: dict | None = None) -> None:
    """48 kHz / 1 HP 그룹의 Kurtogram과 선택 대역 엔벨로프를 저장한다."""
    if all_data is None:
        all_data = load_all_datasets(BASE_KEYS)

    print("\n" + "=" * 60)
    print("  48 kHz / 1 HP / 0.021\" Kurtogram + 엔벨로프 분석")
    print("  정상 기준: Normal_1 (48 kHz / 1 HP)")
    print("=" * 60)

    for key in BASE_KEYS:
        data = all_data[key]
        print(f"\n  처리 중: {key}")
        result = compute_kurtogram(data["signal"], data["fs"])
        best = result["best"]
        band = (best["low"], best["high"])
        plot_kurtogram(data, result=result, save=True)
        plot_paper_envelope_spectrum(data, band=band, save=True)
        plt.close("all")


def run_ball_segments(segment_seconds: float = 1.0) -> list[Path]:
    """227DE를 잘라 기존 양식의 엔벨로프 그래프를 구간마다 저장한다.

    기존 분석 그대로: Kurtogram 대역 -> 밴드패스 -> 일반 엔벨로프 -> FFT.
    시간에 따른 차이만 비교하도록 전체 기록에서 고른 대역을 모두 사용한다.
    BSF 점선도 이론 위치에 고정한다(피크에 따라 점선을 이동하지 않는다).
    """
    data = load_mat_file("B021_1")
    fs, signal = data["fs"], data["signal"]
    if not np.isfinite(segment_seconds) or segment_seconds <= 0:
        raise ValueError("구간 길이는 0보다 큰 초 단위 숫자여야 합니다.")
    window = round(segment_seconds * fs)
    if window < 128 or window > len(signal):
        raise ValueError("구간은 128개 이상의 샘플을 포함하고 전체 기록보다 길지 않아야 합니다.")

    # 기본은 0~1초, 1~2초, ... . 남은 끝부분은 마지막 1초 구간으로 포함한다.
    starts = list(range(0, len(signal) - window + 1, window))
    if starts[-1] + window < len(signal):
        starts.append(len(signal) - window)
    intervals = [(0, len(signal)), *[(start, start + window) for start in starts]]

    best = compute_kurtogram(signal, fs)["best"]
    band = (best["low"], best["high"])
    # 구간 길이를 바꿔 실행한 결과가 섞이지 않도록 길이별 폴더에 저장한다.
    output = Path(RESULTS_DIR) / "ball_segments" / f"{segment_seconds:g}s"
    output.mkdir(parents=True, exist_ok=True)
    figures = []
    for index, (start, stop) in enumerate(intervals):
        part = {
            **data,
            "signal": signal[start:stop],
            "n_samples": stop - start,
            "duration_s": (stop - start) / fs,
            # 제목과 결함 주파수 계산에 같은 파일 기록 RPM을 사용한다.
            "info": {**data["info"], "rpm": data["rpm"]},
        }
        fig = plot_paper_envelope_spectrum(
            part, freq_names=["BSF"], band=band,
            refine_markers=False, save=False,
        )
        ax = fig.axes[0]
        label = "Full record" if index == 0 else "Segment"
        ax.set_title(
            ax.get_title() + f"\n{label}: {start/fs:.3f} ~ {stop/fs:.3f} s",
            fontsize=11, fontweight="bold",
        )
        filename = ("00_full" if index == 0 else f"{index:02d}_segment")
        filename += f"_{start/fs:.3f}-{stop/fs:.3f}s.png"
        figures.append((fig, output / filename))
        plt.close(fig)

    # 자동 확대 때문에 약한 구간이 강해 보이지 않도록 동일한 y축을 쓴다.
    common_max = max(fig.axes[0].get_ylim()[1] for fig, _ in figures)
    for fig, path in figures:
        fig.axes[0].set_ylim(0, common_max)
        fig.tight_layout()
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"  [저장] {path}")

    print(f"\n227DE: {len(signal)/fs:.5f}초, 파일 기록 {data['rpm']:g} RPM")
    print(f"전체 1장 + 구간별 {len(starts)}장, 공통 대역 {band[0]:g}~{band[1]:g} Hz")
    if len(starts) > 1 and starts[-1] < starts[-2] + window:
        print("마지막 그림은 기록 끝을 포함하도록 앞 구간과 일부 겹칩니다.")
    print("초록색 BSF 점선과 그 배수 근처에 파란 피크가 나타나는지 비교하세요.")
    return [path for _, path in figures]


def run_shaft_removal_comparison(all_data: dict):
    """
    48kHz / 0.021" 데이터에 축 동기 성분 제거를 적용한 결과를 생성한다.

    이 데이터는 결함 폭이 넓어(0.021") BPFO 고조파가 약한 반면
    1×RPM 고조파가 강해서 논문 Fig.6처럼 깔끔한 그림이 나오지 않는다.
    축 동기 성분을 제거하면 BPFO 계열의 상대적 가시성이 개선된다.
    """
    print("\n" + "=" * 60)
    print("  48kHz / 0.021\" 데이터 — 축 동기 성분 제거 비교")
    print("=" * 60)

    for key in ["OR021@6_1", "IR021_1"]:
        data = all_data[key]
        freq_name = "BPFO" if data["info"]["fault_type"] == "Outer Race" else "BPFI"
        print(f"\n  처리 중: {key} ({freq_name})")
        plot_paper_envelope_spectrum(
            data, freq_names=[freq_name], remove_shaft_orders=True, save=True
        )
        print_harmonic_amplitudes(data, freq_name)
        plt.close("all")


def main():
    print("\n" + "=" * 60)
    print("  CWRU 베어링 결함 주파수 검증 프로젝트")
    print("  48kHz | 0.021\" fault | 1HP (1772 RPM)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1단계: 결함 주파수 계산 결과 출력
    # ------------------------------------------------------------------
    print_defect_frequencies(1772)

    # ------------------------------------------------------------------
    # 2단계: 전체 데이터셋 로드
    # ------------------------------------------------------------------
    print("\n[1/4] 데이터셋 로딩 중...")
    all_data = load_all_datasets(BASE_KEYS)
    for data in all_data.values():
        print_dataset_summary(data)

    # ------------------------------------------------------------------
    # 3단계: 개별 데이터셋 분석 플롯 (시간 영역 + FFT + 엔벨로프)
    # ------------------------------------------------------------------
    print("\n[2/4] 개별 데이터셋 분석 플롯 생성 중...")
    for key, data in all_data.items():
        print(f"\n  분석 중: {key}")
        plot_analysis(data, save=True)
        plt.close("all")

    # ------------------------------------------------------------------
    # 4단계: Normal vs 결함 비교 플롯
    # ------------------------------------------------------------------
    print("\n[3/4] Normal vs 결함 비교 플롯 생성 중...")
    normal_data = all_data["Normal_1"]
    fault_keys = [k for k in all_data.keys() if k != "Normal_1"]

    for fault_key in fault_keys:
        print(f"\n  비교 중: Normal_1 vs {fault_key}")
        plot_comparison(normal_data, all_data[fault_key], save=True)
        plt.close("all")

    # ------------------------------------------------------------------
    # 5단계: 전체 엔벨로프 스펙트럼 한눈에 보기
    # ------------------------------------------------------------------
    print("\n[4/4] 전체 엔벨로프 스펙트럼 오버뷰 생성 중...")
    plot_all_envelope_overview(all_data, save=True)
    plt.close("all")

    # ==================================================================
    # Method 1: 원신호 제곱 엔벨로프 스펙트럼 (Smith & Randall 2015, §5.1)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Method 1 - 원신호 제곱 엔벨로프 스펙트럼 분석")
    print("  (밴드패스 필터 없이, 원신호 -> 제곱 엔벨로프 -> FFT)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # M1-1: 개별 데이터셋 Method 1 분석 플롯
    # ------------------------------------------------------------------
    print("\n[M1-1] Method 1 개별 분석 플롯 생성 중...")
    for key, data in all_data.items():
        print(f"\n  Method 1 분석 중: {key}")
        plot_method1_analysis(data, save=True)
        plt.close("all")

    # ------------------------------------------------------------------
    # M1-2: Normal vs 결함 Method 1 비교 플롯
    # ------------------------------------------------------------------
    print("\n[M1-2] Method 1 Normal vs 결함 비교 플롯 생성 중...")
    for fault_key in fault_keys:
        print(f"\n  Method 1 비교 중: Normal_1 vs {fault_key}")
        plot_method1_comparison(normal_data, all_data[fault_key], save=True)
        plt.close("all")

    # ------------------------------------------------------------------
    # M1-3: 전체 Method 1 제곱 엔벨로프 스펙트럼 오버뷰
    # ------------------------------------------------------------------
    print("\n[M1-3] Method 1 전체 오버뷰 생성 중...")
    plot_method1_all_overview(all_data, save=True)
    plt.close("all")

    # ==================================================================
    # Method 2: 켑스트럼 프리화이트닝 (Smith & Randall 2015, §5.2)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Method 2 - 켑스트럼 프리화이트닝 + 제곱 엔벨로프 스펙트럼")
    print("  (주파수 크기 균일화 -> 제곱 엔벨로프 -> FFT)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # M2-1: 개별 데이터셋 Method 2 분석 플롯
    # ------------------------------------------------------------------
    print("\n[M2-1] Method 2 개별 분석 플롯 생성 중...")
    for key, data in all_data.items():
        print(f"\n  Method 2 분석 중: {key}")
        plot_method2_analysis(data, save=True)
        plt.close("all")

    # ------------------------------------------------------------------
    # M2-2: Normal vs 결함 Method 2 비교 플롯
    # ------------------------------------------------------------------
    print("\n[M2-2] Method 2 Normal vs 결함 비교 플롯 생성 중...")
    for fault_key in fault_keys:
        print(f"\n  Method 2 비교 중: Normal_1 vs {fault_key}")
        plot_method2_comparison(normal_data, all_data[fault_key], save=True)
        plt.close("all")

    # ------------------------------------------------------------------
    # M2-3: 전체 Method 2 제곱 엔벨로프 스펙트럼 오버뷰
    # ------------------------------------------------------------------
    print("\n[M2-3] Method 2 전체 오버뷰 생성 중...")
    plot_method2_all_overview(all_data, save=True)
    plt.close("all")

    # ==================================================================
    # Method 1 vs Method 2 비교
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Method 1 vs Method 2 비교")
    print("=" * 60)

    print("\n[M-CMP] Method 1 vs 2 비교 플롯 생성 중...")
    for key, data in all_data.items():
        if key == "Normal_1":
            continue
        print(f"\n  M1 vs M2 비교 중: {key}")
        plot_method_comparison(data, save=True)
        plt.close("all")

    # ==================================================================
    # 논문 Fig.5~8 재현 + 축 동기 성분 제거 비교
    # ==================================================================
    run_base_kurtogram_envelopes(all_data)
    run_paper_reproduction()
    run_kurtograms()
    run_shaft_removal_comparison(all_data)

    # ------------------------------------------------------------------
    # 완료 요약
    # ------------------------------------------------------------------
    n_m1 = len(all_data) + len(fault_keys) + 1
    n_m2 = len(all_data) + len(fault_keys) + 1
    n_m1_vs_m2 = len(fault_keys)
    n_baseline = len(all_data) + len(fault_keys) + 1
    n_paper = len(PAPER_FIGURES)
    n_kurtogram = len(PAPER_KEYS)
    n_base_kurtogram_envelope = 2 * len(BASE_KEYS)
    n_shaft_removal = 2  # OR021@6_1 / IR021_1 제거 후
    n_total = (n_baseline + n_m1 + n_m2 + n_m1_vs_m2
               + n_paper + n_kurtogram + n_base_kurtogram_envelope
               + n_shaft_removal)

    print("\n" + "=" * 60)
    print("  분석 완료!")
    print("=" * 60)
    print("\n  === 기존 분석 (밴드패스 엔벨로프) ===")
    print(f"    - 개별 + 비교 + 오버뷰  : {n_baseline}개")
    print("\n  === Method 1 (원신호 제곱 엔벨로프) ===")
    print(f"    - 개별 + 비교 + 오버뷰  : {n_m1}개")
    print("\n  === Method 2 (켑스트럼 프리화이트닝) ===")
    print(f"    - 개별 + 비교 + 오버뷰  : {n_m2}개")
    print("\n  === Method 1 vs 2 비교 ===")
    print(f"    - 결함 데이터 비교      : {n_m1_vs_m2}개")
    print("\n  === 논문 Fig.5~8 재현 (12kHz, 0.007\", 0HP) ===")
    print(f"    - 논문 대응 그림        : {n_paper}개")
    print(f"    - Kurtogram             : {n_kurtogram}개")
    print("\n  === 48kHz, 0.021\", 1HP (정상 기준 포함) ===")
    print(f"    - Kurtogram + 엔벨로프 : {n_base_kurtogram_envelope}개")
    print(f"    - 축 동기 성분 제거 비교: {n_shaft_removal}개")
    print(f"\n  총 이미지 파일            : {n_total}개")
    print("\n  결과 저장 위치: results/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CWRU 베어링 결함 주파수 분석")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--base-kurtogram-envelope", action="store_true",
        help="48 kHz / 1 HP / 0.021 inch 그룹의 Kurtogram과 엔벨로프만 생성",
    )
    mode.add_argument(
        "--ball-segments", action="store_true",
        help="227DE를 시간 구간별로 잘라 기존 양식의 엔벨로프 그래프 생성",
    )
    parser.add_argument(
        "--segment-seconds", type=float, default=1.0,
        help="--ball-segments에서 사용할 구간 길이(초), 기본 1초",
    )
    args = parser.parse_args()
    if args.ball_segments:
        try:
            run_ball_segments(args.segment_seconds)
        except ValueError as error:
            parser.error(str(error))
    elif args.base_kurtogram_envelope:
        run_base_kurtogram_envelopes()
    else:
        main()
