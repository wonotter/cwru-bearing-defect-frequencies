"""
CWRU 베어링 결함 주파수 검증 프로젝트 - 메인 실행 파이프라인

48kHz, 0.021", 1HP(1772 RPM) 조건에서 수집된 6개 데이터셋에 대해
FFT 및 엔벨로프 스펙트럼 분석을 수행하고, 결함 주파수 마커를 표시하여
BPFO/BPFI/BSF/FTF가 실제로 관측되는지 검증한다.

실행 방법:
    python main.py
"""

import matplotlib.pyplot as plt

from config import BASE_KEYS, PAPER_KEYS
from data_loader import load_all_datasets, print_dataset_summary
from signal_analysis import print_defect_frequencies
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
        - 처리  : 3.2~4.5 kHz 하우징 공진 밴드패스 → 일반 엔벨로프(|Hilbert|) → FFT
        - 표시  : x축 0~1000 Hz, 해당 결함 주파수 고조파만 점선 표시

    전대역 엔벨로프는 BPFO 5~7차가 다시 커져 Fig.6과 달라진다.
    하우징 공진만 남기면 고조파가 단조 감소한다.
    """
    print("\n" + "=" * 60)
    print("  논문 Fig.5~8 재현")
    print("  12kHz | 0.007\" fault | 0HP (1797 RPM)")
    print("  3.2~4.5 kHz 하우징 공진 밴드패스 + 일반 엔벨로프")
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
            data, freq_names=[freq_name], remove_shaft_orders=False, save=True
        )
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
    run_paper_reproduction()
    run_shaft_removal_comparison(all_data)

    # ------------------------------------------------------------------
    # 완료 요약
    # ------------------------------------------------------------------
    n_m1 = len(all_data) + len(fault_keys) + 1
    n_m2 = len(all_data) + len(fault_keys) + 1
    n_m1_vs_m2 = len(fault_keys)
    n_baseline = len(all_data) + len(fault_keys) + 1
    n_paper = len(PAPER_FIGURES)
    n_shaft_removal = 4  # OR021@6_1 / IR021_1 × (제거 전, 제거 후)
    n_total = (n_baseline + n_m1 + n_m2 + n_m1_vs_m2
               + n_paper + n_shaft_removal)

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
    print(f"    - 축 동기 성분 제거 비교: {n_shaft_removal}개")
    print(f"\n  총 이미지 파일            : {n_total}개")
    print("\n  결과 저장 위치: results/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
