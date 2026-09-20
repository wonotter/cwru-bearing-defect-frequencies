"""
CWRU 베어링 결함 주파수 검증 프로젝트 - 메인 실행 파이프라인

48kHz, 0.021", 1HP(1772 RPM) 조건에서 수집된 6개 데이터셋에 대해
FFT 및 엔벨로프 스펙트럼 분석을 수행하고, 결함 주파수 마커를 표시하여
BPFO/BPFI/BSF/FTF가 실제로 관측되는지 검증한다.

실행 방법:
    python main.py
"""

import matplotlib.pyplot as plt

from config import DATASETS
from data_loader import load_mat_file, load_all_datasets, print_dataset_summary
from signal_analysis import print_defect_frequencies
from visualization import (
    plot_analysis,
    plot_comparison,
    plot_all_envelope_overview,
    plot_method1_analysis,
    plot_method1_all_overview,
    plot_method1_comparison,
)


def main():
    print("\n" + "=" * 60)
    print("  CWRU 베어링 결함 주파수 검증 프로젝트")
    print("  48kHz | 0.021\" fault | 1HP (1772 RPM)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1단계: 결함 주파수 계산 결과 출력
    # ------------------------------------------------------------------
    rpm = 1772
    print_defect_frequencies(rpm)

    # ------------------------------------------------------------------
    # 2단계: 전체 데이터셋 로드
    # ------------------------------------------------------------------
    print("\n[1/4] 데이터셋 로딩 중...")
    all_data = load_all_datasets()
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

    # ------------------------------------------------------------------
    # 완료 요약
    # ------------------------------------------------------------------
    n_method1_individual = len(all_data)
    n_method1_comparison = len(fault_keys)
    n_total = (
        len(all_data) + len(fault_keys) + 1
        + n_method1_individual + n_method1_comparison + 1
    )

    print("\n" + "=" * 60)
    print("  분석 완료!")
    print("=" * 60)
    print(f"\n  === 기존 분석 (밴드패스 엔벨로프) ===")
    print(f"    - 개별 분석 플롯     : {len(all_data)}개")
    print(f"    - 비교 플롯          : {len(fault_keys)}개")
    print(f"    - 전체 오버뷰 플롯   : 1개")
    print(f"\n  === Method 1 (원신호 제곱 엔벨로프) ===")
    print(f"    - 개별 분석 플롯     : {n_method1_individual}개")
    print(f"    - 비교 플롯          : {n_method1_comparison}개")
    print(f"    - 전체 오버뷰 플롯   : 1개")
    print(f"\n  총 이미지 파일         : {n_total}개")
    print(f"\n  결과 저장 위치: results/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
