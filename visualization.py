"""
CWRU 베어링 결함 주파수 검증 프로젝트 - 시각화 모듈

시간 영역 파형, FFT 스펙트럼, 엔벨로프 스펙트럼 플롯을 생성하고
결함 주파수 마커를 표시한다.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from config import (
    SAMPLING_RATE,
    RESULTS_DIR,
    DEFECT_FREQ_COLORS,
    FFT_PLOT_FREQ_MAX,
    ENVELOPE_PLOT_FREQ_MAX,
    FIGURE_DPI,
)
from signal_analysis import (
    compute_fft,
    compute_envelope_spectrum,
    compute_squared_envelope_spectrum_method1,
    calculate_defect_frequencies,
)

# 한글 폰트 설정 (Windows: Malgun Gothic)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# =========================================================================
# 개별 데이터셋 3종 플롯 (시간 영역 + FFT + 엔벨로프)
# =========================================================================

def plot_analysis(data: dict, save: bool = True) -> plt.Figure:
    """
    단일 데이터셋에 대해 3종 분석 플롯을 생성한다.

    (a) 시간 영역 파형
    (b) FFT 스펙트럼 + 결함 주파수 마커
    (c) 엔벨로프 스펙트럼 + 결함 주파수 마커

    Parameters
    ----------
    data : dict
        data_loader.load_mat_file()의 반환값
    save : bool
        True이면 results/ 디렉토리에 이미지 저장

    Returns
    -------
    matplotlib.figure.Figure
    """
    signal = data["signal"]
    info = data["info"]
    key = data["key"]
    rpm = info["rpm"]

    # 결함 주파수 계산
    defect_freqs = calculate_defect_frequencies(rpm)

    # FFT / 엔벨로프 스펙트럼 계산
    fft_freqs, fft_mag = compute_fft(signal)
    env_freqs, env_mag = compute_envelope_spectrum(signal)

    # 시간 축 생성
    time_axis = np.arange(len(signal)) / SAMPLING_RATE

    # Figure 생성: 3행 1열
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(
        f"{key}  —  {info['description']}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # (a) 시간 영역 파형
    _plot_time_domain(axes[0], time_axis, signal)

    # (b) FFT 스펙트럼
    _plot_fft_spectrum(axes[1], fft_freqs, fft_mag, defect_freqs, info)

    # (c) 엔벨로프 스펙트럼
    _plot_envelope_spectrum(axes[2], env_freqs, env_mag, defect_freqs, info)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        _save_figure(fig, f"{key}_analysis.png")

    return fig


# =========================================================================
# Normal vs 결함 비교 플롯
# =========================================================================

def plot_comparison(
    normal_data: dict,
    fault_data: dict,
    save: bool = True,
) -> plt.Figure:
    """
    정상(Normal) 데이터와 결함 데이터의 엔벨로프 스펙트럼을 비교한다.

    Parameters
    ----------
    normal_data : dict
        Normal 데이터셋 (load_mat_file 반환값)
    fault_data : dict
        결함 데이터셋 (load_mat_file 반환값)
    save : bool
        True이면 results/ 디렉토리에 이미지 저장

    Returns
    -------
    matplotlib.figure.Figure
    """
    fault_info = fault_data["info"]
    fault_key = fault_data["key"]
    rpm = fault_info["rpm"]

    defect_freqs = calculate_defect_frequencies(rpm)

    # 엔벨로프 스펙트럼 계산
    norm_freqs, norm_mag = compute_envelope_spectrum(normal_data["signal"])
    fault_freqs, fault_mag = compute_envelope_spectrum(fault_data["signal"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"엔벨로프 스펙트럼 비교: Normal vs {fault_key}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # 상단: Normal
    _plot_envelope_spectrum(
        axes[0], norm_freqs, norm_mag, defect_freqs,
        normal_data["info"], title="Normal_1 (정상)"
    )

    # 하단: 결함
    _plot_envelope_spectrum(
        axes[1], fault_freqs, fault_mag, defect_freqs,
        fault_info, title=f"{fault_key} ({fault_info['description']})"
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        _save_figure(fig, f"comparison_Normal_vs_{fault_key}.png")

    return fig


# =========================================================================
# 전체 결함 유형 한눈에 보기 플롯
# =========================================================================

def plot_all_envelope_overview(
    all_data: dict,
    save: bool = True,
) -> plt.Figure:
    """
    모든 데이터셋의 엔벨로프 스펙트럼을 한 화면에 비교한다.

    Parameters
    ----------
    all_data : dict
        {dataset_key: load_mat_file() 반환값, ...}
    save : bool
        True이면 results/ 디렉토리에 이미지 저장

    Returns
    -------
    matplotlib.figure.Figure
    """
    keys = list(all_data.keys())
    n = len(keys)

    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    fig.suptitle(
        "전체 데이터셋 엔벨로프 스펙트럼 비교 (48kHz, 0.021\", 1HP)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    rpm = list(all_data.values())[0]["info"]["rpm"]
    defect_freqs = calculate_defect_frequencies(rpm)

    for i, key in enumerate(keys):
        data = all_data[key]
        env_freqs, env_mag = compute_envelope_spectrum(data["signal"])
        _plot_envelope_spectrum(
            axes[i], env_freqs, env_mag, defect_freqs,
            data["info"],
            title=f"{key} — {data['info']['description']}",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save:
        _save_figure(fig, "overview_all_envelope.png")

    return fig


# =========================================================================
# 내부 플롯 헬퍼 함수
# =========================================================================

def _plot_time_domain(
    ax: plt.Axes,
    time: np.ndarray,
    signal: np.ndarray,
) -> None:
    """시간 영역 파형을 그린다."""
    ax.plot(time, signal, linewidth=0.3, color="#2C3E50")
    ax.set_title("(a) 시간 영역 파형", fontsize=11)
    ax.set_xlabel("시간 (초)")
    ax.set_ylabel("가속도 (g)")
    ax.grid(True, alpha=0.3)


def _plot_fft_spectrum(
    ax: plt.Axes,
    freqs: np.ndarray,
    magnitude: np.ndarray,
    defect_freqs: dict,
    info: dict,
    title: str | None = None,
) -> None:
    """FFT 스펙트럼에 결함 주파수 마커를 표시한다."""
    freq_mask = freqs <= FFT_PLOT_FREQ_MAX
    ax.plot(freqs[freq_mask], magnitude[freq_mask],
            linewidth=0.5, color="#2C3E50")

    _add_defect_markers(ax, defect_freqs, info)

    ax.set_title(title or "(b) FFT 스펙트럼", fontsize=11)
    ax.set_xlabel("주파수 (Hz)")
    ax.set_ylabel("진폭")
    ax.set_xlim(0, FFT_PLOT_FREQ_MAX)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7, ncol=2)


def _plot_envelope_spectrum(
    ax: plt.Axes,
    freqs: np.ndarray,
    magnitude: np.ndarray,
    defect_freqs: dict,
    info: dict,
    title: str | None = None,
) -> None:
    """엔벨로프 스펙트럼에 결함 주파수 마커를 표시한다."""
    freq_mask = freqs <= ENVELOPE_PLOT_FREQ_MAX
    ax.plot(freqs[freq_mask], magnitude[freq_mask],
            linewidth=0.5, color="#2C3E50")

    _add_defect_markers(ax, defect_freqs, info)

    ax.set_title(title or "(c) 엔벨로프 스펙트럼", fontsize=11)
    ax.set_xlabel("주파수 (Hz)")
    ax.set_ylabel("엔벨로프 진폭")
    ax.set_xlim(0, ENVELOPE_PLOT_FREQ_MAX)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7, ncol=2)


def _add_defect_markers(
    ax: plt.Axes,
    defect_freqs: dict,
    info: dict,
) -> None:
    """
    결함 주파수 수직선 마커를 플롯에 추가한다.

    각 결함 유형별로 기본 주파수 + 고조파를 점선으로 표시하며,
    해당 데이터의 결함 유형에 맞는 주파수는 굵게 강조한다.
    """
    fault_type = info["fault_type"]

    # 결함 유형 → 대응하는 주파수 이름 매핑
    fault_to_freq = {
        "Inner Race": "BPFI",
        "Outer Race": "BPFO",
        "Ball": "BSF",
    }
    primary_freq_name = fault_to_freq.get(fault_type)

    # 1×RPM 마커
    shaft_freq = defect_freqs["shaft_freq"]
    rpm_style = DEFECT_FREQ_COLORS["1xRPM"]
    ax.axvline(shaft_freq, color=rpm_style["color"],
               linestyle=":", linewidth=0.8, alpha=0.6,
               label=f"1×RPM ({shaft_freq:.1f} Hz)")

    # 각 결함 주파수 마커
    for freq_name in ["BPFO", "BPFI", "BSF", "FTF"]:
        harmonics = defect_freqs[freq_name]
        style = DEFECT_FREQ_COLORS[freq_name]
        is_primary = (freq_name == primary_freq_name)

        for i, freq in enumerate(harmonics):
            lw = 1.5 if is_primary else 0.7
            alpha = 0.9 if is_primary else 0.4
            label = (f"{style['label']} ({harmonics[0]:.1f} Hz)"
                     if i == 0 else None)

            ax.axvline(freq, color=style["color"],
                       linestyle="--", linewidth=lw, alpha=alpha,
                       label=label)


def _save_figure(fig: plt.Figure, filename: str) -> None:
    """Figure를 results/ 디렉토리에 저장한다."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, filename)
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"  [저장] {filepath}")


# =========================================================================
# Method 1: 원신호 제곱 엔벨로프 스펙트럼 (Smith & Randall 2015, §5.1)
# =========================================================================

def plot_method1_analysis(data: dict, save: bool = True) -> plt.Figure:
    """
    논문 Method 1을 적용한 단일 데이터셋 분석 플롯을 생성한다.

    (a) 원신호 시간 영역 파형
    (b) 제곱 엔벨로프 — 시간 영역
    (c) 제곱 엔벨로프 스펙트럼 + 결함 주파수 마커

    Parameters
    ----------
    data : dict
        data_loader.load_mat_file()의 반환값
    save : bool
        True이면 results/ 디렉토리에 이미지 저장

    Returns
    -------
    matplotlib.figure.Figure
    """
    signal = data["signal"]
    info = data["info"]
    key = data["key"]
    rpm = info["rpm"]

    defect_freqs = calculate_defect_frequencies(rpm)
    env_freqs, env_mag, sq_envelope = (
        compute_squared_envelope_spectrum_method1(signal)
    )
    time_axis = np.arange(len(signal)) / SAMPLING_RATE

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(
        f"Method 1 — {key}  |  {info['description']}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # (a) 원신호 시간 영역
    _plot_time_domain(axes[0], time_axis, signal)

    # (b) 제곱 엔벨로프 시간 영역
    _plot_squared_envelope_time(axes[1], time_axis, sq_envelope, defect_freqs)

    # (c) 제곱 엔벨로프 스펙트럼
    _plot_squared_envelope_spectrum(
        axes[2], env_freqs, env_mag, defect_freqs, info,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        _save_figure(fig, f"method1_{key}.png")

    return fig


def plot_method1_all_overview(
    all_data: dict,
    save: bool = True,
) -> plt.Figure:
    """
    모든 데이터셋의 Method 1 제곱 엔벨로프 스펙트럼을 한 화면에 비교한다.

    Parameters
    ----------
    all_data : dict
        {dataset_key: load_mat_file() 반환값, ...}
    save : bool
        True이면 results/ 디렉토리에 이미지 저장

    Returns
    -------
    matplotlib.figure.Figure
    """
    keys = list(all_data.keys())
    n = len(keys)

    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    fig.suptitle(
        "Method 1 — 전체 데이터셋 제곱 엔벨로프 스펙트럼 비교\n"
        "(원신호 → 힐베르트 → 제곱 엔벨로프 → FFT, 필터 없음)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    rpm = list(all_data.values())[0]["info"]["rpm"]
    defect_freqs = calculate_defect_frequencies(rpm)

    for i, key in enumerate(keys):
        data = all_data[key]
        env_freqs, env_mag, _ = compute_squared_envelope_spectrum_method1(
            data["signal"]
        )
        _plot_squared_envelope_spectrum(
            axes[i], env_freqs, env_mag, defect_freqs,
            data["info"],
            title=f"{key} — {data['info']['description']}",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save:
        _save_figure(fig, "method1_overview_all.png")

    return fig


def plot_method1_comparison(
    normal_data: dict,
    fault_data: dict,
    save: bool = True,
) -> plt.Figure:
    """
    Method 1 기준으로 정상 vs 결함 제곱 엔벨로프 스펙트럼을 비교한다.

    Parameters
    ----------
    normal_data : dict
        Normal 데이터셋 (load_mat_file 반환값)
    fault_data : dict
        결함 데이터셋 (load_mat_file 반환값)
    save : bool
        True이면 results/ 디렉토리에 이미지 저장

    Returns
    -------
    matplotlib.figure.Figure
    """
    fault_info = fault_data["info"]
    fault_key = fault_data["key"]
    rpm = fault_info["rpm"]
    defect_freqs = calculate_defect_frequencies(rpm)

    norm_freqs, norm_mag, _ = compute_squared_envelope_spectrum_method1(
        normal_data["signal"]
    )
    fault_freqs, fault_mag, _ = compute_squared_envelope_spectrum_method1(
        fault_data["signal"]
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"Method 1 — 제곱 엔벨로프 스펙트럼 비교: Normal vs {fault_key}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    _plot_squared_envelope_spectrum(
        axes[0], norm_freqs, norm_mag, defect_freqs,
        normal_data["info"], title="Normal_1 (정상)",
    )
    _plot_squared_envelope_spectrum(
        axes[1], fault_freqs, fault_mag, defect_freqs,
        fault_info, title=f"{fault_key} ({fault_info['description']})",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        _save_figure(fig, f"method1_comparison_Normal_vs_{fault_key}.png")

    return fig


# =========================================================================
# Method 1 전용 내부 헬퍼
# =========================================================================

def _plot_squared_envelope_time(
    ax: plt.Axes,
    time: np.ndarray,
    sq_envelope: np.ndarray,
    defect_freqs: dict,
) -> None:
    """제곱 엔벨로프의 시간 영역 파형을 그린다."""
    ax.plot(time, sq_envelope, linewidth=0.3, color="#8E44AD")
    ax.set_title("(b) 제곱 엔벨로프 (시간 영역)", fontsize=11)
    ax.set_xlabel("시간 (초)")
    ax.set_ylabel("진폭²")
    ax.grid(True, alpha=0.3)

    shaft_freq = defect_freqs["shaft_freq"]
    shaft_period = 1.0 / shaft_freq
    if time[-1] > 5 * shaft_period:
        ax.set_xlim(0, 10 * shaft_period)


def _plot_squared_envelope_spectrum(
    ax: plt.Axes,
    freqs: np.ndarray,
    magnitude: np.ndarray,
    defect_freqs: dict,
    info: dict,
    title: str | None = None,
) -> None:
    """제곱 엔벨로프 스펙트럼에 결함 주파수 마커를 표시한다."""
    freq_mask = freqs <= ENVELOPE_PLOT_FREQ_MAX
    ax.plot(freqs[freq_mask], magnitude[freq_mask],
            linewidth=0.5, color="#8E44AD")

    _add_defect_markers(ax, defect_freqs, info)

    ax.set_title(
        title or "(c) 제곱 엔벨로프 스펙트럼 (Method 1)",
        fontsize=11,
    )
    ax.set_xlabel("주파수 (Hz)")
    ax.set_ylabel("진폭²")
    ax.set_xlim(0, ENVELOPE_PLOT_FREQ_MAX)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
