"""
CWRU 베어링 결함 주파수 검증 프로젝트 - 시각화 모듈

시간 영역 파형, FFT 스펙트럼, 엔벨로프 스펙트럼 플롯을 생성하고
결함 주파수 마커를 표시한다.
"""

import os

from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

from config import (
    RESULTS_DIR,
    DEFECT_FREQ_COLORS,
    FFT_PLOT_FREQ_MAX,
    ENVELOPE_PLOT_FREQ_MAX,
    PAPER_PLOT_FREQ_MAX,
    PAPER_N_HARMONICS,
    KURTOGRAM_MIN_BANDWIDTH,
    FIGURE_DPI,
)
from signal_analysis import (
    compute_fft,
    compute_bandpass_envelope_spectrum,
    compute_envelope_spectrum,
    compute_kurtogram,
    compute_paper_envelope_spectrum,
    envelope_band_from_kurtogram,
    compute_squared_envelope_spectrum_method1,
    compute_squared_envelope_spectrum_method2,
    calculate_defect_frequencies,
    refine_base_frequency,
)

# 결함 유형 → 대응하는 결함 주파수 이름
FAULT_TYPE_TO_FREQ = {
    "Inner Race": "BPFI",
    "Outer Race": "BPFO",
    "Ball": "BSF",
}

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
    fs = data["fs"]

    # 결함 주파수 계산
    defect_freqs = calculate_defect_frequencies(data["rpm"])

    # FFT / 엔벨로프 스펙트럼 계산
    fft_freqs, fft_mag = compute_fft(signal, fs)
    env_freqs, env_mag = compute_bandpass_envelope_spectrum(signal, fs)

    # 시간 축 생성
    time_axis = np.arange(len(signal)) / fs

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

    defect_freqs = calculate_defect_frequencies(fault_data["rpm"])

    # 엔벨로프 스펙트럼 계산
    norm_freqs, norm_mag = compute_bandpass_envelope_spectrum(
        normal_data["signal"], normal_data["fs"]
    )
    fault_freqs, fault_mag = compute_bandpass_envelope_spectrum(
        fault_data["signal"], fault_data["fs"]
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"엔벨로프 스펙트럼 비교: Normal vs {fault_key}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # 상단: Normal (마커는 정상 데이터 자신의 RPM 기준으로 계산)
    _plot_envelope_spectrum(
        axes[0], norm_freqs, norm_mag,
        calculate_defect_frequencies(normal_data["rpm"]),
        normal_data["info"], title=f"{normal_data['key']} (정상)"
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

    for i, key in enumerate(keys):
        data = all_data[key]
        env_freqs, env_mag = compute_bandpass_envelope_spectrum(
            data["signal"], data["fs"]
        )
        _plot_envelope_spectrum(
            axes[i], env_freqs, env_mag,
            calculate_defect_frequencies(data["rpm"]),
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
    primary_freq_name = FAULT_TYPE_TO_FREQ.get(info["fault_type"])

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
# Kurtogram
# =========================================================================

def _english_fault_label(info: dict) -> str:
    """Kurtogram 제목에 쓸 결함 설명을 영어로 만든다."""
    names = {
        "Normal": "Normal Bearing",
        "Inner Race": "Inner Race Fault",
        "Outer Race": "Outer Race Fault",
        "Ball": "Ball Fault",
    }
    label = names.get(info["fault_type"], info["fault_type"])
    diameter = info.get("fault_diameter_inch")
    if diameter:
        label = f"{label} {diameter:.3f} in"
    if info["fault_type"] == "Outer Race" and info.get("or_position"):
        clock = str(info["or_position"]).split(":", 1)[0]
        label = f"{label} at {clock} o'clock"
    return f"{label} ({info['load_hp']} HP, {info['rpm']} RPM)"


def plot_kurtogram(data: dict, save: bool = True) -> plt.Figure:
    """
    한 데이터셋의 Kurtogram을 그린다.

    가로축은 주파수, 세로축은 "얼마나 잘게 나눴는지"이다.
    색이 밝을수록 그 칸의 충격 점수가 높다.
    엔벨로프에 쓰라고 고른 칸은 검은 테두리로 표시한다.

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
    result = compute_kurtogram(data["signal"], data["fs"])
    rows = result["rows"]
    best = result["best"]
    nyquist = data["fs"] / 2.0

    all_scores = np.concatenate([row["kurtosis"] for row in rows])
    # 색 범위는 실제로 비교하는 칸(폭 1000 Hz 이상) 기준으로 잡는다.
    # 아주 좁은 칸 하나의 점수가 크면 나머지 칸이 전부 같은 색이 된다.
    wide_scores = np.concatenate([
        row["kurtosis"] for row in rows if row["bandwidth"] >= KURTOGRAM_MIN_BANDWIDTH
    ])
    color_norm = Normalize(
        vmin=float(min(0.0, np.min(all_scores))),
        vmax=float(np.max(wide_scores)),
    )
    colormap = plt.get_cmap("jet")

    fig, ax = plt.subplots(figsize=(10, 5))
    for row_index, row in enumerate(rows):
        bandwidth = row["bandwidth"]
        for center, score in zip(row["centers"], row["kurtosis"]):
            ax.add_patch(Rectangle(
                (center - bandwidth / 2.0, row_index),
                bandwidth,
                1.0,
                facecolor=colormap(color_norm(score)),
                edgecolor="white",
                linewidth=0.3,
            ))

    # 고른 칸만 검은 테두리
    best_row = next(index for index, row in enumerate(rows) if row["level"] == best["level"])
    ax.add_patch(Rectangle(
        (best["low"], best_row),
        best["bandwidth"],
        1.0,
        fill=False,
        edgecolor="black",
        linewidth=1.8,
    ))

    ax.set_xlim(0, nyquist)
    ax.set_ylim(0, len(rows))
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([
        f"Level {row['level']} ({row['bandwidth']:.0f} Hz)"
        for row in rows
    ])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Level (Bandwidth)")
    ax.set_title(
        f"Kurtogram: {data['key']} — {_english_fault_label(data['info'])}\n"
        f"K_max = {best['kurtosis']:.2f} at level {best['level']}, "
        f"Center Frequency = {best['center']:.0f} Hz, "
        f"Bandwidth = {best['bandwidth']:.0f} Hz",
        fontsize=11,
        fontweight="bold",
    )

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=color_norm, cmap=colormap),
        ax=ax,
        pad=0.02,
    )
    colorbar.set_label("Spectral Kurtosis")
    fig.tight_layout()

    if save:
        filename = f"kurtogram_{data['key'].replace('@', 'at')}.png"
        _save_figure(fig, filename)

    _print_kurtogram_summary(data["key"], result)
    return fig


def _print_kurtogram_summary(key: str, result: dict) -> None:
    """단계마다 가장 높은 칸과, 최종으로 고른 구간을 출력한다."""
    print(f"\n  [{key}] Kurtogram")
    print(f"    {'단계':>4} {'폭(Hz)':>10} {'최고 점수':>10} {'그 칸의 중심(Hz)':>16}")
    for row in result["rows"]:
        index = int(np.argmax(row["kurtosis"]))
        print(
            f"    {row['level']:4d} {row['bandwidth']:10.0f} "
            f"{row['kurtosis'][index]:10.2f} {row['centers'][index]:16.0f}"
        )
    best = result["best"]
    print(
        f"    → 엔벨로프에 쓸 구간: {best['low']:.0f}~{best['high']:.0f} Hz "
        f"(단계 {best['level']}, 점수 {best['kurtosis']:.2f})"
    )


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
    fs = data["fs"]

    defect_freqs = calculate_defect_frequencies(data["rpm"])
    env_freqs, env_mag, sq_envelope = (
        compute_squared_envelope_spectrum_method1(signal, fs)
    )
    time_axis = np.arange(len(signal)) / fs

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

    for i, key in enumerate(keys):
        data = all_data[key]
        env_freqs, env_mag, _ = compute_squared_envelope_spectrum_method1(
            data["signal"], data["fs"]
        )
        _plot_squared_envelope_spectrum(
            axes[i], env_freqs, env_mag,
            calculate_defect_frequencies(data["rpm"]),
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
    defect_freqs = calculate_defect_frequencies(fault_data["rpm"])

    norm_freqs, norm_mag, _ = compute_squared_envelope_spectrum_method1(
        normal_data["signal"], normal_data["fs"]
    )
    fault_freqs, fault_mag, _ = compute_squared_envelope_spectrum_method1(
        fault_data["signal"], fault_data["fs"]
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"Method 1 — 제곱 엔벨로프 스펙트럼 비교: Normal vs {fault_key}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    _plot_squared_envelope_spectrum(
        axes[0], norm_freqs, norm_mag,
        calculate_defect_frequencies(normal_data["rpm"]),
        normal_data["info"], title=f"{normal_data['key']} (정상)",
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


# =========================================================================
# Method 2: 켑스트럼 프리화이트닝 (Smith & Randall 2015, §5.2)
# =========================================================================

def plot_method2_analysis(data: dict, save: bool = True) -> plt.Figure:
    """
    논문 Method 2를 적용한 단일 데이터셋 분석 플롯을 생성한다.

    (a) 원신호 시간 영역 파형
    (b) 프리화이트닝된 신호 (시간 영역)
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
    fs = data["fs"]

    defect_freqs = calculate_defect_frequencies(data["rpm"])
    env_freqs, env_mag, sq_envelope, prewhitened = (
        compute_squared_envelope_spectrum_method2(signal, fs)
    )
    time_axis = np.arange(len(signal)) / fs

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(
        f"Method 2 (Cepstrum Prewhitening) — {key}  |  {info['description']}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # (a) 원신호 시간 영역
    _plot_time_domain(axes[0], time_axis, signal)

    # (b) 프리화이트닝된 신호
    _plot_prewhitened_time(axes[1], time_axis, prewhitened, defect_freqs)

    # (c) 제곱 엔벨로프 스펙트럼
    _plot_squared_envelope_spectrum(
        axes[2], env_freqs, env_mag, defect_freqs, info,
        title="(c) 제곱 엔벨로프 스펙트럼 (Method 2 — Prewhitening)",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        _save_figure(fig, f"method2_{key}.png")

    return fig


def plot_method2_all_overview(
    all_data: dict,
    save: bool = True,
) -> plt.Figure:
    """
    모든 데이터셋의 Method 2 제곱 엔벨로프 스펙트럼을 한 화면에 비교한다.

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
        "Method 2 (Cepstrum Prewhitening) — 전체 데이터셋 제곱 엔벨로프 스펙트럼 비교",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    for i, key in enumerate(keys):
        data = all_data[key]
        env_freqs, env_mag, _, _ = compute_squared_envelope_spectrum_method2(
            data["signal"], data["fs"]
        )
        _plot_squared_envelope_spectrum(
            axes[i], env_freqs, env_mag,
            calculate_defect_frequencies(data["rpm"]),
            data["info"],
            title=f"{key} — {data['info']['description']}",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save:
        _save_figure(fig, "method2_overview_all.png")

    return fig


def plot_method2_comparison(
    normal_data: dict,
    fault_data: dict,
    save: bool = True,
) -> plt.Figure:
    """
    Method 2 기준으로 정상 vs 결함 제곱 엔벨로프 스펙트럼을 비교한다.

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
    defect_freqs = calculate_defect_frequencies(fault_data["rpm"])

    norm_freqs, norm_mag, _, _ = compute_squared_envelope_spectrum_method2(
        normal_data["signal"], normal_data["fs"]
    )
    fault_freqs, fault_mag, _, _ = compute_squared_envelope_spectrum_method2(
        fault_data["signal"], fault_data["fs"]
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"Method 2 — 제곱 엔벨로프 스펙트럼 비교: Normal vs {fault_key}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    _plot_squared_envelope_spectrum(
        axes[0], norm_freqs, norm_mag,
        calculate_defect_frequencies(normal_data["rpm"]),
        normal_data["info"], title=f"{normal_data['key']} (정상)",
    )
    _plot_squared_envelope_spectrum(
        axes[1], fault_freqs, fault_mag, defect_freqs,
        fault_info, title=f"{fault_key} ({fault_info['description']})",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        _save_figure(fig, f"method2_comparison_Normal_vs_{fault_key}.png")

    return fig


def plot_method_comparison(
    data: dict,
    save: bool = True,
) -> plt.Figure:
    """
    단일 데이터셋에 대해 Method 1 vs Method 2 결과를 나란히 비교한다.

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
    fs = data["fs"]

    defect_freqs = calculate_defect_frequencies(data["rpm"])

    m1_freqs, m1_mag, _ = compute_squared_envelope_spectrum_method1(signal, fs)
    m2_freqs, m2_mag, _, _ = compute_squared_envelope_spectrum_method2(signal, fs)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"Method 1 vs Method 2 — {key}  |  {info['description']}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    _plot_squared_envelope_spectrum(
        axes[0], m1_freqs, m1_mag, defect_freqs, info,
        title="Method 1 (Raw Signal)",
    )
    _plot_squared_envelope_spectrum(
        axes[1], m2_freqs, m2_mag, defect_freqs, info,
        title="Method 2 (Cepstrum Prewhitening)",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        _save_figure(fig, f"m1_vs_m2_{key}.png")

    return fig


# =========================================================================
# Method 2 전용 내부 헬퍼
# =========================================================================

def _plot_prewhitened_time(
    ax: plt.Axes,
    time: np.ndarray,
    prewhitened: np.ndarray,
    defect_freqs: dict,
) -> None:
    """프리화이트닝된 신호의 시간 영역 파형을 그린다."""
    ax.plot(time, prewhitened, linewidth=0.3, color="#16A085")
    ax.set_title("(b) 프리화이트닝된 신호 (시간 영역)", fontsize=11)
    ax.set_xlabel("시간 (초)")
    ax.set_ylabel("진폭 (정규화)")
    ax.grid(True, alpha=0.3)

    shaft_freq = defect_freqs["shaft_freq"]
    shaft_period = 1.0 / shaft_freq
    if time[-1] > 5 * shaft_period:
        ax.set_xlim(0, 10 * shaft_period)


# =========================================================================
# 논문(Alonso-González et al., 2023) Fig.5~8 재현 플롯
# =========================================================================

_FAULT_CATEGORY_LABELS = {
    "Inner Race": "Inner Race Fault",
    "Outer Race": "Outer Race Fault",
    "Ball": "Ball Fault",
    "Normal": "Normal",
}


def _envelope_spectrum_title(
    key: str,
    info: dict,
    band: tuple[float, float] | None,
    remove_shaft_orders: bool,
) -> str:
    """
    엔벨로프 스펙트럼 그림의 영어 제목을 만든다.

    1행: Envelope Spectrum: {파일명} - {결함 종류} {결함 크기} ({부하}HP, {RPM} RPM)
    2행: Kurtogram {하한}~{상한} Hz  (밴드패스를 쓴 경우에만)

    RPM은 .mat 실측값이 아니라 config의 공칭값을 쓴다.
    """
    category = _FAULT_CATEGORY_LABELS.get(info["fault_type"], info["fault_type"])
    diameter = info.get("fault_diameter_inch")
    if diameter is not None:
        fault_text = f'{category} {diameter:.3f}"'
    else:
        fault_text = category

    # 외륜은 결함 위치(6시, 3시, 12시)가 파일마다 다르므로 제목에 포함한다.
    position = info.get("or_position")
    if position:
        clock = position.split(":", 1)[0]
        fault_text = f"{fault_text} @{clock}"

    title = (
        f"Envelope Spectrum: {key} - {fault_text} "
        f"({info['load_hp']}HP, {info['rpm']} RPM)"
    )
    if remove_shaft_orders:
        title += " [shaft orders removed]"
    if band is not None:
        low, high = band
        title += f"\nKurtogram {low:.0f}~{high:.0f} Hz"
    return title


def plot_paper_envelope_spectrum(
    data: dict,
    freq_names: list[str] | None = None,
    remove_shaft_orders: bool = False,
    refine_markers: bool = True,
    save: bool = True,
) -> plt.Figure:
    """
    논문 Fig.5~8 형식의 엔벨로프 스펙트럼 단일 플롯을 생성한다.

    논문 그림의 구성 요소를 그대로 따른다.
        - Kurtogram이 고른 구간을 Butterworth 밴드패스한 뒤 일반 엔벨로프(|Hilbert|)
        - 진폭 단위 g, x축 0 ~ PAPER_PLOT_FREQ_MAX(1000) Hz
        - 해당 결함 유형의 특성 주파수 고조파만 빨간 점선으로 표시

    Parameters
    ----------
    data : dict
        data_loader.load_mat_file()의 반환값
    freq_names : list[str] | None
        표시할 결함 주파수 이름 목록. None이면 데이터의 결함 유형에
        대응하는 주파수 하나만 표시하고, 정상 데이터는 BPFI/BPFO/BSF를
        모두 표시한다(논문 Fig.7과 동일).
    remove_shaft_orders : bool
        True이면 엔벨로프에서 축 회전 동기 성분(1×RPM 고조파)을 제거한다.
        BPFO 고조파가 축 성분에 묻히는 48kHz / 0.021" 데이터용 옵션이며,
        논문 재현 시에는 False로 둔다.
    refine_markers : bool
        True이면 1차 피크 위치로 기본 주파수를 보정해 고조파 마커를 찍는다.
        RPM 오차가 고조파 차수만큼 증폭되는 문제를 없애 준다.
        결함 유형과 일치하는 주파수에만 적용되며, 정상 데이터에는
        (스냅할 실제 피크가 없으므로) 적용하지 않는다.
    save : bool
        True이면 results/ 디렉토리에 이미지 저장

    Returns
    -------
    matplotlib.figure.Figure
    """
    signal = data["signal"]
    info = data["info"]
    key = data["key"]
    fs = data["fs"]
    rpm = data["rpm"]

    defect_freqs = calculate_defect_frequencies(
        rpm, n_harmonics=PAPER_N_HARMONICS
    )

    if remove_shaft_orders:
        env_freqs, env_mag, _ = compute_envelope_spectrum(
            signal, fs, band=None, squared=False,
            remove_shaft_orders=True,
            shaft_freq=defect_freqs["shaft_freq"],
        )
        band = None
    else:
        low, high = envelope_band_from_kurtogram(signal, fs)
        env_freqs, env_mag, _ = compute_paper_envelope_spectrum(
            signal, fs, band=(low, high)
        )
        band = (low, high)

    # 표시할 결함 주파수 결정
    primary = FAULT_TYPE_TO_FREQ.get(info["fault_type"])
    if freq_names is None:
        freq_names = [primary] if primary else ["BPFI", "BPFO", "BSF"]

    fig, ax = plt.subplots(figsize=(10, 5))

    freq_mask = env_freqs <= PAPER_PLOT_FREQ_MAX
    ax.plot(env_freqs[freq_mask], env_mag[freq_mask],
            linewidth=0.7, color="#1F77B4", zorder=2, label="Spectrum")

    # 마커 점선은 zorder를 낮춰 스펙트럼 피크를 가리지 않게 한다.
    for freq_name in freq_names:
        style = DEFECT_FREQ_COLORS[freq_name]
        base = defect_freqs[freq_name][0]

        # 실제 결함 유형과 일치하는 주파수만 보정한다.
        if refine_markers and freq_name == primary:
            base = refine_base_frequency(env_freqs, env_mag, base)

        for i in range(1, PAPER_N_HARMONICS + 1):
            freq = base * i
            if freq > PAPER_PLOT_FREQ_MAX:
                break
            ax.axvline(
                freq, color=style["color"], linestyle="--", linewidth=1.0,
                alpha=0.8, zorder=1,
                label=f"{freq_name} ({base:.1f} Hz)" if i == 1 else None,
            )

    ax.set_title(
        _envelope_spectrum_title(
            key, info, band, remove_shaft_orders
        ),
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("주파수 (Hz)")
    ax.set_ylabel("진폭 (g)")
    ax.set_xlim(0, PAPER_PLOT_FREQ_MAX)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()

    if save:
        filename = f"paper_envelope_{key.replace('@', 'at')}"
        filename += "_shaft_removed.png" if remove_shaft_orders else ".png"
        _save_figure(fig, filename)

    return fig


def print_harmonic_amplitudes(
    data: dict,
    freq_name: str,
    n_harmonics: int = PAPER_N_HARMONICS,
    tolerance_hz: float = 1.5,
) -> None:
    """
    논문 방식 엔벨로프 스펙트럼에서 결함 주파수 고조파의 진폭을 출력한다.

    논문 Fig.6과 수치를 직접 대조하거나, ML 특징값(AmplitudeBPFO 등)을
    추출할 때 사용한다.

    Parameters
    ----------
    data : dict
        data_loader.load_mat_file()의 반환값
    freq_name : str
        "BPFO", "BPFI", "BSF", "FTF" 중 하나
    n_harmonics : int
        확인할 고조파 개수
    tolerance_hz : float
        고조파 위치 주변에서 최대 피크를 찾을 허용 범위 (Hz)
    """
    env_freqs, env_mag, _ = compute_paper_envelope_spectrum(
        data["signal"], data["fs"]
    )
    defect_freqs = calculate_defect_frequencies(
        data["rpm"], n_harmonics=n_harmonics
    )
    shaft_freq = defect_freqs["shaft_freq"]
    theoretical_base = defect_freqs[freq_name][0]

    # 1차 피크 위치로 기본 주파수를 보정한다. 보정하지 않으면 RPM 오차가
    # 고조파 차수만큼 증폭되어 고차 고조파의 진폭이 0으로 잘못 측정된다.
    measured_base = refine_base_frequency(env_freqs, env_mag, theoretical_base)

    def peak_amplitude(target: float) -> tuple[float, float]:
        """target ±tolerance_hz 구간의 최대 피크 (주파수, 진폭)을 반환."""
        window = (env_freqs > target - tolerance_hz) & \
                 (env_freqs < target + tolerance_hz)
        if not window.any():
            return target, 0.0
        index = np.argmax(env_mag[window])
        return env_freqs[window][index], env_mag[window][index]

    _, shaft_amp = peak_amplitude(
        refine_base_frequency(env_freqs, env_mag, shaft_freq)
    )

    print(f"\n  [{data['key']}] {freq_name} 고조파 진폭 "
          f"(RPM={data['rpm']:.1f}, f_r={shaft_freq:.2f} Hz)")
    print(f"    이론 {freq_name} = {theoretical_base:.2f} Hz, "
          f"실측 보정값 = {measured_base:.2f} Hz")
    print(f"    {'차수':>4s} {'이론(Hz)':>10s} {'실측(Hz)':>10s} {'진폭(g)':>10s}")

    base_amp = 0.0
    for i in range(1, n_harmonics + 1):
        theoretical = theoretical_base * i
        if theoretical > PAPER_PLOT_FREQ_MAX:
            break
        found_freq, amplitude = peak_amplitude(measured_base * i)
        if i == 1:
            base_amp = amplitude
        print(f"    {i:>4d} {theoretical:>10.2f} {found_freq:>10.2f} "
              f"{amplitude:>10.4f}")

    if shaft_amp > 0:
        print(f"    → 1×RPM({shaft_freq:.1f} Hz) 진폭 = {shaft_amp:.4f} g, "
              f"{freq_name}/1×RPM 비율 = {base_amp / shaft_amp:.2f}")
