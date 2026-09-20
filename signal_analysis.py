"""
CWRU 베어링 결함 주파수 검증 프로젝트 - 신호 분석 모듈

FFT 스펙트럼, 엔벨로프 스펙트럼 분석, 결함 주파수 계산 등
핵심 신호 처리 함수를 제공한다.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from config import (
    SAMPLING_RATE,
    DEFECT_FREQ_MULTIPLIERS,
    ENVELOPE_BANDPASS_LOW,
    ENVELOPE_BANDPASS_HIGH,
    ENVELOPE_FILTER_ORDER,
    N_HARMONICS,
)


# =========================================================================
# FFT 분석
# =========================================================================

def compute_fft(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    실수 신호의 단측 FFT(진폭 스펙트럼)를 계산한다.

    Parameters
    ----------
    signal : np.ndarray
        시간 영역 신호 (1D)

    Returns
    -------
    freqs : np.ndarray
        주파수 축 (Hz)
    magnitude : np.ndarray
        진폭 스펙트럼 (정규화됨: 2/N)
    """
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLING_RATE)
    fft_values = np.fft.rfft(signal)
    magnitude = (2.0 / n) * np.abs(fft_values)
    # DC 성분은 정규화 계수 1/N 적용
    magnitude[0] /= 2.0
    return freqs, magnitude


# =========================================================================
# 엔벨로프 스펙트럼 분석
# =========================================================================

def compute_squared_envelope_spectrum_method1(
    signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    논문 Method 1: 원신호의 제곱 엔벨로프 스펙트럼을 계산한다.

    Smith & Randall (2015), Section 5.1 — "Envelope analysis of the raw signal"
    밴드패스 필터 없이, 원신호 전체 대역(full bandwidth)에 대해
    제곱 엔벨로프 스펙트럼(squared envelope spectrum)을 구한다.

    분석 파이프라인:
        1. 원신호 → 힐베르트 변환 → 해석 신호(analytic signal) 생성
        2. |해석 신호|² → 제곱 엔벨로프(squared envelope) 추출
        3. DC 성분 제거 (평균 차감)
        4. 제곱 엔벨로프에 FFT 적용 → 제곱 엔벨로프 스펙트럼

    Parameters
    ----------
    signal : np.ndarray
        시간 영역 진동 신호 (1D, 원신호 그대로)

    Returns
    -------
    freqs : np.ndarray
        주파수 축 (Hz)
    sq_env_magnitude : np.ndarray
        제곱 엔벨로프 스펙트럼 진폭
    squared_envelope : np.ndarray
        시간 영역의 제곱 엔벨로프 (시간 영역 플롯에 사용)
    """
    # 1단계: 힐베르트 변환으로 해석 신호 생성 (필터링 없이 원신호 직접 사용)
    analytic_signal = hilbert(signal)

    # 2단계: 제곱 엔벨로프 추출 — envelope² = |analytic_signal|²
    envelope = np.abs(analytic_signal)
    squared_envelope = envelope ** 2

    # 3단계: DC 성분 제거 (평균 차감)
    squared_envelope_ac = squared_envelope - np.mean(squared_envelope)

    # 4단계: 제곱 엔벨로프에 FFT 적용
    freqs, sq_env_magnitude = compute_fft(squared_envelope_ac)

    return freqs, sq_env_magnitude, squared_envelope


def compute_envelope_spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    밴드패스 엔벨로프 스펙트럼을 계산한다 (기존 방식 유지).

    주의: 이 함수는 밴드패스 필터(2~5 kHz)를 적용하므로
    논문의 Method 1과는 다르다. Method 1은
    compute_squared_envelope_spectrum_method1()을 사용할 것.

    분석 파이프라인:
        1. 밴드패스 필터 (Butterworth, 2~5 kHz)
           → 베어링 하우징 공진 대역 분리
        2. 힐베르트 변환 → 해석 신호(analytic signal) 생성
        3. |해석 신호| → 엔벨로프 추출, DC 제거
        4. 엔벨로프에 FFT 적용 → 엔벨로프 스펙트럼

    Parameters
    ----------
    signal : np.ndarray
        시간 영역 진동 신호 (1D)

    Returns
    -------
    freqs : np.ndarray
        주파수 축 (Hz)
    envelope_magnitude : np.ndarray
        엔벨로프 스펙트럼 진폭
    """
    # 1단계: 밴드패스 필터 (Butterworth SOS)
    filtered = _bandpass_filter(signal)

    # 2단계: 힐베르트 변환으로 해석 신호 생성
    analytic_signal = hilbert(filtered)

    # 3단계: 엔벨로프 추출 및 DC 제거
    envelope = np.abs(analytic_signal)
    envelope = envelope - np.mean(envelope)

    # 4단계: 엔벨로프에 FFT 적용
    freqs, envelope_magnitude = compute_fft(envelope)

    return freqs, envelope_magnitude


def _bandpass_filter(signal: np.ndarray) -> np.ndarray:
    """
    Butterworth 밴드패스 필터를 적용한다.

    SOS(Second-Order Sections) 형식을 사용하여 수치적 안정성을 확보하고,
    sosfiltfilt(영위상 필터)로 위상 왜곡 없이 필터링한다.

    Parameters
    ----------
    signal : np.ndarray
        입력 신호 (1D)

    Returns
    -------
    np.ndarray
        밴드패스 필터링된 신호
    """
    nyquist = SAMPLING_RATE / 2.0
    low = ENVELOPE_BANDPASS_LOW / nyquist
    high = ENVELOPE_BANDPASS_HIGH / nyquist

    sos = butter(
        N=ENVELOPE_FILTER_ORDER,
        Wn=[low, high],
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(sos, signal)


# =========================================================================
# 결함 주파수 계산
# =========================================================================

def calculate_defect_frequencies(rpm: int) -> dict:
    """
    주어진 RPM에서 각 결함 유형의 주파수와 고조파를 계산한다.

    Parameters
    ----------
    rpm : int
        모터 회전 속도 (RPM)

    Returns
    -------
    dict
        {
            "shaft_freq": 축 회전 주파수 (Hz),
            "BPFI": [1차, 2차, ..., N차 고조파 주파수 리스트],
            "BPFO": [...],
            "BSF":  [...],
            "FTF":  [...],
        }
    """
    shaft_freq = rpm / 60.0

    result = {"shaft_freq": shaft_freq}
    for name, multiplier in DEFECT_FREQ_MULTIPLIERS.items():
        base_freq = multiplier * shaft_freq
        harmonics = [base_freq * (i + 1) for i in range(N_HARMONICS)]
        result[name] = harmonics

    return result


def print_defect_frequencies(rpm: int) -> None:
    """
    결함 주파수 계산 결과를 콘솔에 보기 좋게 출력한다.

    Parameters
    ----------
    rpm : int
        모터 회전 속도 (RPM)
    """
    freqs = calculate_defect_frequencies(rpm)
    shaft = freqs["shaft_freq"]

    print(f"\n{'=' * 60}")
    print(f"  결함 주파수 계산 결과 (RPM={rpm}, f_r={shaft:.3f} Hz)")
    print(f"{'=' * 60}")

    for name in ["BPFO", "BPFI", "BSF", "FTF"]:
        multiplier = DEFECT_FREQ_MULTIPLIERS[name]
        harmonics = freqs[name]
        print(f"\n  {name} (배수: {multiplier})")
        for i, freq in enumerate(harmonics):
            label = "기본" if i == 0 else f"{i + 1}차"
            print(f"    {label:>4s}: {freq:8.2f} Hz")

    print(f"\n  축 회전 주파수 (1×RPM): {shaft:.3f} Hz")
    print(f"{'=' * 60}\n")
