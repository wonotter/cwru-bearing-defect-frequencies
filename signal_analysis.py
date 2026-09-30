"""
CWRU 베어링 결함 주파수 검증 프로젝트 - 신호 분석 모듈

FFT 스펙트럼, 엔벨로프 스펙트럼 분석, 결함 주파수 계산 등
핵심 신호 처리 함수를 제공한다.

샘플링 레이트는 데이터셋 그룹(12kHz / 48kHz)에 따라 다르므로
모든 함수가 fs를 명시적 인자로 받는다.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, hilbert, sosfiltfilt

from config import (
    DEFECT_FREQ_MULTIPLIERS,
    ENVELOPE_BANDPASS_LOW,
    ENVELOPE_BANDPASS_HIGH,
    ENVELOPE_FILTER_ORDER,
    KURTOGRAM_MIN_BANDWIDTH,
    KURTOGRAM_NLEVEL,
    N_HARMONICS,
    PAPER_FILTER_ORDER,
)


# =========================================================================
# FFT 분석
# =========================================================================

def compute_fft(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    실수 신호의 단측 FFT(진폭 스펙트럼)를 계산한다.

    Parameters
    ----------
    signal : np.ndarray
        시간 영역 신호 (1D)
    fs : float
        샘플링 레이트 (Hz)

    Returns
    -------
    freqs : np.ndarray
        주파수 축 (Hz)
    magnitude : np.ndarray
        진폭 스펙트럼 (정규화됨: 2/N)
    """
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_values = np.fft.rfft(signal)
    magnitude = (2.0 / n) * np.abs(fft_values)
    # DC 성분은 정규화 계수 1/N 적용
    magnitude[0] /= 2.0
    return freqs, magnitude


# =========================================================================
# 엔벨로프 추출 공통 엔진
# =========================================================================

def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low: float,
    high: float,
    order: int = ENVELOPE_FILTER_ORDER,
) -> np.ndarray:
    """
    Butterworth 밴드패스 필터를 적용한다.

    SOS(Second-Order Sections) 형식을 사용하여 수치적 안정성을 확보하고,
    sosfiltfilt(영위상 필터)로 위상 왜곡 없이 필터링한다.

    Parameters
    ----------
    signal : np.ndarray
        입력 신호 (1D)
    fs : float
        샘플링 레이트 (Hz)
    low, high : float
        통과 대역의 하한/상한 (Hz)
    order : int
        Butterworth 필터 차수

    Returns
    -------
    np.ndarray
        밴드패스 필터링된 신호

    Raises
    ------
    ValueError
        통과 대역이 [0, fs/2] 범위를 벗어나거나 low >= high 일 때
    """
    nyquist = fs / 2.0
    if not (0 < low < high < nyquist):
        raise ValueError(
            f"유효하지 않은 통과 대역입니다: low={low}, high={high}, "
            f"나이퀴스트={nyquist}"
        )

    sos = butter(
        N=order,
        Wn=[low / nyquist, high / nyquist],
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(sos, signal)


def extract_envelope(
    signal: np.ndarray,
    fs: float,
    band: tuple[float, float] | None = None,
    squared: bool = False,
) -> np.ndarray:
    """
    힐베르트 변환으로 신호의 엔벨로프를 추출한다.

    Parameters
    ----------
    signal : np.ndarray
        시간 영역 진동 신호 (1D)
    fs : float
        샘플링 레이트 (Hz)
    band : tuple[float, float] | None
        (하한, 상한) Hz. None이면 밴드패스 없이 전대역을 사용한다.
    squared : bool
        True이면 제곱 엔벨로프(|analytic|²), False이면 일반 엔벨로프(|analytic|)

    Returns
    -------
    np.ndarray
        엔벨로프 (DC 성분 미제거 상태)
    """
    if band is not None:
        signal = bandpass_filter(signal, fs, band[0], band[1])

    envelope = np.abs(hilbert(signal))
    return envelope ** 2 if squared else envelope


def remove_shaft_synchronous_component(
    envelope: np.ndarray,
    fs: float,
    shaft_freq: float,
    samples_per_rev: int = 512,
) -> tuple[np.ndarray, float]:
    """
    엔벨로프에서 축 회전에 동기된 성분을 제거한다(회전 동기 평균 차감).

    원리:
        엔벨로프를 각도 영역(1회전 = samples_per_rev 샘플)으로 리샘플한 뒤
        회전별로 쌓아 평균하면, 축 회전에 동기된 성분(1×RPM 및 그 고조파)만
        남는다. 이 평균 파형을 각 회전에서 차감하면 축 동기 성분이 사라지고
        베어링 결함 성분(BPFO 등 축 회전수의 비정수배)만 남는다.

    용도:
        BPFO 고조파가 1×RPM 고조파에 묻히는 경우(예: 48kHz / 0.021" 데이터)
        축 성분을 제거하여 BPFO 계열의 가시성을 높인다.

    한계:
        BPFO ± n×(축 회전수) 측대역(sideband)은 축 동기 성분이 아니므로
        이 방법으로 제거되지 않는다.

    Parameters
    ----------
    envelope : np.ndarray
        시간 영역 엔벨로프 (1D)
    fs : float
        엔벨로프의 샘플링 레이트 (Hz)
    shaft_freq : float
        축 회전 주파수 (Hz)
    samples_per_rev : int
        각도 영역 리샘플 해상도 (1회전당 샘플 수)

    Returns
    -------
    residual : np.ndarray
        축 동기 성분이 제거된 엔벨로프 (각도 영역 등간격 샘플)
    fs_resampled : float
        residual의 유효 샘플링 레이트 (Hz) = samples_per_rev × shaft_freq
        → 이 값으로 FFT하면 주파수 축이 다시 Hz 단위가 된다.

    Raises
    ------
    ValueError
        신호 길이가 2회전 미만이어서 동기 평균을 낼 수 없을 때
    """
    duration_s = len(envelope) / fs
    n_revolutions = int(duration_s * shaft_freq)
    if n_revolutions < 2:
        raise ValueError(
            f"회전 동기 평균에는 최소 2회전이 필요합니다 "
            f"(현재 {n_revolutions}회전)"
        )

    # 1단계: 시간 영역 → 각도 영역 리샘플 (선형 보간)
    time_axis = np.arange(len(envelope)) / fs
    interpolator = interp1d(
        time_axis, envelope, kind="linear", fill_value="extrapolate"
    )
    fs_resampled = samples_per_rev * shaft_freq
    angle_times = np.arange(n_revolutions * samples_per_rev) / fs_resampled
    resampled = interpolator(angle_times).reshape(n_revolutions, samples_per_rev)

    # 2단계: 회전 동기 평균(축 동기 성분) 계산 후 차감
    synchronous_average = resampled.mean(axis=0)
    residual = (resampled - synchronous_average).ravel()

    return residual, fs_resampled


def compute_envelope_spectrum(
    signal: np.ndarray,
    fs: float,
    band: tuple[float, float] | None = None,
    squared: bool = False,
    remove_shaft_orders: bool = False,
    shaft_freq: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    엔벨로프 스펙트럼을 계산하는 공통 함수.

    분석 파이프라인:
        1. (선택) 밴드패스 필터 → 공진 대역 분리
        2. 힐베르트 변환 → 해석 신호 → 엔벨로프 추출
        3. (선택) 회전 동기 성분 제거
        4. DC 성분 제거 (평균 차감)
        5. FFT → 엔벨로프 스펙트럼

    Parameters
    ----------
    signal : np.ndarray
        시간 영역 진동 신호 (1D)
    fs : float
        샘플링 레이트 (Hz)
    band : tuple[float, float] | None
        밴드패스 통과 대역 (Hz). None이면 전대역.
    squared : bool
        True이면 제곱 엔벨로프 스펙트럼(SES), False이면 일반 엔벨로프 스펙트럼
    remove_shaft_orders : bool
        True이면 엔벨로프에서 축 회전 동기 성분을 제거한다.
    shaft_freq : float | None
        축 회전 주파수 (Hz). remove_shaft_orders=True일 때 필수.

    Returns
    -------
    freqs : np.ndarray
        주파수 축 (Hz)
    magnitude : np.ndarray
        엔벨로프 스펙트럼 진폭
    envelope : np.ndarray
        시간 영역 엔벨로프 (플롯용, DC 미제거 원본)

    Raises
    ------
    ValueError
        remove_shaft_orders=True인데 shaft_freq가 주어지지 않았을 때
    """
    envelope = extract_envelope(signal, fs, band=band, squared=squared)

    spectrum_input, spectrum_fs = envelope, fs
    if remove_shaft_orders:
        if shaft_freq is None:
            raise ValueError(
                "remove_shaft_orders=True인 경우 shaft_freq를 지정해야 합니다."
            )
        spectrum_input, spectrum_fs = remove_shaft_synchronous_component(
            envelope - envelope.mean(), fs, shaft_freq
        )

    freqs, magnitude = compute_fft(
        spectrum_input - spectrum_input.mean(), spectrum_fs
    )
    return freqs, magnitude, envelope


# =========================================================================
# 논문(Alonso-González et al., 2023) Fig.5~8 재현 방식
# =========================================================================

def compute_paper_envelope_spectrum(
    signal: np.ndarray,
    fs: float,
    band: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    논문 형식의 엔벨로프 스펙트럼을 계산한다.

    처리:
        1. Kurtogram에서 충격 점수가 가장 높은 칸을 고른다.
           band를 넘기면 그 구간을 그대로 쓴다.
        2. 그 구간만 Butterworth 밴드패스로 통과시킨다.
        3. 힐베르트 변환 → 일반 엔벨로프 |analytic| (제곱하지 않음)
        4. DC 제거 후 FFT → 진폭 스펙트럼 (단위 g)

    Parameters
    ----------
    signal : np.ndarray
        시간 영역 진동 신호 (1D)
    fs : float
        샘플링 레이트 (Hz)
    band : tuple[float, float] | None
        (하한, 상한) Hz. None이면 Kurtogram이 고른 칸을 사용한다.

    Returns
    -------
    freqs : np.ndarray
        주파수 축 (Hz)
    magnitude : np.ndarray
        엔벨로프 스펙트럼 진폭 (g)
    envelope : np.ndarray
        시간 영역 엔벨로프
    """
    if band is None:
        band = envelope_band_from_kurtogram(signal, fs)
    low, high = _clip_bandpass_edges(band[0], band[1], fs)
    filtered = bandpass_filter(
        signal, fs, low, high, order=PAPER_FILTER_ORDER
    )
    return compute_envelope_spectrum(filtered, fs, band=None, squared=False)


def envelope_band_from_kurtogram(
    signal: np.ndarray,
    fs: float,
) -> tuple[float, float]:
    """
    Kurtogram 검은 테두리 칸의 하한과 상한(Hz)을 반환한다.

    폭이 KURTOGRAM_MIN_BANDWIDTH 이상인 칸 중 충격 점수가 가장 큰 칸이다.
    """
    best = compute_kurtogram(signal, fs)["best"]
    return float(best["low"]), float(best["high"])


def _clip_bandpass_edges(
    low: float,
    high: float,
    fs: float,
) -> tuple[float, float]:
    """밴드패스 차단 주파수를 (0, 나이퀴스트) 안으로 넣는다."""
    nyquist = fs / 2.0
    high = min(high, nyquist * 0.99)
    low = max(low, 1.0)
    if low >= high:
        low = high * 0.5
    return low, high


# =========================================================================
# Kurtogram
# =========================================================================

def compute_kurtogram(
    signal: np.ndarray,
    fs: float,
    nlevel: int = KURTOGRAM_NLEVEL,
    min_bandwidth: float = KURTOGRAM_MIN_BANDWIDTH,
) -> dict:
    """
    주파수 구간별 충격 세기를 계산한다.

    0 Hz부터 샘플링 레이트의 절반까지를 점점 잘게 나눈다.
        단계 0: 구간 1개 (전체)
        단계 1: 구간 2개
        단계 2: 구간 4개
        ...
    각 구간만 남긴 뒤, 그 신호의 충격 점수(스펙트럴 커토시스)를 구한다.
    점수가 클수록 그 주파수에서 베어링 충격이 더 뚜렷하다는 뜻이다.

    Parameters
    ----------
    signal : np.ndarray
        시간 영역 진동 신호 (1D)
    fs : float
        샘플링 레이트 (Hz)
    nlevel : int
        가장 잘게 나눌 단계. 5이면 최대 32칸.
    min_bandwidth : float
        엔벨로프에 쓸 구간을 고를 때, 이 폭(Hz)보다 좁은 칸은 후보에서 뺀다.

    Returns
    -------
    dict
        {
            "fs": 샘플링 레이트,
            "rows": [단계별 칸 정보, ...],
            "best": 폭 조건을 만족하는 칸 중 점수가 가장 높은 칸,
        }
        각 row: level, bandwidth, centers, kurtosis
        best: level, center, bandwidth, low, high, kurtosis
    """
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)
    nyquist = fs / 2.0

    rows = []
    for level in range(nlevel + 1):
        n_bands = 2 ** level
        bandwidth = nyquist / n_bands
        centers = np.empty(n_bands)
        scores = np.empty(n_bands)

        for index in range(n_bands):
            low = index * bandwidth
            high = (index + 1) * bandwidth
            centers[index] = 0.5 * (low + high)
            kept = _keep_frequency_band(signal, fs, low, high)
            scores[index] = _spectral_kurtosis(kept)

        rows.append({
            "level": level,
            "bandwidth": float(bandwidth),
            "centers": centers,
            "kurtosis": scores,
        })

    best = _select_kurtogram_band(rows, min_bandwidth)
    return {"fs": float(fs), "rows": rows, "best": best}


def _spectral_kurtosis(signal: np.ndarray) -> float:
    """
    한 구간에 충격이 얼마나 모여 있는지 점수로 나타낸다.

    힐베르트 변환으로 진동의 크기 변화를 구한 뒤,
    그 크기가 크게 증가하면 점수가 커진다.
    크기 변화가 적으면 점수는 0 근처에 머문다.
    """
    amplitude = np.abs(hilbert(signal))
    power = amplitude ** 2
    mean_power = float(np.mean(power))
    if mean_power <= 0.0:
        return 0.0
    return float(np.mean(power ** 2) / (mean_power ** 2) - 2.0)


def _keep_frequency_band(
    signal: np.ndarray,
    fs: float,
    low: float,
    high: float,
) -> np.ndarray:
    """
    low~high(Hz)만 남긴다.

    맨 위 칸을 고역 통과로 처리하면 나이퀴스트 직전까지 전부 열려
    점수가 비정상적으로 커진다. 모든 칸을 같은 방식의 대역 제한으로 자른다.
    """
    nyquist = fs / 2.0
    if low <= 0.0 and high >= nyquist * 0.999:
        return signal

    # butter() 차단 비율은 0과 1 사이여야 한다.
    low_ratio = max(low / nyquist, 1e-4)
    high_ratio = min(high / nyquist, 0.999)
    if high_ratio - low_ratio < 0.01:
        return np.zeros_like(signal)

    if low <= 0.0:
        sos = butter(3, high_ratio, btype="low", output="sos")
    else:
        sos = butter(3, [low_ratio, high_ratio], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def _select_kurtogram_band(rows: list[dict], min_bandwidth: float) -> dict:
    """폭이 충분한 칸 중에서 충격 점수가 가장 큰 칸을 고른다."""
    candidates = []
    for row in rows:
        if row["bandwidth"] < min_bandwidth:
            continue
        index = int(np.argmax(row["kurtosis"]))
        center = float(row["centers"][index])
        bandwidth = float(row["bandwidth"])
        candidates.append({
            "level": int(row["level"]),
            "center": center,
            "bandwidth": bandwidth,
            "low": center - bandwidth / 2.0,
            "high": center + bandwidth / 2.0,
            "kurtosis": float(row["kurtosis"][index]),
        })

    if not candidates:
        raise ValueError(
            "Kurtogram에서 고를 수 있는 구간이 없습니다. "
            "min_bandwidth를 낮추거나 nlevel을 확인하세요."
        )
    return max(candidates, key=lambda item: item["kurtosis"])


# =========================================================================
# Smith & Randall (2015) Method 1 / Method 2
# =========================================================================

def compute_squared_envelope_spectrum_method1(
    signal: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    논문 Method 1: 원신호의 제곱 엔벨로프 스펙트럼을 계산한다.

    Smith & Randall (2015), Section 5.1 — "Envelope analysis of the raw signal"
    밴드패스 필터 없이, 원신호 전체 대역(full bandwidth)에 대해
    제곱 엔벨로프 스펙트럼(squared envelope spectrum)을 구한다.

    Parameters
    ----------
    signal : np.ndarray
        시간 영역 진동 신호 (1D, 원신호 그대로)
    fs : float
        샘플링 레이트 (Hz)

    Returns
    -------
    freqs : np.ndarray
        주파수 축 (Hz)
    sq_env_magnitude : np.ndarray
        제곱 엔벨로프 스펙트럼 진폭
    squared_envelope : np.ndarray
        시간 영역의 제곱 엔벨로프 (시간 영역 플롯에 사용)
    """
    return compute_envelope_spectrum(signal, fs, band=None, squared=True)


def cepstrum_prewhiten(signal: np.ndarray) -> np.ndarray:
    """
    켑스트럼 프리화이트닝: 모든 주파수 성분의 크기를 균일화한다.

    Smith & Randall (2015), Section 5.2 참조.
    원래 Sawalhi & Randall (2011)에서 제안된 기법.

    원리:
        FFT 스펙트럼의 크기(magnitude)를 모두 1로 만들고,
        위상(phase) 정보만 보존한 뒤 IFFT로 시간 영역에 복원한다.
        이렇게 하면 모든 주파수 대역의 PSD가 동일해지므로,
        임펄스성이 높은 대역이 시간 영역에서 자연스럽게 부각된다.

    수식:
        X(f) = |X(f)| * exp(j*phi(f))   -- 원래 스펙트럼
        X_pw(f) = exp(j*phi(f))          -- 크기=1, 위상만 유지
        x_pw(t) = IFFT(X_pw(f))          -- 프리화이트닝된 신호

    Parameters
    ----------
    signal : np.ndarray
        시간 영역 진동 신호 (1D)

    Returns
    -------
    np.ndarray
        프리화이트닝된 신호 (시간 영역)
    """
    n = len(signal)
    X = np.fft.rfft(signal)

    # 크기 스펙트럼 (0으로 나누기 방지)
    magnitude = np.abs(X)
    magnitude[magnitude == 0] = 1.0

    # 크기를 1로 정규화 → 위상만 남김
    X_pw = X / magnitude

    # 시간 영역으로 복원
    return np.fft.irfft(X_pw, n=n)


def compute_squared_envelope_spectrum_method2(
    signal: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    논문 Method 2: 켑스트럼 프리화이트닝 후 제곱 엔벨로프 스펙트럼을 계산한다.

    Smith & Randall (2015), Section 5.2 — "Cepstrum prewhitening"

    Parameters
    ----------
    signal : np.ndarray
        시간 영역 진동 신호 (1D, 원신호 그대로)
    fs : float
        샘플링 레이트 (Hz)

    Returns
    -------
    freqs : np.ndarray
        주파수 축 (Hz)
    sq_env_magnitude : np.ndarray
        제곱 엔벨로프 스펙트럼 진폭
    squared_envelope : np.ndarray
        시간 영역의 제곱 엔벨로프 (시간 영역 플롯에 사용)
    prewhitened : np.ndarray
        프리화이트닝된 시간 영역 신호 (플롯에 사용)
    """
    prewhitened = cepstrum_prewhiten(signal)
    freqs, magnitude, squared_envelope = compute_envelope_spectrum(
        prewhitened, fs, band=None, squared=True
    )
    return freqs, magnitude, squared_envelope, prewhitened


def compute_bandpass_envelope_spectrum(
    signal: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    config에 설정된 공진 대역(기본 2~5 kHz)으로 밴드패스한 뒤
    일반 엔벨로프 스펙트럼을 계산한다.

    12kHz 데이터는 나이퀴스트가 6kHz이므로 상한을 자동으로 잘라낸다.

    Parameters
    ----------
    signal : np.ndarray
        시간 영역 진동 신호 (1D)
    fs : float
        샘플링 레이트 (Hz)

    Returns
    -------
    freqs : np.ndarray
        주파수 축 (Hz)
    envelope_magnitude : np.ndarray
        엔벨로프 스펙트럼 진폭
    """
    high = min(ENVELOPE_BANDPASS_HIGH, fs / 2.0 * 0.95)
    freqs, magnitude, _ = compute_envelope_spectrum(
        signal, fs, band=(ENVELOPE_BANDPASS_LOW, high), squared=False
    )
    return freqs, magnitude


# =========================================================================
# 결함 주파수 계산
# =========================================================================

def calculate_defect_frequencies(
    rpm: float,
    n_harmonics: int = N_HARMONICS,
) -> dict:
    """
    주어진 RPM에서 각 결함 유형의 주파수와 고조파를 계산한다.

    Parameters
    ----------
    rpm : float
        모터 회전 속도 (RPM)
    n_harmonics : int
        계산할 고조파 개수

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
        result[name] = [base_freq * (i + 1) for i in range(n_harmonics)]

    return result


def refine_base_frequency(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    theoretical_freq: float,
    search_ratio: float = 0.02,
    min_peak_ratio: float = 3.0,
    noise_band: tuple[float, float] = (5.0, 1_000.0),
) -> float:
    """
    이론 결함 주파수 근처의 실제 피크 위치를 찾아 기본 주파수를 보정한다.

    필요성:
        CWRU가 제공하는 RPM은 소수점 단위까지 정확하지 않고, 실제 축 속도도
        미세하게 변동한다. 결함 주파수는 RPM에 선형 비례하므로 이 오차가
        고조파 차수만큼 증폭된다. 예를 들어 기본 주파수 오차가 0.3 Hz면
        6차 고조파에서는 1.8 Hz 어긋나 마커가 피크를 벗어난다.
        따라서 1차 피크에서 실측 기본 주파수를 구해 고조파를 계산하면
        모든 차수에서 마커와 피크가 일치한다.

    유의성 검사:
        결함이 없거나 해당 결함 주파수가 나타나지 않는 경우(정상 베어링,
        볼 결함의 BSF 등) 탐색 구간의 최대값은 노이즈일 뿐이다. 이를
        기본 주파수로 채택하면 없는 결함을 있는 것처럼 보이게 만든다.
        따라서 피크가 노이즈 수준(스펙트럼 중앙값)의 min_peak_ratio배를
        넘지 못하면 보정하지 않고 이론값을 반환한다.

    Parameters
    ----------
    freqs : np.ndarray
        스펙트럼 주파수 축 (Hz)
    magnitude : np.ndarray
        스펙트럼 진폭
    theoretical_freq : float
        이론 기본 결함 주파수 (Hz)
    search_ratio : float
        탐색 범위 비율. 기본 0.02 → 이론값의 ±2% 구간에서 최대 피크를 찾는다.
    min_peak_ratio : float
        노이즈 수준 대비 최소 피크 배율. 이 값을 넘지 못하면 보정하지 않는다.
    noise_band : tuple[float, float]
        노이즈 수준(중앙값)을 추정할 주파수 구간 (Hz)

    Returns
    -------
    float
        실측 기본 주파수 (Hz). 탐색 구간이 비었거나 피크가 유의하지 않으면
        이론값을 그대로 반환한다.
    """
    low = theoretical_freq * (1.0 - search_ratio)
    high = theoretical_freq * (1.0 + search_ratio)
    window = (freqs >= low) & (freqs <= high)

    if not window.any():
        return theoretical_freq

    peak_index = int(np.argmax(magnitude[window]))
    peak_amplitude = float(magnitude[window][peak_index])

    noise_window = (freqs >= noise_band[0]) & (freqs <= noise_band[1])
    noise_level = float(np.median(magnitude[noise_window])) \
        if noise_window.any() else 0.0

    if noise_level > 0 and peak_amplitude < min_peak_ratio * noise_level:
        return theoretical_freq

    return float(freqs[window][peak_index])


def print_defect_frequencies(rpm: float) -> None:
    """
    결함 주파수 계산 결과를 콘솔에 보기 좋게 출력한다.

    Parameters
    ----------
    rpm : float
        모터 회전 속도 (RPM)
    """
    freqs = calculate_defect_frequencies(rpm)
    shaft = freqs["shaft_freq"]

    print(f"\n{'=' * 60}")
    print(f"  결함 주파수 계산 결과 (RPM={rpm:.1f}, f_r={shaft:.3f} Hz)")
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
