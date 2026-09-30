"""
CWRU 베어링 결함 주파수 검증 프로젝트 - 설정 모듈

베어링 사양, 데이터셋 매핑, 결함 주파수 배수, 분석 파라미터 등
프로젝트 전역에서 사용하는 상수를 정의한다.
"""

import os

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ---------------------------------------------------------------------------
# 샘플링 설정
#
# CWRU 데이터셋은 그룹마다 샘플링 레이트가 다르다.
#   - normal baseline     : 48 kHz
#   - 48k drive end fault : 48 kHz
#   - 12k drive end fault : 12 kHz
#
# 따라서 전역 상수 하나로 다룰 수 없고, DATASETS의 "fs" 값을 사용해야 한다.
# 아래 상수는 fs 정보가 없을 때의 기본값(폴백)으로만 쓰인다.
# ---------------------------------------------------------------------------
DEFAULT_SAMPLING_RATE = 48_000  # Hz

# ---------------------------------------------------------------------------
# 베어링 사양: SKF 6205-2RS JEM (Drive End)
# ---------------------------------------------------------------------------
BEARING = {
    "model": "SKF 6205-2RS JEM",
    "n_balls": 9,                   # 볼 개수
    "ball_diameter_inch": 0.3126,   # 볼 직경 (인치)
    "pitch_diameter_inch": 1.537,   # 피치 직경 (인치)
    "contact_angle_deg": 0.0,       # 접촉각 (도)
}

# ---------------------------------------------------------------------------
# 결함 주파수 배수 (축 회전 주파수의 배수)
#
# CWRU 공식 사이트 기준값 사용:
#   - BPFI (Inner Race)      : 5.4152
#   - BPFO (Outer Race)      : 3.5848
#   - BSF  (Rolling Element) : 4.7135  (= 2 × 볼 자전 주파수)
#   - FTF  (Cage Train)      : 0.39828
#
# BSF 참고:
#   볼 결함 시 볼 1회전당 내륜·외륜에 각 1회씩, 총 2회 충격 발생.
#   따라서 실제 관측되는 결함 주파수 = 2 × BSF(볼 자전 주파수).
#   Smith & Randall (2015) 논문의 BSF=2.357은 볼 자전 주파수이며,
#   CWRU 사이트의 4.7135 = 2 × 2.357 이 결함 충격 주파수이다.
# ---------------------------------------------------------------------------
DEFECT_FREQ_MULTIPLIERS = {
    "BPFI": 5.4152,     # Ball Pass Frequency Inner race
    "BPFO": 3.5848,     # Ball Pass Frequency Outer race
    "BSF":  4.7135,     # Ball Spin Frequency (2×, 결함 충격 기준)
    "FTF":  0.39828,    # Fundamental Train Frequency (케이지)
}

# ---------------------------------------------------------------------------
# 결함 주파수 마커 색상 및 스타일
# ---------------------------------------------------------------------------
DEFECT_FREQ_COLORS = {
    "BPFO": {"color": "#E74C3C", "label": "BPFO (Outer Race)"},
    "BPFI": {"color": "#3498DB", "label": "BPFI (Inner Race)"},
    "BSF":  {"color": "#2ECC71", "label": "BSF (Ball)"},
    "FTF":  {"color": "#F39C12", "label": "FTF (Cage)"},
    "1xRPM": {"color": "#95A5A6", "label": "1× RPM"},
}

# ---------------------------------------------------------------------------
# 데이터셋 매핑
#
# 두 그룹을 함께 관리한다.
#   1) BASE_KEYS  : 48kHz, 0.021", 1HP  — 기존 검증 대상
#   2) PAPER_KEYS : 12kHz, 0.007", 0HP  — 논문(Alonso-González et al., 2023)
#                   Fig.5~8 재현 대상
#
# "rpm"은 CWRU 문서상의 공칭값이며, 실제 분석에는 .mat 파일에 들어 있는
# X{id}RPM 실측값을 우선 사용한다(data_loader 참조).
# 파일명은 CWRU 원본 Recording ID를 그대로 유지한다.
# ---------------------------------------------------------------------------
DATASETS = {
    # ---------------- 48kHz / 0.021" / 1HP ----------------
    "Normal_1": {
        "file": os.path.join(DATA_DIR, "normal_baseline", "98.mat"),
        "file_id": 98,
        "fs": 48_000,
        "fault_type": "Normal",
        "fault_diameter_inch": None,
        "load_hp": 1,
        "rpm": 1772,
        "description": "정상 베어링 (1HP, 1772 RPM)",
    },
    "IR021_1": {
        "file": os.path.join(DATA_DIR, "48k_drive_end_fault", "inch021_1HP", "214.mat"),
        "file_id": 214,
        "fs": 48_000,
        "fault_type": "Inner Race",
        "fault_diameter_inch": 0.021,
        "load_hp": 1,
        "rpm": 1772,
        "description": "내륜 결함 0.021\" (1HP, 1772 RPM)",
    },
    "B021_1": {
        "file": os.path.join(DATA_DIR, "48k_drive_end_fault", "inch021_1HP", "227.mat"),
        "file_id": 227,
        "fs": 48_000,
        "fault_type": "Ball",
        "fault_diameter_inch": 0.021,
        "load_hp": 1,
        "rpm": 1772,
        "description": "볼 결함 0.021\" (1HP, 1772 RPM)",
    },
    "OR021@6_1": {
        "file": os.path.join(DATA_DIR, "48k_drive_end_fault", "inch021_1HP", "239.mat"),
        "file_id": 239,
        "fs": 48_000,
        "fault_type": "Outer Race",
        "fault_diameter_inch": 0.021,
        "load_hp": 1,
        "rpm": 1772,
        "or_position": "6:00 (하중대 중심)",
        "description": "외륜 결함 0.021\" @6시 (1HP, 1772 RPM)",
    },
    "OR021@3_1": {
        "file": os.path.join(DATA_DIR, "48k_drive_end_fault", "inch021_1HP", "251.mat"),
        "file_id": 251,
        "fs": 48_000,
        "fault_type": "Outer Race",
        "fault_diameter_inch": 0.021,
        "load_hp": 1,
        "rpm": 1772,
        "or_position": "3:00 (하중대 직교)",
        "description": "외륜 결함 0.021\" @3시 (1HP, 1772 RPM)",
    },
    "OR021@12_1": {
        "file": os.path.join(DATA_DIR, "48k_drive_end_fault", "inch021_1HP", "263.mat"),
        "file_id": 263,
        "fs": 48_000,
        "fault_type": "Outer Race",
        "fault_diameter_inch": 0.021,
        "load_hp": 1,
        "rpm": 1772,
        "or_position": "12:00 (하중대 반대)",
        "description": "외륜 결함 0.021\" @12시 (1HP, 1772 RPM)",
    },

    # ---------------- 12kHz / 0.007" / 0HP (논문 Fig.5~8) ----------------
    # 주의: normal baseline(97.mat)은 CWRU 원본이 48kHz로 수집되었다.
    #       논문 본문은 12kHz로 기술하고 있으나 원본 스펙을 따른다.
    "Normal_0": {
        "file": os.path.join(DATA_DIR, "normal_baseline", "97.mat"),
        "file_id": 97,
        "fs": 48_000,
        "fault_type": "Normal",
        "fault_diameter_inch": None,
        "load_hp": 0,
        "rpm": 1797,
        "description": "정상 베어링 (0HP, 1797 RPM)",
    },
    "IR007_0": {
        "file": os.path.join(DATA_DIR, "12k_drive_end_fault", "inch007_0HP", "105.mat"),
        "file_id": 105,
        "fs": 12_000,
        "fault_type": "Inner Race",
        "fault_diameter_inch": 0.007,
        "load_hp": 0,
        "rpm": 1797,
        "description": "내륜 결함 0.007\" (0HP, 1797 RPM)",
    },
    "B007_0": {
        "file": os.path.join(DATA_DIR, "12k_drive_end_fault", "inch007_0HP", "118.mat"),
        "file_id": 118,
        "fs": 12_000,
        "fault_type": "Ball",
        "fault_diameter_inch": 0.007,
        "load_hp": 0,
        "rpm": 1797,
        "description": "볼 결함 0.007\" (0HP, 1797 RPM)",
    },
    "OR007@6_0": {
        "file": os.path.join(DATA_DIR, "12k_drive_end_fault", "inch007_0HP", "130.mat"),
        "file_id": 130,
        "fs": 12_000,
        "fault_type": "Outer Race",
        "fault_diameter_inch": 0.007,
        "load_hp": 0,
        "rpm": 1797,
        "or_position": "6:00 (하중대 중심)",
        "description": "외륜 결함 0.007\" @6시 (0HP, 1797 RPM)",
    },
}

# 그룹별 데이터셋 키 목록
BASE_KEYS = ["Normal_1", "IR021_1", "B021_1",
             "OR021@6_1", "OR021@3_1", "OR021@12_1"]
PAPER_KEYS = ["Normal_0", "IR007_0", "OR007@6_0", "B007_0"]

# ---------------------------------------------------------------------------
# 엔벨로프 분석 파라미터
#
# SKF 6205 소형 베어링의 하우징 공진 대역은 약 2~6 kHz.
# 결함 임펄스가 이 공진을 가진하므로, 밴드패스 필터로 해당 대역을
# 분리한 뒤 엔벨로프를 추출하면 결함 주파수가 뚜렷하게 드러남.
# (Smith & Randall, 2015 동일 접근법)
# ---------------------------------------------------------------------------
ENVELOPE_BANDPASS_LOW = 2_000    # Hz
ENVELOPE_BANDPASS_HIGH = 5_000   # Hz
ENVELOPE_FILTER_ORDER = 5        # Butterworth 필터 차수

# ---------------------------------------------------------------------------
# 시각화 파라미터
# ---------------------------------------------------------------------------
FFT_PLOT_FREQ_MAX = 1_000       # FFT 플롯 x축 최대 주파수 (Hz)
ENVELOPE_PLOT_FREQ_MAX = 500    # 엔벨로프 플롯 x축 최대 주파수 (Hz)
N_HARMONICS = 5                 # 결함 주파수 고조파 표시 개수
FIGURE_DPI = 150                # 저장 이미지 해상도

# ---------------------------------------------------------------------------
# 논문 재현(Fig.5~8) 전용 파라미터
#
# 논문 Fig.6은 x축 0~1000 Hz 범위에 BPFO 고조파 9개를 표시한다.
# 진폭 단위는 g이며, 제곱 엔벨로프가 아닌 일반 엔벨로프(|Hilbert|)를 사용한다.
# 밴드패스 구간은 파일마다 Kurtogram이 고른 칸(검은 테두리)을 쓴다.
# ---------------------------------------------------------------------------
PAPER_PLOT_FREQ_MAX = 1_000     # 논문 스타일 플롯 x축 최대 주파수 (Hz)
PAPER_N_HARMONICS = 9           # 논문 스타일 플롯 고조파 표시 개수
PAPER_FILTER_ORDER = 8          # Kurtogram 구간에 적용할 Butterworth 차수

# ---------------------------------------------------------------------------
# Kurtogram
#
# 0 Hz ~ 샘플링/2 를 1칸, 2칸, 4칸, ... 으로 나누고
# 각 칸의 충격 세기(스펙트럴 커토시스)를 색으로 그린다.
# 엔벨로프에 쓸 구간은 "폭이 KURTOGRAM_MIN_BANDWIDTH 이상인 칸" 중에서
# 점수가 가장 높은 칸으로 고른다. 너무 좁은 칸은 엔벨로프의
# 높은 배수(수백 Hz)를 담지 못한다.
# ---------------------------------------------------------------------------
KURTOGRAM_NLEVEL = 5            # 0이면 통대역 1칸, 5이면 가장 좁은 칸까지
KURTOGRAM_MIN_BANDWIDTH = 1_000 # Hz, 대역을 고를 때 이보다 좁은 칸은 제외
