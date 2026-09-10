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
# ---------------------------------------------------------------------------
SAMPLING_RATE = 48_000  # Hz (Normal Baseline + 48k Drive End Fault)

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
# 데이터셋 매핑: 48kHz, 0.021", 1HP (1772 RPM)
#
# 파일명은 CWRU 원본 Recording ID를 그대로 유지한다.
# ---------------------------------------------------------------------------
DATASETS = {
    "Normal_1": {
        "file": os.path.join(DATA_DIR, "normal_baseline", "98.mat"),
        "file_id": 98,
        "fault_type": "Normal",
        "fault_diameter_inch": None,
        "load_hp": 1,
        "rpm": 1772,
        "description": "정상 베어링 (1HP, 1772 RPM)",
    },
    "IR021_1": {
        "file": os.path.join(DATA_DIR, "48k_drive_end_fault", "inch021_1HP", "214.mat"),
        "file_id": 214,
        "fault_type": "Inner Race",
        "fault_diameter_inch": 0.021,
        "load_hp": 1,
        "rpm": 1772,
        "description": "내륜 결함 0.021\" (1HP, 1772 RPM)",
    },
    "B021_1": {
        "file": os.path.join(DATA_DIR, "48k_drive_end_fault", "inch021_1HP", "227.mat"),
        "file_id": 227,
        "fault_type": "Ball",
        "fault_diameter_inch": 0.021,
        "load_hp": 1,
        "rpm": 1772,
        "description": "볼 결함 0.021\" (1HP, 1772 RPM)",
    },
    "OR021@6_1": {
        "file": os.path.join(DATA_DIR, "48k_drive_end_fault", "inch021_1HP", "239.mat"),
        "file_id": 239,
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
        "fault_type": "Outer Race",
        "fault_diameter_inch": 0.021,
        "load_hp": 1,
        "rpm": 1772,
        "or_position": "12:00 (하중대 반대)",
        "description": "외륜 결함 0.021\" @12시 (1HP, 1772 RPM)",
    },
}

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
