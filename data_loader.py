"""
CWRU 베어링 결함 주파수 검증 프로젝트 - 데이터 로딩 모듈

CWRU Bearing Data Center의 .mat 파일을 로드하고,
Drive End(DE) 가속도계 시계열 데이터를 추출한다.
"""

import numpy as np
import scipy.io

from config import DATASETS, SAMPLING_RATE


def load_mat_file(dataset_key: str) -> dict:
    """
    지정된 데이터셋 키에 해당하는 .mat 파일을 로드한다.

    Parameters
    ----------
    dataset_key : str
        DATASETS 딕셔너리의 키 (예: "Normal_1", "IR021_1")

    Returns
    -------
    dict
        {
            "key":        데이터셋 키 (str),
            "signal":     DE 가속도계 시계열 (1D np.ndarray),
            "n_samples":  총 샘플 수 (int),
            "duration_s": 측정 시간 (float, 초),
            "info":       DATASETS에 정의된 메타정보 (dict),
        }

    Raises
    ------
    FileNotFoundError
        .mat 파일이 존재하지 않을 때
    KeyError
        DE 시계열 키를 .mat 파일에서 찾을 수 없을 때
    """
    if dataset_key not in DATASETS:
        raise KeyError(f"알 수 없는 데이터셋 키: '{dataset_key}'. "
                       f"사용 가능: {list(DATASETS.keys())}")

    info = DATASETS[dataset_key]
    file_path = info["file"]
    file_id = info["file_id"]

    mat = scipy.io.loadmat(file_path)

    # DE(Drive End) 시계열 키 탐색: "X{file_id}_DE_time" 형식
    de_key = _find_de_key(mat, file_id)
    signal = mat[de_key].flatten().astype(np.float64)

    n_samples = len(signal)
    duration_s = n_samples / SAMPLING_RATE

    return {
        "key": dataset_key,
        "signal": signal,
        "n_samples": n_samples,
        "duration_s": duration_s,
        "info": info,
    }


def load_all_datasets() -> dict:
    """
    config.py에 정의된 모든 데이터셋을 로드한다.

    Returns
    -------
    dict
        {dataset_key: load_mat_file(dataset_key)의 반환값, ...}
    """
    loaded = {}
    for key in DATASETS:
        loaded[key] = load_mat_file(key)
    return loaded


def print_dataset_summary(data: dict) -> None:
    """
    로드된 데이터셋의 요약 정보를 콘솔에 출력한다.

    Parameters
    ----------
    data : dict
        load_mat_file()의 반환값
    """
    info = data["info"]
    print(f"{'=' * 60}")
    print(f"  데이터셋  : {data['key']}")
    print(f"  파일 ID   : {info['file_id']}.mat")
    print(f"  설명      : {info['description']}")
    print(f"  결함 유형 : {info['fault_type']}")
    print(f"  모터 부하 : {info['load_hp']} HP")
    print(f"  모터 RPM  : {info['rpm']}")
    print(f"  샘플 수   : {data['n_samples']:,}")
    print(f"  측정 시간 : {data['duration_s']:.2f} 초")
    print(f"  샘플링    : {SAMPLING_RATE:,} Hz")
    print(f"  신호 통계 : mean={data['signal'].mean():.6f}, "
          f"std={data['signal'].std():.6f}, "
          f"peak={np.abs(data['signal']).max():.6f}")
    print(f"{'=' * 60}")


def _find_de_key(mat: dict, file_id: int) -> str:
    """
    .mat 파일에서 Drive End 시계열 변수 키를 탐색한다.

    CWRU .mat 파일의 DE 키 형식: "X{file_id}_DE_time"
    일부 파일은 3자리 패딩 없이 저장되어 있으므로 유연하게 탐색한다.

    Parameters
    ----------
    mat : dict
        scipy.io.loadmat()으로 로드된 딕셔너리
    file_id : int
        CWRU Recording ID (예: 98, 214)

    Returns
    -------
    str
        매칭된 DE 키

    Raises
    ------
    KeyError
        DE 키를 찾을 수 없을 때
    """
    # 정확한 매칭 시도
    exact_key = f"X{file_id:03d}_DE_time"
    if exact_key in mat:
        return exact_key

    exact_key_no_pad = f"X{file_id}_DE_time"
    if exact_key_no_pad in mat:
        return exact_key_no_pad

    # 'DE_time'이 포함된 키를 유연하게 탐색
    for key in mat.keys():
        if "DE_time" in key:
            return key

    # DE 키가 없으면 사용 가능한 키 목록을 보여줌
    available = [k for k in mat.keys() if not k.startswith("__")]
    raise KeyError(
        f"파일 ID {file_id}에서 DE 시계열을 찾을 수 없습니다. "
        f"사용 가능한 키: {available}"
    )
