"""同步性分析 - 滞后相关 (Lagged Correlation)

对每对说话人的语音特征时间序列，计算lag=0到lag_max的Pearson相关，
取绝对值最大的相关系数及其对应lag作为同步性指标。
使用置换检验判定显著性。
"""

import logging
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_lagged_correlation(series_a: np.ndarray, series_b: np.ndarray,
                               max_lag: int = 5) -> Dict[int, float]:
    """计算滞后交叉相关

    对lag从-max_lag到+max_lag，计算series_a与series_b之间的Pearson相关。
    正lag表示series_b领先(series_a滞后)，负lag表示series_a领先。

    Args:
        series_a: 说话人A的特征时间序列
        series_b: 说话人B的特征时间序列
        max_lag: 最大滞后步数

    Returns:
        字典 {lag: pearson_r}, lag从-max_lag到+max_lag
    """
    series_a = np.asarray(series_a, dtype=np.float64)
    series_b = np.asarray(series_b, dtype=np.float64)

    n = min(len(series_a), len(series_b))
    if n < 3:
        logger.warning(f"序列长度不足({n}<3), 无法计算相关")
        return {lag: np.nan for lag in range(-max_lag, max_lag + 1)}

    lag_correlations = {}

    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            a = series_a[:n]
            b = series_b[:n]
        elif lag > 0:
            # 正lag: B领先, A滞后 -> 比较A[lag:] vs B[:n-lag]
            effective_n = n - lag
            if effective_n < 3:
                lag_correlations[lag] = np.nan
                continue
            a = series_a[lag:lag + effective_n]
            b = series_b[:effective_n]
        else:
            # 负lag: A领先, B滞后 -> 比较A[:n+lag] vs B[-lag:]
            abs_lag = abs(lag)
            effective_n = n - abs_lag
            if effective_n < 3:
                lag_correlations[lag] = np.nan
                continue
            a = series_a[:effective_n]
            b = series_b[abs_lag:abs_lag + effective_n]

        # 检查方差是否为零
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            lag_correlations[lag] = 0.0
            continue

        r, _ = stats.pearsonr(a, b)
        lag_correlations[lag] = r if np.isfinite(r) else 0.0

    return lag_correlations


def find_peak_correlation(lag_correlations: Dict[int, float]) -> Tuple[int, float, float]:
    """找到绝对值最大的相关系数及其对应lag

    Args:
        lag_correlations: {lag: r_value} 字典

    Returns:
        (best_lag, best_abs_r, best_r_signed)
        - best_lag: 最大|r|对应的lag
        - best_abs_r: 最大|r|值
        - best_r_signed: 带符号的r值
    """
    best_lag = 0
    best_abs_r = 0.0
    best_r_signed = 0.0

    for lag, r in lag_correlations.items():
        if np.isnan(r):
            continue
        abs_r = abs(r)
        if abs_r > best_abs_r:
            best_abs_r = abs_r
            best_r_signed = r
            best_lag = lag

    return best_lag, best_abs_r, best_r_signed


def permutation_test_synchrony(series_a: np.ndarray, series_b: np.ndarray,
                               max_lag: int = 5, n_perm: int = 500,
                               seed: int = 42) -> Dict:
    """置换检验评估同步性显著性

    通过打乱series_a的顺序，计算置换分布下的峰值相关，
    以此评估观测到的同步性是否显著高于随机水平。

    Args:
        series_a: 说话人A的特征时间序列
        series_b: 说话人B的特征时间序列
        max_lag: 最大滞后步数
        n_perm: 置换次数
        seed: 随机种子

    Returns:
        字典: {
            obs_peak_r: 观测峰值|r|,
            obs_peak_lag: 观测峰值对应lag,
            perm_mean: 置换分布均值,
            perm_std: 置换分布标准差,
            p_value: p值,
            significant: 是否显著 (alpha=0.05)
        }
    """
    series_a = np.asarray(series_a, dtype=np.float64)
    series_b = np.asarray(series_b, dtype=np.float64)

    # 计算观测值
    obs_lag_corr = compute_lagged_correlation(series_a, series_b, max_lag)
    obs_lag, obs_peak_r, obs_r_signed = find_peak_correlation(obs_lag_corr)

    # 如果观测值无效，直接返回
    if np.isnan(obs_peak_r) or obs_peak_r == 0:
        return {
            'obs_peak_r': obs_peak_r,
            'obs_peak_lag': obs_lag,
            'perm_mean': np.nan,
            'perm_std': np.nan,
            'p_value': 1.0,
            'significant': False
        }

    # 置换检验
    rng = np.random.default_rng(seed)
    perm_peaks = np.zeros(n_perm)

    for i in range(n_perm):
        # 打乱series_a
        shuffled_a = rng.permutation(series_a)
        perm_lag_corr = compute_lagged_correlation(shuffled_a, series_b, max_lag)
        _, perm_peak, _ = find_peak_correlation(perm_lag_corr)
        perm_peaks[i] = perm_peak

    # 计算p值 (单侧: 观测峰值 >= 置换峰值的比例)
    p_value = np.mean(perm_peaks >= obs_peak_r)

    return {
        'obs_peak_r': obs_peak_r,
        'obs_peak_lag': obs_lag,
        'obs_r_signed': obs_r_signed,
        'perm_mean': np.mean(perm_peaks),
        'perm_std': np.std(perm_peaks),
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def analyze_synchrony_all(feature_df: pd.DataFrame,
                          features: List[str],
                          speaker_col: str = 'speaker',
                          time_col: str = 'start',
                          file_col: str = 'file',
                          max_lag: int = 5,
                          n_perm: int = 500,
                          alpha: float = 0.05,
                          callback: Optional[Callable[[float, str], None]] = None
                          ) -> pd.DataFrame:
    """主函数: 分析所有文件×特征的同步性

    对每个对话文件中的每个特征:
    1. 按说话人拆分为两个时间序列
    2. 按说话轮次索引对齐(取较短长度)
    3. 计算滞后相关 + 置换检验

    Args:
        feature_df: 特征DataFrame, 包含列 [file, speaker, start, end, feature1, ...]
        features: 要分析的特征列名列表
        speaker_col: 说话人列名
        time_col: 时间列名
        file_col: 文件/对话标识列名
        max_lag: 最大滞后步数
        n_perm: 置换次数
        alpha: 显著性水平
        callback: 进度回调函数

    Returns:
        结果DataFrame: [file, feature, peak_r, peak_lag, p_value, significant, direction]
    """
    results = []

    files = feature_df[file_col].unique()
    total_tasks = len(files) * len(features)
    completed = 0

    for file_id in files:
        file_data = feature_df[feature_df[file_col] == file_id].copy()
        file_data = file_data.sort_values(time_col).reset_index(drop=True)

        speakers = file_data[speaker_col].unique()
        if len(speakers) < 2:
            logger.warning(f"文件 {file_id} 只有 {len(speakers)} 个说话人, 跳过")
            completed += len(features)
            continue

        # 取前两个说话人
        spk_a, spk_b = speakers[0], speakers[1]
        data_a = file_data[file_data[speaker_col] == spk_a].reset_index(drop=True)
        data_b = file_data[file_data[speaker_col] == spk_b].reset_index(drop=True)

        for feature in features:
            completed += 1

            if feature not in feature_df.columns:
                logger.warning(f"特征 '{feature}' 不在DataFrame中, 跳过")
                continue

            # 提取特征值序列
            series_a = data_a[feature].dropna().values
            series_b = data_b[feature].dropna().values

            # 对齐长度 (取较短序列)
            min_len = min(len(series_a), len(series_b))
            if min_len < max_lag + 3:
                logger.debug(
                    f"文件={file_id}, 特征={feature}: 序列太短({min_len}), 跳过"
                )
                results.append({
                    'file': file_id,
                    'feature': feature,
                    'peak_r': np.nan,
                    'peak_lag': np.nan,
                    'p_value': np.nan,
                    'significant': False,
                    'direction': 'insufficient_data'
                })
                continue

            series_a = series_a[:min_len]
            series_b = series_b[:min_len]

            # 执行置换检验
            test_result = permutation_test_synchrony(
                series_a, series_b,
                max_lag=max_lag,
                n_perm=n_perm,
                seed=42
            )

            # 判断方向
            if test_result['significant']:
                if test_result.get('obs_r_signed', test_result['obs_peak_r']) > 0:
                    direction = 'positive_sync'
                else:
                    direction = 'negative_sync'
            else:
                direction = 'not_significant'

            results.append({
                'file': file_id,
                'feature': feature,
                'peak_r': test_result['obs_peak_r'],
                'peak_lag': test_result['obs_peak_lag'],
                'p_value': test_result['p_value'],
                'significant': test_result['p_value'] < alpha,
                'direction': direction
            })

            if callback:
                progress = 100.0 * completed / total_tasks
                callback(progress, f"同步性分析: {file_id} - {feature}")

    if callback:
        callback(100.0, "同步性分析完成")

    result_df = pd.DataFrame(results)
    return result_df
