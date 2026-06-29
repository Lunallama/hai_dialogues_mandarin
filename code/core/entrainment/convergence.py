"""趋同性分析 - Convergence/Divergence

计算相邻跨说话人对的特征差异是否随时间减小(趋同)或增大(趋离)。
使用线性回归斜率 + 置换检验。
"""

import logging
from typing import List, Dict, Optional, Callable

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_adjacent_diffs(feature_df: pd.DataFrame,
                           feature: str,
                           speaker_col: str = 'speaker',
                           time_col: str = 'start') -> pd.DataFrame:
    """计算相邻跨说话人对的特征绝对差值

    对于一个对话的所有发言(按时间排序), 找出相邻的来自不同说话人的发言对,
    计算它们之间特征值的绝对差异。

    Args:
        feature_df: 单个对话的特征DataFrame (已按时间排序)
        feature: 特征列名
        speaker_col: 说话人列名
        time_col: 时间列名

    Returns:
        DataFrame: [pair_idx, time_center, diff]
            - pair_idx: 对的序号
            - time_center: 两个发言时间中点
            - diff: 特征绝对差值
    """
    df = feature_df.sort_values(time_col).reset_index(drop=True)

    pairs = []
    pair_idx = 0

    for i in range(len(df) - 1):
        # 只取相邻且来自不同说话人的对
        if df.iloc[i][speaker_col] != df.iloc[i + 1][speaker_col]:
            val_a = df.iloc[i][feature]
            val_b = df.iloc[i + 1][feature]

            # 跳过NaN值
            if pd.isna(val_a) or pd.isna(val_b):
                continue

            time_a = df.iloc[i][time_col]
            time_b = df.iloc[i + 1][time_col]

            pairs.append({
                'pair_idx': pair_idx,
                'time_center': (time_a + time_b) / 2.0,
                'diff': abs(val_b - val_a)
            })
            pair_idx += 1

    return pd.DataFrame(pairs) if pairs else pd.DataFrame(
        columns=['pair_idx', 'time_center', 'diff']
    )


def compute_convergence_slope(times: np.ndarray,
                              diffs: np.ndarray) -> tuple:
    """计算差异值随时间的线性回归斜率

    通过线性回归分析特征差异是否随时间变化:
    - 斜率 < 0: 趋同 (差异随时间减小)
    - 斜率 > 0: 趋离 (差异随时间增大)
    - 斜率 ≈ 0: 无明显趋势

    Args:
        times: 时间序列 (自变量)
        diffs: 差异值序列 (因变量)

    Returns:
        (slope, p_value, r_value)
        - slope: 回归斜率
        - p_value: 斜率的统计显著性 (双侧)
        - r_value: 相关系数
    """
    times = np.asarray(times, dtype=np.float64)
    diffs = np.asarray(diffs, dtype=np.float64)

    if len(times) < 3:
        return np.nan, np.nan, np.nan

    # 检查方差
    if np.std(times) < 1e-10 or np.std(diffs) < 1e-10:
        return 0.0, 1.0, 0.0

    slope, intercept, r_value, p_value, std_err = stats.linregress(times, diffs)

    return slope, p_value, r_value


def permutation_test_convergence(feature_df: pd.DataFrame,
                                 feature: str,
                                 speaker_col: str = 'speaker',
                                 time_col: str = 'start',
                                 n_perm: int = 500,
                                 seed: int = 42) -> Dict:
    """置换检验评估趋同性显著性

    观测: 计算真实相邻跨说话人对的差异-时间斜率。
    置换: 打乱某一说话人的特征值顺序, 重新计算相邻对差异和斜率。
    p值: 置换斜率 <= 观测斜率的比例 (单侧, 趋同 = 负斜率)。

    Args:
        feature_df: 单个对话的特征DataFrame
        feature: 特征列名
        speaker_col: 说话人列名
        time_col: 时间列名
        n_perm: 置换次数
        seed: 随机种子

    Returns:
        字典: {
            obs_slope: 观测斜率,
            perm_mean_slope: 置换分布均值,
            p_value: p值,
            significant: 是否显著 (alpha=0.05),
            direction: 趋同方向 ('convergence'/'divergence'/'none')
        }
    """
    df = feature_df.sort_values(time_col).reset_index(drop=True)

    # 计算观测的相邻差异
    obs_pairs = compute_adjacent_diffs(df, feature, speaker_col, time_col)

    if len(obs_pairs) < 3:
        return {
            'obs_slope': np.nan,
            'perm_mean_slope': np.nan,
            'p_value': np.nan,
            'significant': False,
            'direction': 'insufficient_data'
        }

    obs_slope, _, _ = compute_convergence_slope(
        obs_pairs['time_center'].values,
        obs_pairs['diff'].values
    )

    if np.isnan(obs_slope):
        return {
            'obs_slope': np.nan,
            'perm_mean_slope': np.nan,
            'p_value': np.nan,
            'significant': False,
            'direction': 'insufficient_data'
        }

    # 置换检验: 打乱其中一个说话人的特征值
    rng = np.random.default_rng(seed)
    speakers = df[speaker_col].unique()

    if len(speakers) < 2:
        return {
            'obs_slope': obs_slope,
            'perm_mean_slope': np.nan,
            'p_value': np.nan,
            'significant': False,
            'direction': 'insufficient_data'
        }

    # 选择第一个说话人进行打乱
    spk_a = speakers[0]
    spk_a_mask = df[speaker_col] == spk_a
    spk_a_values = df.loc[spk_a_mask, feature].values.copy()

    perm_slopes = np.zeros(n_perm)

    for i in range(n_perm):
        # 打乱说话人A的特征值
        shuffled_values = rng.permutation(spk_a_values)
        df_perm = df.copy()
        df_perm.loc[spk_a_mask, feature] = shuffled_values

        # 重新计算相邻差异和斜率
        perm_pairs = compute_adjacent_diffs(df_perm, feature, speaker_col, time_col)
        if len(perm_pairs) < 3:
            perm_slopes[i] = 0.0
            continue

        slope, _, _ = compute_convergence_slope(
            perm_pairs['time_center'].values,
            perm_pairs['diff'].values
        )
        perm_slopes[i] = slope if np.isfinite(slope) else 0.0

    # p值: 趋同检验 (单侧, 观测斜率越负越好)
    # p = proportion of permuted slopes <= observed slope
    p_value = np.mean(perm_slopes <= obs_slope)

    # 判断方向
    if p_value < 0.05 and obs_slope < 0:
        direction = 'convergence'
    elif obs_slope > 0:
        # 反向检验: 趋离
        p_diverge = np.mean(perm_slopes >= obs_slope)
        if p_diverge < 0.05:
            direction = 'divergence'
        else:
            direction = 'none'
    else:
        direction = 'none'

    return {
        'obs_slope': obs_slope,
        'perm_mean_slope': np.mean(perm_slopes),
        'perm_std_slope': np.std(perm_slopes),
        'p_value': p_value,
        'significant': p_value < 0.05,
        'direction': direction
    }


def analyze_convergence_all(feature_df: pd.DataFrame,
                            features: List[str],
                            speaker_col: str = 'speaker',
                            time_col: str = 'start',
                            file_col: str = 'file',
                            n_perm: int = 500,
                            alpha: float = 0.05,
                            callback: Optional[Callable[[float, str], None]] = None
                            ) -> pd.DataFrame:
    """主函数: 分析所有文件×特征的趋同性

    对每个对话文件中的每个特征, 计算趋同/趋离指标。

    Args:
        feature_df: 特征DataFrame, 包含列 [file, speaker, start, end, feature1, ...]
        features: 要分析的特征列名列表
        speaker_col: 说话人列名
        time_col: 时间列名
        file_col: 文件/对话标识列名
        n_perm: 置换次数
        alpha: 显著性水平
        callback: 进度回调函数

    Returns:
        结果DataFrame: [file, feature, slope, p_value, significant, direction]
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
            for feature in features:
                results.append({
                    'file': file_id,
                    'feature': feature,
                    'slope': np.nan,
                    'p_value': np.nan,
                    'significant': False,
                    'direction': 'insufficient_speakers'
                })
            completed += len(features)
            continue

        for feature in features:
            completed += 1

            if feature not in feature_df.columns:
                logger.warning(f"特征 '{feature}' 不在DataFrame中, 跳过")
                results.append({
                    'file': file_id,
                    'feature': feature,
                    'slope': np.nan,
                    'p_value': np.nan,
                    'significant': False,
                    'direction': 'missing_feature'
                })
                continue

            # 执行置换检验
            test_result = permutation_test_convergence(
                file_data, feature,
                speaker_col=speaker_col,
                time_col=time_col,
                n_perm=n_perm,
                seed=42
            )

            results.append({
                'file': file_id,
                'feature': feature,
                'slope': test_result['obs_slope'],
                'p_value': test_result['p_value'],
                'significant': test_result['p_value'] < alpha if not np.isnan(
                    test_result['p_value']) else False,
                'direction': test_result['direction']
            })

            if callback:
                progress = 100.0 * completed / total_tasks
                callback(progress, f"趋同性分析: {file_id} - {feature}")

    if callback:
        callback(100.0, "趋同性分析完成")

    result_df = pd.DataFrame(results)
    return result_df
