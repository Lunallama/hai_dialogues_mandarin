"""EI Score - 整合趋同性指数计算

EI = alpha * lag_component + beta * corr_component + gamma * slope_component

其中:
- lag_component = (1 - optimal_lag / L_max)  # 滞后越小越好
- corr_component = peak_correlation          # 相关越高越好
- slope_component = (1 - slope)              # 斜率越负越好（差异减小）

参考: C:\\Users\\15825\\Desktop\\HCI\\07entrainment_scores.ipynb
"""

import logging
from typing import Dict, List, Tuple, Optional
from itertools import product

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_ei(correlation: float, lag: float, slope: float,
                 L_max: int = 5,
                 alpha: float = 1 / 3, beta: float = 1 / 3,
                 gamma: float = 1 / 3) -> float:
    """计算单个EI值 (Entrainment Index)

    EI = alpha * lag_component + beta * corr_component + gamma * slope_component

    各分量:
    - lag_component = 1 - |optimal_lag| / L_max  (滞后越小, 同步性越好)
    - corr_component = |peak_correlation|        (相关越高, 关联越强)
    - slope_component = max(0, -slope) / max_possible_slope  (负斜率表示趋同)

    为简化, slope_component = (1 + slope) 当 slope in [-1, 0], 即归一化到 [0, 1]
    实际使用 sigmoid 归一化: 1 / (1 + exp(slope * scale))

    Args:
        correlation: 峰值相关系数 |r|
        lag: 最优滞后步数 (绝对值)
        slope: 趋同斜率 (负值表示趋同)
        L_max: 最大允许滞后
        alpha: lag分量权重
        beta: correlation分量权重
        gamma: slope分量权重

    Returns:
        EI值 (float), 范围 [0, 1]; 若输入无效则返回 NaN
    """
    # 输入验证
    if any(np.isnan(x) for x in [correlation, lag, slope] if x is not None):
        return np.nan

    if np.isnan(correlation) or np.isnan(lag) or np.isnan(slope):
        return np.nan

    # Lag分量: 滞后越小越好
    abs_lag = abs(lag)
    lag_component = max(0.0, 1.0 - abs_lag / L_max)

    # Correlation分量: 相关越高越好
    corr_component = min(abs(correlation), 1.0)

    # Slope分量: 斜率越负越好 (差异减小 = 趋同)
    # 使用sigmoid归一化, 使得 slope=0 -> 0.5, slope<0 -> >0.5, slope>0 -> <0.5
    # 然后线性缩放到 [0, 1]
    slope_scale = 5.0  # 缩放因子, 控制sigmoid的陡峭程度
    slope_component = 1.0 / (1.0 + np.exp(slope * slope_scale))

    # 加权求和
    ei = alpha * lag_component + beta * corr_component + gamma * slope_component

    # 确保在 [0, 1] 范围内
    ei = np.clip(ei, 0.0, 1.0)

    return float(ei)


def calculate_speaker_ei(synchrony_df: pd.DataFrame,
                         convergence_df: pd.DataFrame,
                         speaker_id: str,
                         features: List[str],
                         L_max: int = 5,
                         alpha: float = 1 / 3, beta: float = 1 / 3,
                         gamma: float = 1 / 3) -> Tuple[float, Dict[str, float]]:
    """计算单个说话人的EI分数

    对一个说话人, 汇总其所有特征的同步性和趋同性结果, 计算综合EI。

    Args:
        synchrony_df: 同步性分析结果 (来自synchrony.analyze_synchrony_all)
                      列: [file, feature, peak_r, peak_lag, p_value, significant, direction]
        convergence_df: 趋同性分析结果 (来自convergence.analyze_convergence_all)
                        列: [file, feature, slope, p_value, significant, direction]
        speaker_id: 说话人标识 (对应file列中的对话标识)
        features: 特征列表
        L_max: 最大滞后
        alpha, beta, gamma: EI各分量权重

    Returns:
        (overall_ei, {feature: ei_value})
        - overall_ei: 所有特征的平均EI
        - feature_eis: 每个特征的EI值
    """
    feature_eis = {}

    # 获取该说话人对应的同步性和趋同性结果
    # speaker_id 可能对应file列
    sync_data = synchrony_df[synchrony_df['file'] == speaker_id] if 'file' in synchrony_df.columns else synchrony_df
    conv_data = convergence_df[convergence_df['file'] == speaker_id] if 'file' in convergence_df.columns else convergence_df

    for feature in features:
        # 从同步性结果获取 peak_r 和 peak_lag
        sync_row = sync_data[sync_data['feature'] == feature]
        if len(sync_row) > 0:
            peak_r = sync_row.iloc[0]['peak_r']
            peak_lag = sync_row.iloc[0]['peak_lag']
        else:
            peak_r = np.nan
            peak_lag = np.nan

        # 从趋同性结果获取 slope
        conv_row = conv_data[conv_data['feature'] == feature]
        if len(conv_row) > 0:
            slope = conv_row.iloc[0]['slope']
        else:
            slope = np.nan

        # 计算EI
        ei = calculate_ei(
            correlation=peak_r if not np.isnan(peak_r) else 0.0,
            lag=peak_lag if not np.isnan(peak_lag) else L_max,
            slope=slope if not np.isnan(slope) else 0.0,
            L_max=L_max,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )
        feature_eis[feature] = ei

    # 计算总体EI (有效特征的平均值)
    valid_eis = [v for v in feature_eis.values() if not np.isnan(v)]
    overall_ei = np.mean(valid_eis) if valid_eis else np.nan

    return overall_ei, feature_eis


def calculate_all_ei_scores(synchrony_df: pd.DataFrame,
                            convergence_df: pd.DataFrame,
                            features: List[str],
                            L_max: int = 5,
                            alpha: float = 1 / 3, beta: float = 1 / 3,
                            gamma: float = 1 / 3) -> pd.DataFrame:
    """计算所有说话人/对话的EI分数

    Args:
        synchrony_df: 同步性分析结果DataFrame
        convergence_df: 趋同性分析结果DataFrame
        features: 特征列表
        L_max: 最大滞后
        alpha, beta, gamma: EI各分量权重

    Returns:
        DataFrame: [speaker_id, overall_ei, overall_ei_z, feature1_ei, feature2_ei, ...]
        (z-score 标准化)
    """
    # 获取所有唯一的文件/对话标识
    all_files = set()
    if 'file' in synchrony_df.columns:
        all_files.update(synchrony_df['file'].unique())
    if 'file' in convergence_df.columns:
        all_files.update(convergence_df['file'].unique())

    if not all_files:
        logger.warning("没有找到任何对话数据")
        return pd.DataFrame()

    records = []

    for file_id in sorted(all_files):
        overall_ei, feature_eis = calculate_speaker_ei(
            synchrony_df=synchrony_df,
            convergence_df=convergence_df,
            speaker_id=file_id,
            features=features,
            L_max=L_max,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )

        record = {
            'speaker_id': file_id,
            'overall_ei': overall_ei,
        }
        for feat, ei_val in feature_eis.items():
            record[f'{feat}_ei'] = ei_val

        records.append(record)

    result_df = pd.DataFrame(records)

    # 计算z-score标准化
    if len(result_df) > 1 and not result_df['overall_ei'].isna().all():
        mean_ei = result_df['overall_ei'].mean()
        std_ei = result_df['overall_ei'].std()
        if std_ei > 1e-10:
            result_df['overall_ei_z'] = (result_df['overall_ei'] - mean_ei) / std_ei
        else:
            result_df['overall_ei_z'] = 0.0
    else:
        result_df['overall_ei_z'] = 0.0

    # 重新排列列顺序
    cols = ['speaker_id', 'overall_ei', 'overall_ei_z']
    feature_cols = [c for c in result_df.columns if c.endswith('_ei') and c != 'overall_ei']
    cols.extend(sorted(feature_cols))
    result_df = result_df[cols]

    return result_df


def optimize_weights(synchrony_df: pd.DataFrame,
                     convergence_df: pd.DataFrame,
                     features: List[str],
                     L_max: int = 5,
                     step: float = 0.1) -> Tuple[float, float, float]:
    """网格搜索优化EI权重组合

    通过遍历所有可能的(alpha, beta, gamma)组合(约束 alpha+beta+gamma=1),
    找到使排名最稳定的权重。

    稳定性标准: 使用留一交叉验证, 计算排名的平均Kendall's tau。

    Args:
        synchrony_df: 同步性分析结果
        convergence_df: 趋同性分析结果
        features: 特征列表
        L_max: 最大滞后
        step: 网格步长 (默认0.1)

    Returns:
        (best_alpha, best_beta, best_gamma): 最优权重组合
    """
    from scipy.stats import kendalltau

    # 生成权重网格 (alpha + beta + gamma = 1)
    weight_grid = []
    steps = np.arange(0.0, 1.0 + step / 2, step)

    for a in steps:
        for b in steps:
            g = 1.0 - a - b
            if g >= -1e-10:  # 允许微小浮点误差
                g = max(0.0, g)
                weight_grid.append((round(a, 2), round(b, 2), round(g, 2)))

    if not weight_grid:
        logger.warning("权重网格为空, 使用默认权重")
        return (1 / 3, 1 / 3, 1 / 3)

    logger.info(f"优化权重: 网格大小 = {len(weight_grid)}")

    # 获取所有文件列表
    all_files = list(synchrony_df['file'].unique()) if 'file' in synchrony_df.columns else []
    n_files = len(all_files)

    if n_files < 3:
        logger.warning(f"对话数量不足({n_files}<3), 使用默认权重")
        return (1 / 3, 1 / 3, 1 / 3)

    best_weights = (1 / 3, 1 / 3, 1 / 3)
    best_stability = -1.0

    for alpha, beta, gamma in weight_grid:
        # 计算完整数据的排名
        full_scores = calculate_all_ei_scores(
            synchrony_df, convergence_df, features,
            L_max=L_max, alpha=alpha, beta=beta, gamma=gamma
        )

        if len(full_scores) < 3:
            continue

        full_rank = full_scores['overall_ei'].rank()

        # 留一交叉验证: 每次去掉一个文件, 计算剩余的排名
        taus = []
        for leave_out_file in all_files:
            # 去掉一个文件
            sync_subset = synchrony_df[synchrony_df['file'] != leave_out_file]
            conv_subset = convergence_df[convergence_df['file'] != leave_out_file]

            subset_scores = calculate_all_ei_scores(
                sync_subset, conv_subset, features,
                L_max=L_max, alpha=alpha, beta=beta, gamma=gamma
            )

            if len(subset_scores) < 2:
                continue

            # 找到重叠的文件并计算排名相关
            overlap = full_scores[full_scores['speaker_id'].isin(
                subset_scores['speaker_id']
            )]

            if len(overlap) < 2:
                continue

            overlap_rank = overlap['overall_ei'].rank()
            subset_rank = subset_scores.set_index('speaker_id').loc[
                overlap['speaker_id'].values, 'overall_ei'
            ].rank()

            tau, _ = kendalltau(overlap_rank.values, subset_rank.values)
            if np.isfinite(tau):
                taus.append(tau)

        # 计算平均稳定性
        if taus:
            mean_tau = np.mean(taus)
            if mean_tau > best_stability:
                best_stability = mean_tau
                best_weights = (alpha, beta, gamma)

    logger.info(
        f"最优权重: alpha={best_weights[0]:.2f}, "
        f"beta={best_weights[1]:.2f}, gamma={best_weights[2]:.2f} "
        f"(stability={best_stability:.4f})"
    )

    return best_weights
