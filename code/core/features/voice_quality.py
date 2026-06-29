"""发声态特征提取：H1-H2, H1-A1, H1-A2, H1-A3, HNR, CPP
支持两种模式：Python内置计算(基于Praat/parselmouth) 或 导入VoiceSauce结果
"""

import numpy as np
import csv


def extract_voice_quality_python(audio_path, start, end):
    """基于 Praat (parselmouth) 的发声态特征提取。

    使用 Praat 的共振峰追踪(Burg法)、cross-correlation F0、
    Harmonicity 和 PowerCepstrogram 来计算可靠的发声态特征。
    包含 Iseli et al. (2007) 带宽校正。

    Features extracted:
        - H1H2: H1*-H2* (校正后的第一、第二谐波幅度差, dB)
        - H1A1: H1*-A1* (校正后的H1与F1处谐波幅度差, dB)
        - H1A2: H1*-A2* (校正后的H1与F2处谐波幅度差, dB)
        - H1A3: H1*-A3* (校正后的H1与F3处谐波幅度差, dB)
        - HNR05: 500Hz以下的谐波噪声比 (dB)
        - HNR15: 1500Hz以下的谐波噪声比 (dB)
        - CPP: 倒谱峰突出度 (dB)

    Args:
        audio_path: Path to audio file.
        start: Start time in seconds.
        end: End time in seconds.

    Returns:
        Dict with keys: H1H2, H1A1, H1A2, H1A3, HNR05, HNR15, CPP.
        Returns NaN values for any feature that cannot be computed.
    """
    nan_result = {
        'H1H2': np.nan,
        'H1A1': np.nan,
        'H1A2': np.nan,
        'H1A3': np.nan,
        'HNR05': np.nan,
        'HNR15': np.nan,
        'CPP': np.nan,
    }

    try:
        import parselmouth
        from parselmouth.praat import call

        # 加载音频并提取片段
        snd = parselmouth.Sound(audio_path)
        snd_seg = snd.extract_part(start, end)

        # 片段太短无法分析
        if snd_seg.duration < 0.05:
            return nan_result

        result = nan_result.copy()

        # ===== 1. F0 估计 (cross-correlation, 比 autocorrelation 更稳健) =====
        pitch = snd_seg.to_pitch_cc(
            pitch_floor=75.0,
            pitch_ceiling=600.0
        )
        f0_values = pitch.selected_array['frequency']
        f0_voiced = f0_values[f0_values > 0]

        if len(f0_voiced) == 0:
            return nan_result

        mean_f0 = float(np.mean(f0_voiced))

        # ===== 2. 共振峰追踪 (Burg 法) =====
        formant = snd_seg.to_formant_burg(
            max_number_of_formants=5,
            maximum_formant=5500.0,
            window_length=0.025,
            pre_emphasis_from=50.0
        )

        # 对所有帧取中位数（比单点更稳健）
        n_frames = call(formant, "Get number of frames")
        f1_vals, f2_vals, f3_vals = [], [], []
        b1_vals, b2_vals, b3_vals = [], [], []

        for i in range(1, n_frames + 1):
            t = call(formant, "Get time from frame number", i)
            f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
            f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
            f3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")

            if not np.isnan(f1) and f1 > 0:
                f1_vals.append(f1)
                b1 = call(formant, "Get bandwidth at time", 1, t, "Hertz", "Linear")
                if not np.isnan(b1):
                    b1_vals.append(b1)
            if not np.isnan(f2) and f2 > 0:
                f2_vals.append(f2)
                b2 = call(formant, "Get bandwidth at time", 2, t, "Hertz", "Linear")
                if not np.isnan(b2):
                    b2_vals.append(b2)
            if not np.isnan(f3) and f3 > 0:
                f3_vals.append(f3)
                b3 = call(formant, "Get bandwidth at time", 3, t, "Hertz", "Linear")
                if not np.isnan(b3):
                    b3_vals.append(b3)

        if not f1_vals or not f2_vals or not f3_vals:
            # 共振峰追踪失败，只能提取 HNR 和 CPP
            result['HNR05'] = _compute_hnr_praat(snd_seg, 500)
            result['HNR15'] = _compute_hnr_praat(snd_seg, 1500)
            result['CPP'] = _compute_cpp_praat(snd_seg)
            return result

        f1_med = float(np.median(f1_vals))
        f2_med = float(np.median(f2_vals))
        f3_med = float(np.median(f3_vals))
        b1_med = float(np.median(b1_vals)) if b1_vals else 80.0
        b2_med = float(np.median(b2_vals)) if b2_vals else 120.0
        b3_med = float(np.median(b3_vals)) if b3_vals else 150.0

        # ===== 3. 频谱谐波幅度 =====
        # 加 Hanning 窗后取频谱
        snd_win = snd_seg.extract_part(
            snd_seg.start_time, snd_seg.end_time,
            window_shape=parselmouth.WindowShape.HANNING,
            relative_width=1.0,
            preserve_times=False
        )
        spectrum = snd_win.to_spectrum()

        # 构建频率数组 (spectrum.xs() 在某些版本不可靠)
        n_bins = spectrum.values.shape[1]
        freqs = np.arange(n_bins) * spectrum.dx
        real_part = spectrum.values[0]
        imag_part = spectrum.values[1]
        power = real_part ** 2 + imag_part ** 2

        # H1: F0 附近的峰值
        h1_db = _get_peak_db(freqs, power, mean_f0, search_hz=mean_f0 * 0.3)
        # H2: 2*F0 附近的峰值
        h2_db = _get_peak_db(freqs, power, 2 * mean_f0, search_hz=mean_f0 * 0.3)

        # A1/A2/A3: 距各共振峰最近的谐波幅度
        a1_freq = _nearest_harmonic(mean_f0, f1_med)
        a2_freq = _nearest_harmonic(mean_f0, f2_med)
        a3_freq = _nearest_harmonic(mean_f0, f3_med)

        a1_db = _get_peak_db(freqs, power, a1_freq, search_hz=mean_f0 * 0.3)
        a2_db = _get_peak_db(freqs, power, a2_freq, search_hz=mean_f0 * 0.3)
        a3_db = _get_peak_db(freqs, power, a3_freq, search_hz=mean_f0 * 0.3)

        # ===== 4. Iseli et al. (2007) 带宽校正 =====
        if not np.isnan(h1_db) and not np.isnan(h2_db):
            h1_corr = (h1_db
                       + _iseli_correction(mean_f0, f1_med, b1_med)
                       + _iseli_correction(mean_f0, f2_med, b2_med)
                       + _iseli_correction(mean_f0, f3_med, b3_med))
            h2_corr = (h2_db
                       + _iseli_correction(2 * mean_f0, f1_med, b1_med)
                       + _iseli_correction(2 * mean_f0, f2_med, b2_med)
                       + _iseli_correction(2 * mean_f0, f3_med, b3_med))
            result['H1H2'] = float(h1_corr - h2_corr)
        else:
            h1_corr = np.nan

        if not np.isnan(h1_db) and not np.isnan(a1_db):
            if np.isnan(h1_corr):
                h1_corr = h1_db
            a1_corr = (a1_db
                       + _iseli_correction(a1_freq, f2_med, b2_med)
                       + _iseli_correction(a1_freq, f3_med, b3_med))
            result['H1A1'] = float(h1_corr - a1_corr)

        if not np.isnan(h1_db) and not np.isnan(a2_db):
            if np.isnan(h1_corr):
                h1_corr = h1_db
            a2_corr = (a2_db
                       + _iseli_correction(a2_freq, f1_med, b1_med)
                       + _iseli_correction(a2_freq, f3_med, b3_med))
            result['H1A2'] = float(h1_corr - a2_corr)

        if not np.isnan(h1_db) and not np.isnan(a3_db):
            if np.isnan(h1_corr):
                h1_corr = h1_db
            a3_corr = (a3_db
                       + _iseli_correction(a3_freq, f1_med, b1_med)
                       + _iseli_correction(a3_freq, f2_med, b2_med))
            result['H1A3'] = float(h1_corr - a3_corr)

        # ===== 5. HNR =====
        result['HNR05'] = _compute_hnr_praat(snd_seg, 500)
        result['HNR15'] = _compute_hnr_praat(snd_seg, 1500)

        # ===== 6. CPP =====
        result['CPP'] = _compute_cpp_praat(snd_seg)

        return result

    except Exception:
        return nan_result


def _get_peak_db(freqs, power, target_freq, search_hz=30.0):
    """在目标频率附近查找功率谱峰值，返回 dB 值。

    Args:
        freqs: 频率数组 (Hz)
        power: 功率数组 (real^2 + imag^2)
        target_freq: 目标频率 (Hz)
        search_hz: 搜索范围 (±Hz)

    Returns:
        峰值功率 (dB)，找不到返回 np.nan
    """
    mask = (freqs >= target_freq - search_hz) & (freqs <= target_freq + search_hz)
    if not np.any(mask):
        return np.nan
    peak_power = np.max(power[mask])
    if peak_power <= 0:
        return np.nan
    return float(10 * np.log10(peak_power + 1e-30))


def _nearest_harmonic(f0, formant_freq):
    """找到距共振峰频率最近的 F0 谐波频率。

    Args:
        f0: 基频 (Hz)
        formant_freq: 目标共振峰频率 (Hz)

    Returns:
        最近的谐波频率 (Hz)
    """
    if f0 <= 0:
        return formant_freq
    harmonic_number = round(formant_freq / f0)
    if harmonic_number < 1:
        harmonic_number = 1
    return harmonic_number * f0


def _iseli_correction(fh, fi, bi):
    """Iseli et al. (2007) 带宽校正。

    计算频率 fh 处的谐波幅度受共振峰 (fi, bi) 影响的校正量。

    Args:
        fh: 谐波频率 (Hz)
        fi: 共振峰频率 (Hz)
        bi: 共振峰带宽 (Hz)

    Returns:
        校正量 (dB)，应加到原始幅度上以去除共振峰影响
    """
    if fi <= 0 or bi <= 0 or fh <= 0:
        return 0.0

    # Iseli (2007) 公式：校正 = 差值形式
    # 20*log10(|H(fh)|) where H is the formant transfer function
    # |H(f)|^2 = 1 / ((1 - (f/fi)^2)^2 + (f*bi/fi^2)^2)  (simplified)
    # 校正 = -10*log10( ((fh^2 - fi^2)^2 + (fh*bi)^2) / (fi^4 + (fh*bi)^2) )
    # 但为了数值稳定，使用标准化形式
    try:
        omega_h = fh / fi
        beta = bi / fi

        numerator = (omega_h ** 2 - 1) ** 2 + (omega_h * beta) ** 2
        denominator = 1 + (omega_h * beta) ** 2

        if numerator <= 0 or denominator <= 0:
            return 0.0

        correction = -10.0 * np.log10(numerator / denominator)
        return float(correction)
    except (ZeroDivisionError, ValueError):
        return 0.0


def _compute_hnr_praat(snd_seg, max_freq_hz):
    """使用 Praat 计算频率限定的 HNR。

    先低通滤波到 max_freq_hz，再用 harmonicity (cc) 计算 HNR。

    Args:
        snd_seg: parselmouth.Sound 对象（音频片段）
        max_freq_hz: 最高频率 (Hz)，如 500 或 1500

    Returns:
        HNR (dB)，失败返回 np.nan
    """
    try:
        from parselmouth.praat import call

        # 低通滤波
        filtered = call(snd_seg, "Filter (pass Hann band)...",
                        0, max_freq_hz, 100.0)

        # 计算 harmonicity
        harmonicity = call(filtered, "To Harmonicity (cc)",
                           0.01,    # time step
                           75.0,    # minimum pitch
                           0.1,     # silence threshold
                           1.0)     # periods per window

        # 提取 HNR 值，排除静音帧 (-200 dB)
        mean_hnr = call(harmonicity, "Get mean", 0, 0)

        if np.isnan(mean_hnr) or mean_hnr == -200:
            return np.nan
        return float(mean_hnr)

    except Exception:
        return np.nan


def _compute_cpp_praat(snd_seg):
    """使用 Praat 的 PowerCepstrogram 计算 CPP (CPPS)。

    Args:
        snd_seg: parselmouth.Sound 对象（音频片段）

    Returns:
        CPPS (dB)，失败返回 np.nan
    """
    try:
        from parselmouth.praat import call

        # 创建 PowerCepstrogram
        power_cepstrogram = call(snd_seg, "To PowerCepstrogram",
                                 60.0,     # pitch floor (Hz)
                                 0.002,    # time step (s)
                                 5000.0,   # maximum frequency (Hz)
                                 50.0)     # pre-emphasis from (Hz)

        # 获取 CPPS (smoothed CPP)
        cpps = call(power_cepstrogram, "Get CPPS",
                    "yes",         # subtract tilt before smoothing
                    0.02,          # time averaging window (s)
                    0.0005,        # quefrency averaging window (s)
                    60.0,          # peak search pitch floor (Hz)
                    330.0,         # peak search pitch ceiling (Hz)
                    0.05,          # tolerance
                    "Parabolic",   # interpolation
                    0.001,         # tilt line quefrency range start (s)
                    0.0,           # tilt line quefrency range end (0=all)
                    "Straight",    # trend line type
                    "Robust")      # fit method

        if np.isnan(cpps):
            return np.nan
        return float(cpps)

    except Exception:
        return np.nan


def import_voicesauce_results(vs_file_path, segments):
    """Import results from a VoiceSauce output file.

    Reads a VoiceSauce CSV output and matches results to segments by time.

    Args:
        vs_file_path: Path to VoiceSauce output file (CSV format).
        segments: List of segment dicts with 'start' and 'end' keys.

    Returns:
        List of feature dicts, one per segment, with voice quality measures.
        Returns NaN values for segments that cannot be matched.
    """
    nan_features = {
        'H1H2': np.nan,
        'H1A1': np.nan,
        'H1A2': np.nan,
        'H1A3': np.nan,
        'HNR05': np.nan,
        'HNR15': np.nan,
        'CPP': np.nan,
    }

    try:
        # Read VoiceSauce output
        vs_data = []
        with open(vs_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vs_data.append(row)

        if not vs_data:
            return [nan_features.copy() for _ in segments]

        # Parse time information from VoiceSauce data
        vs_times = []
        for row in vs_data:
            # VoiceSauce typically has 't_ms' or 'time' columns
            time_val = None
            for time_key in ['t_ms', 'time', 'Time', 'T']:
                if time_key in row:
                    try:
                        time_val = float(row[time_key])
                        # Convert ms to seconds if needed
                        if time_key == 't_ms':
                            time_val /= 1000.0
                    except (ValueError, TypeError):
                        continue
                    break
            vs_times.append(time_val)

        # Map column names to our feature names
        column_mapping = {
            'H1H2': ['H1H2', 'H1-H2', 'H1_H2'],
            'H1A1': ['H1A1', 'H1-A1', 'H1_A1'],
            'H1A2': ['H1A2', 'H1-A2', 'H1_A2'],
            'H1A3': ['H1A3', 'H1-A3', 'H1_A3'],
            'HNR05': ['HNR05', 'HNR_05', 'HNR'],
            'HNR15': ['HNR15', 'HNR_15'],
            'CPP': ['CPP', 'cpp'],
        }

        # Find actual column names in data
        actual_columns = {}
        if vs_data:
            available_cols = set(vs_data[0].keys())
            for feat_name, possible_names in column_mapping.items():
                for col_name in possible_names:
                    if col_name in available_cols:
                        actual_columns[feat_name] = col_name
                        break

        # Match segments to VoiceSauce data by time
        results = []
        for seg in segments:
            seg_start = seg['start']
            seg_end = seg['end']
            seg_features = nan_features.copy()

            # Collect VoiceSauce frames within this segment
            frame_values = {key: [] for key in nan_features.keys()}

            for i, (row, t) in enumerate(zip(vs_data, vs_times)):
                if t is None:
                    continue
                if seg_start <= t <= seg_end:
                    for feat_name, col_name in actual_columns.items():
                        try:
                            val = float(row[col_name])
                            if not np.isnan(val):
                                frame_values[feat_name].append(val)
                        except (ValueError, TypeError, KeyError):
                            continue

            # Average the frame values for each feature
            for feat_name, values in frame_values.items():
                if values:
                    seg_features[feat_name] = float(np.mean(values))

            results.append(seg_features)

        return results

    except Exception:
        return [nan_features.copy() for _ in segments]
