"""MFCC 特征提取 (无 librosa 依赖，使用 scipy + numpy)"""

import numpy as np
import soundfile as sf
from scipy.fftpack import dct
from scipy.signal import get_window


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filter_bank(n_filters, n_fft, sr):
    """创建 Mel 滤波器组"""
    low_mel = _hz_to_mel(0)
    high_mel = _hz_to_mel(sr / 2)
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        for j in range(left, center):
            if center != left:
                filters[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                filters[i, j] = (right - j) / (right - center)

    return filters


def _compute_mfcc(signal, sr, n_mfcc=13, n_fft=2048, hop_length=512, n_mels=128):
    """从信号计算 MFCC

    Args:
        signal: 一维音频信号 (float32)
        sr: 采样率
        n_mfcc: MFCC 系数数量
        n_fft: FFT 窗口长度
        hop_length: 帧移
        n_mels: Mel 滤波器数量

    Returns:
        mfccs: shape (n_mfcc, n_frames)
    """
    # 预加重
    signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

    # 分帧
    frame_length = n_fft
    num_frames = 1 + (len(signal) - frame_length) // hop_length
    if num_frames <= 0:
        return np.zeros((n_mfcc, 0))

    indices = (np.arange(frame_length)[None, :] +
               np.arange(num_frames)[:, None] * hop_length)
    frames = signal[indices]

    # 加窗
    window = get_window('hann', frame_length)
    frames = frames * window

    # FFT -> 功率谱
    mag = np.abs(np.fft.rfft(frames, n=n_fft))
    power_spec = (mag ** 2) / n_fft

    # Mel 滤波
    mel_basis = _mel_filter_bank(n_mels, n_fft, sr)
    mel_spec = np.dot(power_spec, mel_basis.T)

    # 对数
    mel_spec = np.where(mel_spec > 1e-10, mel_spec, 1e-10)
    log_mel = np.log(mel_spec)

    # DCT -> MFCC
    mfccs = dct(log_mel, type=2, axis=1, norm='ortho')[:, :n_mfcc]

    return mfccs.T  # (n_mfcc, n_frames)


def extract_mfcc(audio_path, start, end, n_mfcc=13, sr=16000):
    """Extract MFCC features for a segment.

    Args:
        audio_path: Path to audio file.
        start: Start time in seconds.
        end: End time in seconds.
        n_mfcc: Number of MFCC coefficients to extract (default 13).
        sr: Sample rate for loading audio (default 16000).

    Returns:
        Dict with keys mfcc_1 through mfcc_{n_mfcc}, each containing the mean
        value of that coefficient over the segment. Returns NaN values if
        extraction fails.
    """
    nan_result = {f'mfcc_{i+1}': np.nan for i in range(n_mfcc)}

    try:
        duration = end - start
        if duration <= 0:
            return nan_result

        # 使用 soundfile 读取指定片段
        info = sf.info(audio_path)
        file_sr = info.samplerate
        start_frame = int(start * file_sr)
        frames_to_read = int(duration * file_sr)

        if frames_to_read <= 0:
            return nan_result

        with sf.SoundFile(audio_path) as f:
            f.seek(start_frame)
            signal = f.read(frames_to_read, dtype='float32')

        # 转单声道
        if signal.ndim > 1:
            signal = signal.mean(axis=1)

        if len(signal) == 0:
            return nan_result

        # 如果文件采样率与目标不同，简单重采样
        if file_sr != sr:
            from scipy.signal import resample
            num_samples = int(len(signal) * sr / file_sr)
            signal = resample(signal, num_samples).astype(np.float32)

        # 提取 MFCC
        mfccs = _compute_mfcc(signal, sr, n_mfcc=n_mfcc)

        if mfccs.shape[1] == 0:
            return nan_result

        mfcc_means = np.mean(mfccs, axis=1)

        result = {}
        for i in range(n_mfcc):
            result[f'mfcc_{i+1}'] = float(mfcc_means[i])

        return result

    except Exception:
        return nan_result
