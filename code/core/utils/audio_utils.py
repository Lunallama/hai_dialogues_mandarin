"""音频处理工具"""

import numpy as np


def load_audio(path, sr=16000):
    """Load audio file, return (signal, sample_rate).

    Tries soundfile first, falls back to librosa.

    Args:
        path: Path to audio file.
        sr: Target sample rate (default 16000).

    Returns:
        Tuple of (signal as numpy array, sample_rate as int).
    """
    try:
        import soundfile as sf
        signal, file_sr = sf.read(path, dtype='float64')

        # Convert to mono if stereo
        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)

        # Resample if needed
        if file_sr != sr:
            import librosa
            signal = librosa.resample(signal, orig_sr=file_sr, target_sr=sr)

        return signal, sr

    except Exception:
        import librosa
        signal, file_sr = librosa.load(path, sr=sr, mono=True)
        return signal, file_sr


def get_audio_segment(signal, sr, start_time, end_time):
    """Extract a time segment from audio array.

    Args:
        signal: Audio signal as numpy array.
        sr: Sample rate.
        start_time: Start time in seconds.
        end_time: End time in seconds.

    Returns:
        Numpy array of the audio segment.
    """
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Clamp to valid range
    start_sample = max(0, start_sample)
    end_sample = min(len(signal), end_sample)

    return signal[start_sample:end_sample]


def get_audio_duration(path):
    """Get audio file duration in seconds.

    Args:
        path: Path to audio file.

    Returns:
        Duration in seconds (float).
    """
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.duration
    except Exception:
        try:
            import librosa
            duration = librosa.get_duration(filename=path)
            return duration
        except Exception:
            # Last resort: load the file and compute duration
            signal, sr = load_audio(path)
            return len(signal) / sr
