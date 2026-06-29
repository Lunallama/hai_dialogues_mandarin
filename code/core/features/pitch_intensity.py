"""使用 parselmouth (Praat) 提取基频和音强"""

import numpy as np
import parselmouth
from parselmouth.praat import call


def extract_pitch_features(audio_path, start, end, min_pitch=75, max_pitch=600):
    """Extract pitch features for a segment.

    Uses parselmouth (Praat) to compute fundamental frequency (F0) statistics.

    Args:
        audio_path: Path to audio file.
        start: Start time in seconds.
        end: End time in seconds.
        min_pitch: Minimum pitch floor in Hz (default 75).
        max_pitch: Maximum pitch ceiling in Hz (default 600).

    Returns:
        Dict with keys: pitch_mean, pitch_max, pitch_min, pitch_std (all in Hz).
        Returns NaN values if extraction fails or no voiced frames found.
    """
    nan_result = {
        'pitch_mean': np.nan,
        'pitch_max': np.nan,
        'pitch_min': np.nan,
        'pitch_std': np.nan,
    }

    try:
        snd = parselmouth.Sound(audio_path)
        snd_segment = snd.extract_part(start, end)
        pitch = snd_segment.to_pitch(pitch_floor=min_pitch, pitch_ceiling=max_pitch)
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]  # remove unvoiced frames

        if len(pitch_values) == 0:
            return nan_result

        return {
            'pitch_mean': float(np.mean(pitch_values)),
            'pitch_max': float(np.max(pitch_values)),
            'pitch_min': float(np.min(pitch_values)),
            'pitch_std': float(np.std(pitch_values)),
        }

    except Exception:
        return nan_result


def extract_intensity_features(audio_path, start, end):
    """Extract intensity features for a segment.

    Uses parselmouth (Praat) to compute intensity statistics.

    Args:
        audio_path: Path to audio file.
        start: Start time in seconds.
        end: End time in seconds.

    Returns:
        Dict with keys: intensity_mean, intensity_max, intensity_min (all in dB).
        Returns NaN values if extraction fails.
    """
    nan_result = {
        'intensity_mean': np.nan,
        'intensity_max': np.nan,
        'intensity_min': np.nan,
    }

    try:
        snd = parselmouth.Sound(audio_path)
        snd_segment = snd.extract_part(start, end)
        intensity = snd_segment.to_intensity()

        intensity_mean = call(intensity, "Get mean", 0, 0, "dB")
        intensity_max = call(intensity, "Get maximum", 0, 0, "Parabolic")
        intensity_min = call(intensity, "Get minimum", 0, 0, "Parabolic")

        return {
            'intensity_mean': float(intensity_mean) if not np.isnan(intensity_mean) else np.nan,
            'intensity_max': float(intensity_max) if not np.isnan(intensity_max) else np.nan,
            'intensity_min': float(intensity_min) if not np.isnan(intensity_min) else np.nan,
        }

    except Exception:
        return nan_result


def extract_all_segments(audio_path, segments):
    """Extract pitch and intensity features for all segments.

    Args:
        audio_path: Path to audio file.
        segments: List of segment dicts with at least 'start' and 'end' keys.

    Returns:
        List of feature dicts, one per segment. Each dict contains all pitch
        and intensity features merged together, plus the original segment info.
    """
    results = []

    for seg in segments:
        start = seg['start']
        end = seg['end']

        pitch_feats = extract_pitch_features(audio_path, start, end)
        intensity_feats = extract_intensity_features(audio_path, start, end)

        # Merge all features with original segment info
        features = {}
        features.update(seg)
        features.update(pitch_feats)
        features.update(intensity_feats)
        results.append(features)

    return results
