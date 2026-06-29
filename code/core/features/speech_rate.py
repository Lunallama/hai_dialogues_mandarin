"""语速计算：音节数 / 时长"""

import numpy as np
from core.utils.syllable_count import count_syllables


def calculate_speech_rate(text, start_time, end_time, language='auto'):
    """Calculate speech rate in syllables per second.

    Args:
        text: The text/transcript of the speech segment.
        start_time: Start time of the segment in seconds.
        end_time: End time of the segment in seconds.
        language: Language code ('zh', 'en', 'auto', or 'mixed'). Default 'auto'.

    Returns:
        Speech rate in syllables per second (float).
        Returns np.nan if duration is zero or text is empty.
    """
    if not text or not text.strip():
        return np.nan

    duration = end_time - start_time
    if duration <= 0:
        return np.nan

    syllables = count_syllables(text, language=language)
    if syllables == 0:
        return np.nan

    return syllables / duration
