"""中英文音节计数工具"""

import re
import unicodedata


def detect_language(text):
    """Detect the language of text based on character distribution.

    Args:
        text: Input text string.

    Returns:
        'zh' for Chinese, 'en' for English, 'mixed' for mixed content.
    """
    if not text or not text.strip():
        return 'en'

    chinese_count = 0
    english_count = 0

    for char in text:
        if '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf':
            chinese_count += 1
        elif char.isalpha():
            english_count += 1

    total = chinese_count + english_count
    if total == 0:
        return 'en'

    zh_ratio = chinese_count / total
    if zh_ratio > 0.7:
        return 'zh'
    elif zh_ratio < 0.3:
        return 'en'
    else:
        return 'mixed'


def count_syllables_chinese(text):
    """Count syllables in Chinese text using pypinyin.

    Each Chinese character corresponds to one syllable.

    Args:
        text: Input Chinese text.

    Returns:
        Number of syllables (int).
    """
    try:
        from pypinyin import pinyin, Style
    except ImportError:
        # Fallback: count Chinese characters directly
        count = 0
        for char in text:
            if '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf':
                count += 1
        return count

    # Count Chinese characters - each is one syllable
    count = 0
    for char in text:
        if '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf':
            count += 1
    return count


def count_syllables_english(text):
    """Count syllables in English text using vowel cluster heuristic.

    Args:
        text: Input English text.

    Returns:
        Number of syllables (int).
    """
    text = text.lower().strip()
    if not text:
        return 0

    # Extract English words only
    words = re.findall(r"[a-z']+", text)
    total = 0

    for word in words:
        syllables = _count_word_syllables(word)
        total += syllables

    return total


def _count_word_syllables(word):
    """Count syllables in a single English word using vowel cluster heuristic.

    Args:
        word: A single English word (lowercase).

    Returns:
        Number of syllables (int, minimum 1 for non-empty words).
    """
    if not word:
        return 0

    word = word.lower()
    vowels = 'aeiouy'
    count = 0
    prev_is_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel

    # Adjust for silent 'e' at end
    if word.endswith('e') and count > 1:
        count -= 1

    # Adjust for common patterns
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        count += 1

    # Adjust for 'ed' ending
    if word.endswith('ed') and len(word) > 3:
        if word[-3] not in 'td':
            count -= 0  # already handled by vowel counting
        # 'ed' after t/d adds a syllable - already counted

    # Minimum 1 syllable per word
    return max(1, count)


def count_syllables(text, language='auto'):
    """Auto-detect language and count syllables.

    If mixed content, count each part separately.

    Args:
        text: Input text.
        language: 'zh', 'en', 'auto', or 'mixed'. Default is 'auto'.

    Returns:
        Total number of syllables (int).
    """
    if not text or not text.strip():
        return 0

    if language == 'auto':
        language = detect_language(text)

    if language == 'zh':
        return count_syllables_chinese(text)
    elif language == 'en':
        return count_syllables_english(text)
    elif language == 'mixed':
        # Count Chinese and English parts separately
        chinese_syllables = 0
        english_parts = []
        current_english = []

        for char in text:
            if '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf':
                chinese_syllables += 1
                if current_english:
                    english_parts.append(''.join(current_english))
                    current_english = []
            else:
                current_english.append(char)

        if current_english:
            english_parts.append(''.join(current_english))

        english_syllables = sum(count_syllables_english(part) for part in english_parts)
        return chinese_syllables + english_syllables
    else:
        # Default to auto detection
        return count_syllables(text, language='auto')
