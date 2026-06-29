"""TextGrid 文件读写工具，使用 tgt 库"""

import csv
import tgt


def write_textgrid(path, segments):
    """Write a Praat TextGrid file from a list of segment dicts.

    Each segment dict should have keys: start, end, text, speaker.
    Creates one interval tier per unique speaker.

    Args:
        path: Output file path for the TextGrid.
        segments: List of dicts with keys 'start', 'end', 'text', 'speaker'.
    """
    textgrid = tgt.core.TextGrid()

    # Group segments by speaker
    speakers = {}
    for seg in segments:
        speaker = seg.get('speaker', 'unknown')
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append(seg)

    # Create one tier per speaker
    for speaker, segs in speakers.items():
        tier = tgt.core.IntervalTier(
            start_time=min(s['start'] for s in segs),
            end_time=max(s['end'] for s in segs),
            name=speaker
        )
        for s in sorted(segs, key=lambda x: x['start']):
            interval = tgt.core.Interval(
                start_time=s['start'],
                end_time=s['end'],
                text=s.get('text', '')
            )
            tier.add_interval(interval)
        textgrid.add_tier(tier)

    tgt.io.write_to_file(textgrid, path, format='long', encoding='utf-8')


def read_textgrid(path):
    """Read a TextGrid and return list of segment dicts.

    Args:
        path: Path to the TextGrid file.

    Returns:
        List of dicts with keys: start, end, text, speaker.
    """
    textgrid = tgt.io.read_textgrid(path, encoding='utf-8')
    segments = []

    for tier in textgrid.tiers:
        speaker = tier.name
        for interval in tier.intervals:
            text = interval.text.strip()
            if text:  # skip empty intervals
                segments.append({
                    'start': interval.start_time,
                    'end': interval.end_time,
                    'text': text,
                    'speaker': speaker,
                })

    # Sort by start time
    segments.sort(key=lambda x: x['start'])
    return segments


def segments_to_csv(segments, csv_path):
    """Write segments to CSV with columns: start, end, speaker, text, sentence_type.

    Args:
        segments: List of segment dicts.
        csv_path: Output CSV file path.
    """
    fieldnames = ['start', 'end', 'speaker', 'text', 'sentence_type']

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for seg in segments:
            row = {
                'start': seg.get('start', ''),
                'end': seg.get('end', ''),
                'speaker': seg.get('speaker', ''),
                'text': seg.get('text', ''),
                'sentence_type': seg.get('sentence_type', ''),
            }
            writer.writerow(row)


def csv_to_segments(csv_path):
    """Read CSV back to segment list.

    Args:
        csv_path: Path to CSV file with columns: start, end, speaker, text, sentence_type.

    Returns:
        List of segment dicts.
    """
    segments = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seg = {
                'start': float(row.get('start', row.get('start_time', 0))),
                'end': float(row.get('end', row.get('end_time', 0))),
                'speaker': row.get('speaker', ''),
                'text': row.get('text', ''),
            }
            if row.get('sentence_type'):
                seg['sentence_type'] = row['sentence_type']
            segments.append(seg)

    return segments
