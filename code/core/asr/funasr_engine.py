"""FunASR引擎 - 阿里达摩院语音识别"""

import os
import logging
from typing import List, Dict, Optional, Callable

logger = logging.getLogger(__name__)


class FunASREngine:
    """基于FunASR的ASR引擎（阿里达摩院）

    支持语音识别、标点恢复和说话人分离。
    """

    def __init__(self, model_name: str = 'paraformer-zh'):
        """初始化FunASR引擎

        Args:
            model_name: 模型名称, 默认 'paraformer-zh' (中文)
                        可选: 'paraformer-en' (英文)
                              'paraformer-zh-streaming' (流式中文)
        """
        self.model = None
        self.model_name = model_name

    def _ensure_model(self):
        """延迟加载模型"""
        if self.model is None:
            try:
                from funasr import AutoModel
            except ImportError:
                raise ImportError(
                    "请安装FunASR: pip install funasr"
                )

            logger.info(f"加载FunASR模型: {self.model_name}")
            self.model = AutoModel(
                model=self.model_name,
                vad_model="fsmn-vad",
                punc_model="ct-punc",
                spk_model="cam++",
            )

    def transcribe(self, audio_path: str,
                   callback: Optional[Callable[[float, str], None]] = None) -> List[Dict]:
        """转录音频文件

        Args:
            audio_path: 音频文件路径
            callback: 进度回调函数 callback(progress_pct, message)

        Returns:
            分段列表: [{start: float, end: float, text: str, speaker: str}, ...]
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        self._ensure_model()

        if callback:
            callback(0.0, "开始FunASR转录...")

        try:
            result = self.model.generate(
                input=audio_path,
                batch_size_s=300
            )
        except Exception as e:
            logger.error(f"FunASR转录失败: {e}")
            raise RuntimeError(f"FunASR转录失败: {e}") from e

        if callback:
            callback(70.0, "解析转录结果...")

        segments = self._parse_result(result)

        if callback:
            callback(100.0, f"FunASR转录完成, 共 {len(segments)} 个句子")

        return segments

    def _parse_result(self, result) -> List[Dict]:
        """解析FunASR的输出结果

        FunASR的输出格式可能因版本不同而有差异，此方法处理多种格式。

        Args:
            result: FunASR generate() 的返回值

        Returns:
            标准化的分段列表
        """
        segments = []

        if not result:
            return segments

        # FunASR返回的是一个列表，每个元素对应一个输入
        for item in result:
            if isinstance(item, dict):
                segments.extend(self._parse_dict_result(item))
            elif isinstance(item, (list, tuple)):
                for sub_item in item:
                    if isinstance(sub_item, dict):
                        segments.extend(self._parse_dict_result(sub_item))
            else:
                # 尝试作为对象处理
                try:
                    segments.extend(self._parse_dict_result(vars(item)))
                except (TypeError, AttributeError):
                    logger.warning(f"无法解析FunASR结果项: {type(item)}")

        return segments

    def _parse_dict_result(self, item: Dict) -> List[Dict]:
        """解析单个字典结果

        Args:
            item: FunASR结果字典

        Returns:
            分段列表
        """
        segments = []

        # 格式1: 包含sentence_info (带时间戳的句子级别结果)
        if 'sentence_info' in item:
            for sent in item['sentence_info']:
                seg = {
                    'start': sent.get('start', 0) / 1000.0,  # ms -> s
                    'end': sent.get('end', 0) / 1000.0,
                    'text': sent.get('text', '').strip(),
                }
                if 'spk' in sent:
                    seg['speaker'] = f"SPEAKER_{sent['spk']:02d}"
                segments.append(seg)
            return segments

        # 格式2: 包含timestamp (词级别时间戳)
        if 'timestamp' in item and 'text' in item:
            text = item['text']
            timestamps = item['timestamp']  # [[start_ms, end_ms], ...]

            if timestamps:
                # 按标点断句
                sentences = self._split_by_punctuation(text, timestamps)
                segments.extend(sentences)
            else:
                # 无时间戳，整段作为一个segment
                segments.append({
                    'start': 0.0,
                    'end': 0.0,
                    'text': text.strip()
                })
            return segments

        # 格式3: 仅有text
        if 'text' in item:
            text = item['text']
            segments.append({
                'start': 0.0,
                'end': 0.0,
                'text': text.strip()
            })
            return segments

        return segments

    def _split_by_punctuation(self, text: str, timestamps: List) -> List[Dict]:
        """按标点符号将文本断句，并对应时间戳

        Args:
            text: 完整文本
            timestamps: 词级别时间戳列表 [[start_ms, end_ms], ...]

        Returns:
            句子级别分段列表
        """
        SENT_ENDINGS = set('。！？.!?；;')

        # 将文本拆分为字符，与timestamps对齐
        # FunASR的timestamp通常与字/词一一对应
        chars = list(text.replace(' ', ''))

        # 如果字符数和时间戳数不匹配，简单处理
        if len(chars) != len(timestamps):
            # 尝试按句号分割
            return self._simple_split(text, timestamps)

        sentences = []
        current_text = ''
        current_start = None

        for i, (char, ts) in enumerate(zip(chars, timestamps)):
            if current_start is None:
                current_start = ts[0] / 1000.0

            current_text += char

            # 检查是否为句子结束
            if char in SENT_ENDINGS or i == len(chars) - 1:
                sentences.append({
                    'start': current_start,
                    'end': ts[1] / 1000.0,
                    'text': current_text.strip()
                })
                current_text = ''
                current_start = None

        return sentences

    def _simple_split(self, text: str, timestamps: List) -> List[Dict]:
        """简单分割，当字符数与时间戳不匹配时使用

        Args:
            text: 完整文本
            timestamps: 时间戳列表

        Returns:
            分段列表
        """
        if not timestamps:
            return [{'start': 0.0, 'end': 0.0, 'text': text.strip()}]

        # 使用整体起止时间
        start = timestamps[0][0] / 1000.0
        end = timestamps[-1][1] / 1000.0

        # 按标点简单分割
        SENT_ENDINGS = '。！？.!?；;'
        import re
        parts = re.split(f'([{re.escape(SENT_ENDINGS)}])', text)

        # 合并标点到前面的文本
        merged = []
        for i, part in enumerate(parts):
            if not part.strip():
                continue
            if part in SENT_ENDINGS and merged:
                merged[-1] += part
            else:
                merged.append(part)

        if not merged:
            return [{'start': start, 'end': end, 'text': text.strip()}]

        # 均匀分配时间
        total_duration = end - start
        segments = []
        for i, sentence in enumerate(merged):
            seg_start = start + total_duration * i / len(merged)
            seg_end = start + total_duration * (i + 1) / len(merged)
            segments.append({
                'start': seg_start,
                'end': seg_end,
                'text': sentence.strip()
            })

        return segments
