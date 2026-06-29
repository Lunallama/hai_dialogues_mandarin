"""Gemini ASR引擎 - 使用Google Gemini API"""

import os
import json
import logging
import time
from typing import List, Dict, Optional, Callable

logger = logging.getLogger(__name__)


class GeminiEngine:
    """基于Google Gemini API的ASR引擎

    利用Gemini的多模态能力进行语音转录，
    支持说话人识别和时间戳标注。
    """

    # 转录提示词模板
    TRANSCRIPTION_PROMPT = """Please transcribe this audio file with precise timestamps and speaker identification.

Requirements:
1. Identify different speakers and label them as SPEAKER_00, SPEAKER_01, etc.
2. Provide start and end timestamps (in seconds) for each utterance.
3. Transcribe the speech content accurately.
4. Language: {language_instruction}

Return the result as a JSON array with this exact format:
[
  {{"start": 0.0, "end": 2.5, "text": "transcribed text here", "speaker": "SPEAKER_00"}},
  {{"start": 2.8, "end": 5.1, "text": "next utterance", "speaker": "SPEAKER_01"}}
]

Important:
- Timestamps should be in seconds (float)
- Include ALL speech content, do not skip any parts
- If you cannot determine the exact timestamp, estimate based on audio position
- Return ONLY the JSON array, no other text
"""

    def __init__(self, api_key: str):
        """初始化Gemini引擎

        Args:
            api_key: Google Gemini API密钥
        """
        if not api_key:
            raise ValueError("Gemini API key不能为空")
        self.api_key = api_key
        self._client = None

    def _ensure_client(self):
        """延迟初始化Gemini客户端"""
        if self._client is None:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "请安装google-generativeai: pip install google-generativeai"
                )

            genai.configure(api_key=self.api_key)
            self._client = genai
            logger.info("Gemini API客户端初始化完成")

    def transcribe(self, audio_path: str, language: str = 'auto',
                   callback: Optional[Callable[[float, str], None]] = None) -> List[Dict]:
        """转录音频文件

        Args:
            audio_path: 音频文件路径
            language: 语言设置 'zh'/'en'/'auto'
            callback: 进度回调函数 callback(progress_pct, message)

        Returns:
            分段列表: [{start: float, end: float, text: str, speaker: str}, ...]
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        self._ensure_client()

        if callback:
            callback(0.0, "准备上传音频到Gemini...")

        # 上传音频文件
        audio_file = self._upload_audio(audio_path, callback)

        if callback:
            callback(40.0, "音频上传完成, 等待转录...")

        # 构建提示词
        language_instruction = self._get_language_instruction(language)
        prompt = self.TRANSCRIPTION_PROMPT.format(
            language_instruction=language_instruction
        )

        # 调用Gemini API
        segments = self._call_gemini(audio_file, prompt, callback)

        if callback:
            callback(100.0, f"Gemini转录完成, 共 {len(segments)} 个句子")

        return segments

    def _upload_audio(self, audio_path: str,
                      callback: Optional[Callable] = None):
        """上传音频文件到Gemini

        Args:
            audio_path: 音频文件路径
            callback: 进度回调

        Returns:
            上传后的文件对象
        """
        import google.generativeai as genai

        if callback:
            callback(10.0, "正在上传音频文件...")

        # 确定MIME类型
        mime_type = self._get_mime_type(audio_path)

        # 上传文件
        audio_file = genai.upload_file(
            path=audio_path,
            mime_type=mime_type
        )

        # 等待文件处理完成
        max_wait = 60  # 最长等待60秒
        waited = 0
        while audio_file.state.name == "PROCESSING" and waited < max_wait:
            time.sleep(2)
            waited += 2
            audio_file = genai.get_file(audio_file.name)
            if callback:
                progress = 10.0 + 30.0 * min(waited / max_wait, 1.0)
                callback(progress, f"等待音频处理... ({waited}s)")

        if audio_file.state.name == "FAILED":
            raise RuntimeError(f"Gemini音频处理失败: {audio_file.state.name}")

        return audio_file

    def _call_gemini(self, audio_file, prompt: str,
                     callback: Optional[Callable] = None) -> List[Dict]:
        """调用Gemini API进行转录

        Args:
            audio_file: 已上传的音频文件对象
            prompt: 转录提示词
            callback: 进度回调

        Returns:
            解析后的分段列表
        """
        import google.generativeai as genai

        if callback:
            callback(50.0, "正在调用Gemini进行转录...")

        # 使用Gemini Pro模型
        model = genai.GenerativeModel('gemini-1.5-pro')

        try:
            response = model.generate_content(
                [audio_file, prompt],
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=8192,
                )
            )
        except Exception as e:
            logger.error(f"Gemini API调用失败: {e}")
            raise RuntimeError(f"Gemini API调用失败: {e}") from e

        if callback:
            callback(80.0, "解析Gemini返回结果...")

        # 解析返回的JSON
        segments = self._parse_response(response.text)

        # 清理上传的文件
        try:
            genai.delete_file(audio_file.name)
        except Exception as e:
            logger.warning(f"清理上传文件失败: {e}")

        return segments

    def _parse_response(self, response_text: str) -> List[Dict]:
        """解析Gemini返回的文本为分段列表

        Args:
            response_text: Gemini返回的原始文本

        Returns:
            标准化的分段列表
        """
        # 尝试直接解析JSON
        text = response_text.strip()

        # 去除可能的markdown代码块标记
        if text.startswith('```json'):
            text = text[7:]
        elif text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 尝试找到JSON数组
            start_idx = text.find('[')
            end_idx = text.rfind(']')
            if start_idx != -1 and end_idx != -1:
                try:
                    data = json.loads(text[start_idx:end_idx + 1])
                except json.JSONDecodeError:
                    logger.error(f"无法解析Gemini返回的JSON: {text[:200]}")
                    return self._fallback_parse(response_text)
            else:
                logger.error(f"Gemini返回中未找到JSON数组: {text[:200]}")
                return self._fallback_parse(response_text)

        # 标准化字段
        segments = []
        for item in data:
            if not isinstance(item, dict):
                continue

            seg = {
                'start': float(item.get('start', 0)),
                'end': float(item.get('end', 0)),
                'text': str(item.get('text', '')).strip(),
            }

            # 可选的speaker字段
            if 'speaker' in item:
                seg['speaker'] = str(item['speaker'])

            if seg['text']:  # 只保留有文本的段落
                segments.append(seg)

        return segments

    def _fallback_parse(self, text: str) -> List[Dict]:
        """当JSON解析失败时的备用解析方法

        将整个文本作为一个segment返回。

        Args:
            text: 原始文本

        Returns:
            单个segment的列表
        """
        logger.warning("使用备用解析方法, 时间戳信息可能不准确")
        # 按行分割，尝试提取有用内容
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        if not lines:
            return []

        # 将所有文本合并为一个segment
        return [{
            'start': 0.0,
            'end': 0.0,
            'text': ' '.join(lines),
            'speaker': 'SPEAKER_00'
        }]

    @staticmethod
    def _get_language_instruction(language: str) -> str:
        """获取语言指示"""
        if language == 'zh':
            return "The audio is in Chinese (Mandarin). Transcribe in Chinese characters."
        elif language == 'en':
            return "The audio is in English. Transcribe in English."
        else:
            return "Auto-detect the language and transcribe accordingly."

    @staticmethod
    def _get_mime_type(audio_path: str) -> str:
        """根据文件扩展名确定MIME类型"""
        ext = os.path.splitext(audio_path)[1].lower()
        mime_map = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.flac': 'audio/flac',
            '.ogg': 'audio/ogg',
            '.m4a': 'audio/mp4',
            '.aac': 'audio/aac',
            '.wma': 'audio/x-ms-wma',
            '.webm': 'audio/webm',
        }
        return mime_map.get(ext, 'audio/wav')
