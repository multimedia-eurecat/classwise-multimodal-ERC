"""End-to-end offline ROB speech processing orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Union

from .audio_io import decode_audio, write_wav
from .domain import SpeechSegment
from .speech_interfaces import Transcriber
from .vad_processing import OfflineVADProcessor

if TYPE_CHECKING:
    import numpy as np


class OfflineROBProcessor:
    """Decode a video and return transcribed ROB VAD speech segments."""

    def __init__(
        self,
        *,
        vad_processor: OfflineVADProcessor,
        transcriber: Transcriber,
        speaker_id: str = "UNKNOWN",
        audio_decoder: Callable[..., "np.ndarray"] = decode_audio,
        wav_writer: Callable[..., Path] = write_wav,
    ) -> None:
        if not speaker_id.strip():
            raise ValueError("speaker_id must not be empty")
        self.vad_processor = vad_processor
        self.transcriber = transcriber
        self.speaker_id = speaker_id
        self._audio_decoder = audio_decoder
        self._wav_writer = wav_writer

    def process_video(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        *,
        conversation_id: Optional[str] = None,
    ) -> List[SpeechSegment]:
        """Process one complete video and materialize each detected segment."""
        source_video = Path(video_path).absolute()
        selected_conversation_id = conversation_id or source_video.stem
        if not selected_conversation_id.strip():
            raise ValueError("conversation_id must not be empty")

        sample_rate = self.vad_processor.config.sample_rate
        audio = self._audio_decoder(source_video, sample_rate=sample_rate)
        regions = self.vad_processor.process(audio)

        audio_dir = Path(output_dir).absolute() / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        segments: List[SpeechSegment] = []

        for segment_id, region in enumerate(regions):
            segment_audio = audio[region.start_sample : region.end_sample].copy()
            transcript = self.transcriber.transcribe(segment_audio, sample_rate)
            audio_path = audio_dir / f"segment_{segment_id:04d}.wav"
            self._wav_writer(audio_path, segment_audio, sample_rate=sample_rate)

            segments.append(
                SpeechSegment(
                    conversation_id=selected_conversation_id,
                    segment_id=segment_id,
                    source_video=source_video,
                    audio_path=audio_path,
                    start_time=region.start_time(sample_rate),
                    end_time=region.end_time(sample_rate),
                    speaker_id=self.speaker_id,
                    transcript=transcript.text,
                    language=transcript.language,
                )
            )

        return segments
