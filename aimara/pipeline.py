"""Composition of ROB speech processing and ERC emotion inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from .erc_features import ERCFeatureBridge
from .erc_model import ERCEmotionModel, EmotionPrediction
from .offline_processor import OfflineROBProcessor


class AIMARAPipeline:
    """Run speech segmentation, transcription, feature extraction, and ERC."""

    def __init__(
        self,
        speech_processor: OfflineROBProcessor,
        feature_bridge: ERCFeatureBridge,
        emotion_model: ERCEmotionModel,
    ) -> None:
        self.speech_processor = speech_processor
        self.feature_bridge = feature_bridge
        self.emotion_model = emotion_model

    def process_video(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> list[EmotionPrediction]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        segments = self.speech_processor.process_video(video_path, output_dir)
        if not segments:
            predictions: list[EmotionPrediction] = []
        else:
            features = self.feature_bridge.build(segments)
            predictions = self.emotion_model.predict(features)

        result_path = output_dir / "predictions.json"
        temporary_path = result_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps([item.to_dict() for item in predictions], indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(result_path)
        return predictions
