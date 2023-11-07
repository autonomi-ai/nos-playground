from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from nos.hub import TorchHubConfig, hf_login


@dataclass(frozen=True)
class WhisperXConfig(TorchHubConfig):
    """WhisperX model configuration."""

    compute_type: str = "float16"
    """Compute type for the model."""

    chunk_length_s: int = 30
    """Chunk length in seconds."""


class WhisperX:
    configs = {
        "m-bain/whisperx-large-v2": WhisperXConfig(
            repo="m-bain/whisperX",
            model_name="large-v2",
            compute_type="float16",
        ),
    }

    def __init__(self, model_name: str = "m-bain/whisperx-large-v2"):
        import whisperx

        try:
            self.cfg = WhisperX.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {WhisperXConfig.configs.keys()}")

        if torch.cuda.is_available():
            self.device_str = "cuda"
        else:
            self.device_str = "cpu"
        self.device = torch.device(self.device_str)
        self.model = whisperx.load_model(self.cfg.model_name, self.device_str, compute_type=self.cfg.compute_type)
        self._load_audio = whisperx.load_audio
        self._load_align_model = whisperx.load_align_model
        self._align = whisperx.align
        self._assign_word_speakers = whisperx.assign_word_speakers
        self._diarize_model = whisperx.DiarizationPipeline(
            model_name="pyannote/speaker-diarization@2.1", use_auth_token=hf_login(), device=self.device
        )

    def transcribe(
        self,
        path: Path,
        batch_size: int = 24,
        align_output: bool = True,
        diarize_output: bool = True,
        language_code: str = "en",
    ) -> List[Dict[str, Any]]:
        """Transcribe the audio file."""
        audio: np.ndarray = self._load_audio(str(path))
        with torch.inference_mode():
            result: Dict[str, Any] = self.model.transcribe(audio, batch_size=batch_size)
            # Align the output
            if align_output:
                alignment_model, alignment_metadata = self._load_align_model(
                    language_code=language_code, device=self.device_str
                )
                result: Dict[str, Any] = self._align(
                    result["segments"],
                    alignment_model,
                    alignment_metadata,
                    str(path),
                    self.device_str,
                    return_char_alignments=False,
                )
            # Diarize the output
            if diarize_output:
                assert align_output, "align_output must be True when diaryze_output is True"
                diarize_segments: pd.DataFrame = self._diarize_model(audio)
                result: Dict[str, Any] = self._assign_word_speakers(diarize_segments, result)
                assert "segments" in result, "segments must be in result"
        return result
