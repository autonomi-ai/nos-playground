import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from nos.hub import TorchHubConfig


@dataclass(frozen=True)
class WhisperXConfig(TorchHubConfig):
    """WhisperX model configuration."""

    compute_type: str = "float16"
    """Compute type for the model."""

    chunk_length_s: int = 30
    """Chunk length in seconds."""


class WhisperX:
    """WhisperX model for audio transcription.

    Based on https://github.com/m-bain/whisperX
    """

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

        # WhisperX model
        self.model = whisperx.load_model(self.cfg.model_name, self.device_str, compute_type=self.cfg.compute_type)
        # Aligment models are loaded on transcribe call to align to a specific language
        self._load_align_model = whisperx.load_align_model
        self._align = whisperx.align
        # TODO (spillai): Add support for diarization

    def transcribe(
        self,
        path: Path,
        batch_size: int = 24,
        align_output: bool = True,
        language_code: str = "en",
    ) -> List[Dict[str, Any]]:
        """Transcribe the audio file."""
        with torch.inference_mode():
            # Transcribe the first chunks before
            result: Dict[str, Any] = self.model.transcribe(str(path), batch_size=batch_size)

            if align_output:
                # Load alignment model and metadata
                alignment_model, alignment_metadata = self._load_align_model(
                    language_code=language_code, device=self.device_str
                )

                result = self._align(
                    result["segments"],
                    alignment_model,
                    alignment_metadata,
                    str(path),
                    self.device_str,
                    return_char_alignments=False,
                )
                # Cleanup alignment model
                del alignment_model
                gc.collect()
                torch.cuda.empty_cache()

        return result
