from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import torch
from nos.hub import HuggingFaceHubConfig
from sentence_transformers import SentenceTransformer, util


@dataclass(frozen=True)
class GTEEmbeddingConfig(HuggingFaceHubConfig):
    """Sentence Embedding model configuration."""

    compute_dtype: str = "float16"
    """Compute type for the model."""


class GTEEmbedding:
    configs = {
        "thenlper/gte-base": GTEEmbeddingConfig(
            model_name="thenlper/gte-base",
            compute_dtype="float16",
        ),
        "thenlper/gte-large": GTEEmbeddingConfig(
            model_name="thenlper/gte-large",
            compute_dtype="float16",
        ),
        "thenlper/gte-small": GTEEmbeddingConfig(
            model_name="thenlper/gte-small",
            compute_dtype="float16",
        ),
        "TaylorAI/gte-tiny": GTEEmbeddingConfig(
            model_name="TaylorAI/gte-tiny",
            compute_dtype="float16",
        ),
    }

    def __init__(self, model_name: str = "TaylorAI/gte-tiny"):
        try:
            self.cfg = GTEEmbedding.configs[model_name]
        except KeyError:
            raise ValueError(
                f"Invalid model_name: {model_name}, available models: {GTEEmbeddingConfig.configs.keys()}"
            )

        if torch.cuda.is_available():
            self.device_str = "cuda"
        else:
            self.device_str = "cpu"
        self.device = torch.device(self.device_str)
        self.model = SentenceTransformer(self.cfg.model_name, device=self.device)

    def embed(self, text: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        output = self.model.encode(text)
        return {"output": output}

    def sentence_similarity(self, sentences: List[str]) -> Dict[str, float]:
        embedding_1 = self.embed(sentences[0])["output"]
        embedding_2 = self.embed(sentences[1])["output"]
        cos_sim = util.pytorch_cos_sim(embedding_1, embedding_2).item()
        return {"output": cos_sim}
