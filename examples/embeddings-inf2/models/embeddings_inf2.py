"""Embeddings model accelerated with AWS Neuron (using optimum-neuron)."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Union

import torch
import torch_neuronx
from nos.constants import NOS_CACHE_DIR
from nos.hub import HuggingFaceHubConfig


@dataclass(frozen=True)
class EmbeddingConfig(HuggingFaceHubConfig):
    """Embeddings model configuration."""

    batch_size: int = 1
    """Batch size for the model."""

    sequence_length: int = 384
    """Sequence length for the model."""


def get_neuon_device_count():
    try:
        return torch_neuronx.xla_impl.data_parallel.device_count()
    except (RuntimeError, AssertionError):
        return 0


def _setup_neuron_env():
    from nos.logging import logger

    # print environment for all neuron related variables
    for k, v in os.environ.items():
        if "NEURON" in k:
            logger.debug(f"{k}={v}")
    cores: int = int(os.getenv("NOS_NEURON_CORES", 2))
    logger.info(f"Setting up neuron env with {cores} cores")
    cache_dir = NOS_CACHE_DIR / "neuron"
    os.environ["NEURONX_CACHE"] = "on"
    os.environ["NEURONX_DUMP_TO"] = str(cache_dir)
    os.environ["NEURON_RT_NUM_CORES"] = str(cores)
    os.environ["NEURON_RT_VISIBLE_CORES"] = ",".join([str(i) for i in range(cores)])
    os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer-inference"


class EmbeddingServiceInf2:
    configs = {
        "BAAI/bge-small-en-v1.5": EmbeddingConfig(
            model_name="BAAI/bge-small-en-v1.5",
        ),
    }

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        _setup_neuron_env()

        from nos.logging import logger
        from optimum.neuron import NeuronModelForSentenceTransformers
        from transformers import AutoTokenizer

        try:
            self.cfg = EmbeddingServiceInf2.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {self.configs.keys()}")

        cache_dir = (
            NOS_CACHE_DIR / "neuron" / f"{self.cfg.model_name}-bs-{self.cfg.batch_size}-sl-{self.cfg.sequence_length}"
        )
        if Path(cache_dir).exists():
            logger.info(f"Loading model from {cache_dir}")
            self.model = NeuronModelForSentenceTransformers.from_pretrained(str(cache_dir))
            logger.info(f"Loaded model from {cache_dir}")
        else:
            # Load Transformers model and export it to AWS Inferentia2
            input_shapes = {
                "batch_size": self.cfg.batch_size,
                "sequence_length": self.cfg.sequence_length,
            }
            self.model = NeuronModelForSentenceTransformers.from_pretrained(
                self.cfg.model_name, export=True, **input_shapes
            )
            self.model.save_pretrained(str(cache_dir))
            logger.info(f"Saved model to {cache_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.logger = logger
        self.logger.info(f"Loaded neuron model: {self.cfg.model_name}")

    @torch.inference_mode()
    def __call__(
        self,
        texts: Union[str, List[str]],
    ) -> Iterable[str]:
        """Embed text with the model."""
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
        )
        outputs = self.model(**inputs)
        return outputs.sentence_embedding.cpu().numpy()
