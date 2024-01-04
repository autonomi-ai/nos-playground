from dataclasses import dataclass

import torch
from nos.hub import HuggingFaceHubConfig, hf_login
from sentence_transformers import SentenceTransformer, util



@dataclass(frozen=True)
class GTEEmbeddingConfig(HuggingFaceHubConfig):
    """Sentence Embedding model configuration."""

    max_input_token_length: int = 512
    """Maximum number of tokens in the input."""

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
        from nos.logging import logger

        try:
            self.cfg = GTEEmbedding.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {GTEEmbeddingConfig.configs.keys()}")

        if torch.cuda.is_available():
            self.device_str = "cuda"
        else:
            self.device_str = "cpu"
        self.device = torch.device(self.device_str)
        self.model = SentenceTransformer(self.cfg.model_name, device = self.device)
        self.logger = logger


    def embed(self, text, convert_to_tensor = False):
        output = self.model.encode(text, convert_to_tensor=convert_to_tensor)


        return {'output': output}
        


    def sentence_similarity(self, sentences):
        print(sentences)
        embedding_1= self.embed(sentences[0], convert_to_tensor=True)["output"]
        print(embedding_1)
        embedding_2 = self.embed(sentences[1], convert_to_tensor=True)["output"]
        print(embedding_2)
        cos_sim = util.pytorch_cos_sim(embedding_1, embedding_2)
        print(cos_sim)
        print(type(cos_sim))

        return {"output" : cos_sim}