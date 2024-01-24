from typing import Any, List, Union

import outlines
import torch
from nos.models.llm import LLM as _LLM
from nos.models.llm import LLMConfig


def llm_configs():
    return {f"{k}-json": v for k, v in _LLM.configs.items()}


class LLMJsonMode:
    configs = {
        "mistralai/Mistral-7B-v0.1-json": LLMConfig(
            model_name="mistralai/Mistral-7B-v0.1",
            compute_dtype="float16",
        ),
        "Trelis/Llama-2-7b-chat-hf-function-calling-json": LLMConfig(
            model_name="Trelis/Llama-2-7b-chat-hf-function-calling",
            compute_dtype="float16",
            additional_kwargs={"trust_remote_code": True},
        ),
        **llm_configs(),
    }

    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1-json"):
        try:
            self.cfg = LLMJsonMode.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {LLMJsonMode.configs.keys()}")

        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)

        model_kwargs = {"torch_dtype": getattr(torch, self.cfg.compute_dtype)}
        if self.cfg.additional_kwargs is not None:
            model_kwargs.update(self.cfg.additional_kwargs)
        self.model = outlines.models.transformers(
            self.cfg.model_name,
            device=self.device_str,
            model_kwargs=model_kwargs,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 1024,
        schema: str = None,
    ) -> Any:
        """Generate JSON responses given the model."""
        if schema is None:
            raise ValueError("schema must be provided")
        generator = outlines.generate.json(self.model, schema_object=schema)
        response = generator(prompts=prompts, max_tokens=max_new_tokens)
        return response
