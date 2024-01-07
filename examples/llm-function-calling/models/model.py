import torch
from nos.hub import hf_login
from nos.models.llm import LLM as _LLM
from nos.models.llm import LLMConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMFunctionCalling(_LLM):
    configs = {
        "Trelis/Llama-2-7b-chat-hf-function-calling": LLMConfig(
            model_name="Trelis/Llama-2-7b-chat-hf-function-calling",
            compute_dtype="float16",
            additional_kwargs={"trust_remote_code": True},
        ),
        **_LLM.configs,
    }

    def __init__(self, model_name: str = "Trelis/Llama-2-7b-chat-hf-function-calling"):
        from nos.logging import logger

        try:
            self.cfg = LLMFunctionCalling.configs[model_name]
        except KeyError:
            raise ValueError(
                f"Invalid model_name: {model_name}, available models: {LLMFunctionCalling.configs.keys()}"
            )

        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)

        token = hf_login() if self.cfg.needs_auth else None
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            torch_dtype=getattr(torch, self.cfg.compute_dtype),
            token=token,
            device_map=self.device_str,
            **(self.cfg.additional_kwargs or {}),
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name,
            token=token,
            **(self.cfg.additional_kwargs or {}),
        )
        self.tokenizer.use_default_system_prompt = False
        self.logger = logger
