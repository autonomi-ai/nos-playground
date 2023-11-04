from dataclasses import dataclass
from threading import Thread
from typing import List, Optional, Tuple

import torch
from nos.hub import HuggingFaceHubConfig, hf_login
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


@dataclass(frozen=True)
class Llama2ChatConfig(HuggingFaceHubConfig):
    """Llama2 chat model configuration."""

    max_new_tokens: int = 2048
    """Maximum number of tokens to generate."""

    max_input_token_length: int = 4096
    """Maximum number of tokens in the input."""

    compute_dtype: str = "float16"
    """Compute type for the model."""


class Llama2Chat:
    configs = {
        "meta-llama/Llama-2-7b-chat-hf": Llama2ChatConfig(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            compute_dtype="float16",
        ),
        "HuggingFaceH4/zephyr-7b-beta": Llama2ChatConfig(
            model_name="HuggingFaceH4/zephyr-7b-beta",
            compute_dtype="float16",
        ),
    }

    def __init__(self, model_name: str = "HuggingFaceH4/zephyr-7b-beta"):
        from nos.logging import logger

        try:
            self.cfg = Llama2Chat.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {Llama2ChatConfig.configs.keys()}")

        if torch.cuda.is_available():
            self.device_str = "cuda"
        else:
            self.device_str = "cpu"
        self.device = torch.device(self.device_str)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            torch_dtype=getattr(torch, self.cfg.compute_dtype),
            use_auth_token=hf_login(),
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, use_auth_token=hf_login())
        self.tokenizer.use_default_system_prompt = False
        self.logger = logger

    def chat(
        self,
        message: str,
        system_prompt: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        num_beams: int = 1,
    ) -> List[str]:
        """Chat with the model."""
        conversation = []
        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})
        if chat_history:
            for user, assistant in chat_history:
                conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
        conversation.append({"role": "user", "content": message})

        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt")
        if input_ids.shape[1] > self.cfg.max_input_token_length:
            input_ids = input_ids[:, -self.cfg.max_input_token_length :]
            self.logger.warning(
                f"Trimmed input from conversation as it was longer than {self.cfg.max_input_token_length} tokens."
            )
        input_ids = input_ids.to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            {"input_ids": input_ids},
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()
        return ["".join(list(streamer))]
