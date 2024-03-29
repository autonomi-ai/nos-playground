import time
from dataclasses import dataclass
from threading import Thread
from typing import Any, Dict, Iterable, List

import torch
from nos.hub import HuggingFaceHubConfig, hf_login
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


@dataclass(frozen=True)
class StreamingChatConfig(HuggingFaceHubConfig):
    """Streaming chat model configuration."""

    max_new_tokens: int = 2048
    """Maximum number of tokens to generate."""

    max_input_token_length: int = 4096
    """Maximum number of tokens in the input."""

    compute_dtype: str = "float16"
    """Compute type for the model."""

    needs_auth: bool = False
    """Whether the model needs authentication."""

    additional_kwargs: Dict[str, Any] = None
    """Additional keyword arguments to pass to the model."""

    chat_template: str = None
    """Chat template to use for the model."""


class StreamingChat:
    configs = {
        "mlabonne/phixtral-2x2_8": StreamingChatConfig(
            model_name="mlabonne/phixtral-2x2_8",
            compute_dtype="float16",
            additional_kwargs={"load_in_4bit": True, "trust_remote_code": True, "torch_dtype": "auto"},
        ),
        "mlabonne/phixtral-4x2_8": StreamingChatConfig(
            model_name="mlabonne/phixtral-4x2_8",
            compute_dtype="float16",
            additional_kwargs={"load_in_4bit": True, "trust_remote_code": True, "torch_dtype": "auto"},
        ),
    }

    def __init__(self, model_name: str = "mlabonne/phixtral-4x2_8"):
        from nos.logging import logger

        try:
            self.cfg = StreamingChat.configs[model_name]
        except KeyError:
            raise ValueError(
                f"Invalid model_name: {model_name}, available models: {StreamingChatConfig.configs.keys()}"
            )

        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)

        # Load the model and tokenizer
        token = hf_login() if self.cfg.needs_auth else None
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            token=token,
            **(self.cfg.additional_kwargs or {}),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, token=token, trust_remote_code=True)
        self.logger = logger

    @torch.inference_mode()
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        num_beams: int = 1,
    ) -> Iterable[str]:
        """Chat with the model."""
        self.logger.debug(f"Conversation: {messages}")
        input_ids = self.tokenizer.apply_chat_template(
            messages, chat_template=self.cfg.chat_template, return_tensors="pt"
        )
        if input_ids.shape[1] > self.cfg.max_input_token_length:
            input_ids = input_ids[:, -self.cfg.max_input_token_length :]
            self.logger.warning(
                f"Trimmed input from conversation as it was longer than {self.cfg.max_input_token_length} tokens."
            )
        input_ids = input_ids.to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, timeout=180.0, skip_prompt=True, skip_special_tokens=True)
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

        start_t = None
        for idx, text in enumerate(streamer):
            yield text
            # We only measure the time after the first token is generated
            if start_t is None:
                start_t = time.perf_counter()
            if idx > 0:
                self.logger.debug(
                    f"""tok/s={idx / (time.perf_counter() - start_t):.2f}, """
                    f"""memory={torch.cuda.memory_allocated(device=self.model.device) / 1024 ** 2:.2f} MB, """
                    f"""allocated={torch.cuda.max_memory_allocated(device=self.model.device) / 1024 ** 2:.2f} MB, """
                    f"""peak={torch.cuda.max_memory_reserved(device=self.model.device) / 1024 ** 2:.2f} MB, """
                )
