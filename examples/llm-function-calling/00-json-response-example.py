import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from enum import Enum
from typing import Any

import rich
from nos.client import Client
from nos.common import tqdm
from pydantic import BaseModel, conint, constr
from tenacity import retry, retry_if_exception_type, stop_after_attempt


def _generate(model, query: str, schema: str, max_new_tokens: int = 1024):
    return model.generate(prompts=query, max_new_tokens=max_new_tokens, schema=schema)


@retry(stop=stop_after_attempt(3), retry=retry_if_exception_type(TimeoutError))
def generate(model, query: str, schema: str = None):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_generate, model, query, schema)
        try:
            return future.result(timeout=10)
        except TimeoutError:
            print("TimeoutError (timeout=10s), retrying ...")
            future.cancel()
            raise TimeoutError("Function took too long to complete. Retrying...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        "-m",
        type=str,
        required=False,
        default="mistralai/Mistral-7B-v0.1-json",
        choices=[
            "mistralai/Mistral-7B-v0.1-json",
            "Trelis/Llama-2-7b-chat-hf-function-calling-json",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0-json",
        ],
    )
    args = parser.parse_args()

    client = Client("[::]:50051")

    models = client.ListModels()
    if args.model_id not in models:
        raise ValueError(f"Invalid model_id: {args.model_id}, available models: {models}")
    model = client.Module(args.model_id)

    class Armor(str, Enum):
        leather = "leather"
        chainmail = "chainmail"
        plate = "plate"

    class Character1(BaseModel):
        name: str
        age: int
        armor: Armor
        strength: int

    class Character2(BaseModel):
        name: constr(max_length=10)
        age: conint(gt=18, lt=99)
        armor: Armor
        strength: conint(gt=1, lt=100)

    QUERY = """
    Generate a new realistic character for my awesome game emulating Game of Thrones:

    name, age (between 1 and 99), armor and strength.
    """

    schema: str = json.dumps(Character1.model_json_schema())
    response: Any = generate(model, QUERY, schema=schema)  # warmup
    for idx in tqdm(range(50), disable=True):
        response: Any = generate(model, QUERY, schema=schema)
        try:
            character = Character1(**response)
            rich.print(f"{idx+1} [green]{character}[/green]")
        except Exception as e:
            rich.print(f"{idx+1} [yellow]{response}[/yellow], {e}")
