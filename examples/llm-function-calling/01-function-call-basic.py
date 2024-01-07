"""
Shamelessly adapted from
https://github.com/namuan/llm-playground/blob/main/local-llm-tools-simple.py
"""
import inspect
import json
import os
from typing import Dict, List, get_type_hints

import requests
from rich import print_json


def generate_full_completion(model: str, prompt: str, **kwargs) -> Dict[str, str]:
    params = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 1.0, "stream": False}
    try:
        base_url = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
        base_url = base_url.rstrip("/")
        response = requests.post(
            base_url + "/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(params),
            timeout=60,
        )
        # print(f"🤖 Request: {json.dumps(params)} -> Response: {response.text}")
        response.raise_for_status()
        return json.loads(response.text)
    except requests.RequestException as err:
        return {"error": f"API call error: {str(err)}"}


class Article:
    pass


class Weather:
    pass


class Directions:
    pass


def calculate_mortgage_payment(loan_amount: int, interest_rate: float, loan_term: int) -> float:
    """Get the monthly mortgage payment given an interest rate percentage."""

    # TODO: you must implement this to actually call it later


def get_article_details(
    title: str,
    authors: List[str],
    short_summary: str,
    date_published: str,
    tags: List[str],
) -> Article:
    '''Get article details from unstructured article text.
    date_published: formatted as "MM/DD/YYYY"'''

    # TODO: you must implement this to actually call it later


def get_weather(city: str) -> Weather:
    """Get the current weather given a city."""

    # TODO: you must implement this to actually call it later


def get_directions(start: str, destination: str) -> Directions:
    """Get directions from Google Directions API.
    start: start address as a string including zipcode (if any)
    destination: end address as a string including zipcode (if any)"""

    # TODO: you must implement this to actually call it later


def get_type_name(t):
    name = str(t)
    if "List" in name or "Dict" in name:
        return name
    else:
        return t.__name__


def function_to_json(func):
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    function_info = {
        "name": func.__name__,
        "description": func.__doc__,
        "parameters": {"type": "object", "properties": {}},
        "returns": type_hints.get("return", "void").__name__,
    }

    for name, _ in signature.parameters.items():
        param_type = get_type_name(type_hints.get(name, type(None)))
        function_info["parameters"]["properties"][name] = {"type": param_type}

    return json.dumps(function_info, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Trelis/Llama-2-7b-chat-hf-function-calling")
    args = parser.parse_args()

    B_FUNC, E_FUNC = "<FUNCTIONS>", "</FUNCTIONS>\n\n"
    B_INST, E_INST = "[INST] ", " [/INST]"  # Llama style

    functions_prompt = f"""
You are a helpful assistant designed to output only JSON.
You have access to the following tools:
{B_FUNC}
{function_to_json(get_weather)}
{function_to_json(calculate_mortgage_payment)}
{function_to_json(get_directions)}
{function_to_json(get_article_details)}
{E_FUNC}

{B_INST}
You must follow these instructions:
Always select one or more of the above tools based on the user query
If a tool is found, you must respond in the JSON format matching the following schema:
{{
   "tools": {{
        "tool": "<name of the selected tool>",
        "tool_input": <parameters for the selected tool, matching the tool's JSON schema
   }}
}}
If there are multiple tools required, make sure a list of tools are returned in a JSON array.
If there is no tool that match the user request, you will respond with empty json.
Do not add any additional Notes or Explanations.
{E_INST}

User Query:
    """

    GPT_MODEL = args.model.replace("/", "--")

    prompts = [
        "What's the weather in London, UK?",
        "Determine the monthly mortgage payment for a loan amount of $200,000, an interest rate of 4%, and a loan term of 30 years.",
        "I'm planning a trip to Killington, Vermont (05751) from Hoboken, NJ (07030). Can you get me weather for both locations and directions?",
    ]

    for prompt in prompts:
        print(f"❓{prompt}")
        question = functions_prompt + prompt
        response = generate_full_completion(GPT_MODEL, question)
        print(f"🤖 Response: {response}")
        # TOFIX (spllai): This is a hack to get the JSON response.
        # We should be able to get the JSON response directly from the API
        # with the function-calling models.
        text = response["choices"][0]["message"]["content"]
        try:
            tidy_response = text.strip().replace("\n", "").replace("\\", "")
            print_json(tidy_response)
        except Exception as exc:
            print(f"❌ Unable to decode JSON. {response}, exc={exc}")


if __name__ == "__main__":
    main()
