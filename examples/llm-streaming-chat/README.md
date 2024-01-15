## LLM Streaming Chat

Start the server via:
```bash
nos serve up -c serve.yaml --http --env-file .env
```

We need to provide the ``.env` file so that the server is able to authenticate with Huggingface to fetch the Llama2 models. Make sure you have been [granted access to the Llama2 models](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) on the Huggingface Hub.


### Run the tests

```bash
# Run the tests for all models
pytest -sv ./tests/test_chat.py -k test_streaming_chat

# Run the tests for tinyllama + 4-bit AWQ
# Serve the model: nos serve up -c serve.yaml
pytest -sv ./tests/test_chat.py -k test_streaming_chat_tinyllama

# Run the tests for phixtral
# Serve the model: nos serve up -c serve.phixtral.yaml
pytest -sv ./tests/test_chat.py -k test_streaming_chat_phixtral
```
