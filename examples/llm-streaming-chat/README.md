## LLM Streaming Chat

Start the server via:
```bash
nos serve up -c serve.tinyllama.yaml --http
```

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
