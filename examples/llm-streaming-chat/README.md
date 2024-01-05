## LLM Streaming Chat

Start the server via:
```bash
nos serve up -c serve.yaml --http --env-file .env
```

We need to provide the ``.env` file so that the server is able to authenticate with Huggingface to fetch the Llama2 models. Make sure you have been [granted access to the Llama2 models](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) on the Huggingface Hub.
