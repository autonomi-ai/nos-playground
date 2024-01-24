### LLM experiments with JSON mode and Function Calling

This example shows how to use the LLM with JSON mode and Function Calling.

- [x] [`00-json-response-example`](./00-json-response-example.py): Use LLMs with JSON mode (using [`outlines`](https://github.com/outlines-dev/outlines/tree/main/outlines)) to automatically generate pydantic objects with strict data models.
- [x] [`01-llm-function-call-basic`](./01-llm-function-call-basic.py): Use LLMs with Function Calling to call tools / functions.

## 🏃‍♂️ Running the examples

### 00-json-response-example

To start the server, run the following command:

```bash
nos serve up -c serve.outlines.yaml
```

You can then modify and run the `00-json-response-example.py` script to test the JSON mode response for the LLM. We currently support the following models:
- [`TinyLlama/TinyLlama-1.1B-Chat-v0.1`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.1)
- [`mistralai/Mistral-7B-v0.1`](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [`Trelis/Llama-2-7b-chat-hf-function-calling`](https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling)

For this example, you will need `tenacity` and `pydantic>2.5` installed.
```bash
pip install tenacity pydantic>2.5
```

Finally run the script:
```bash
python 00-json-response-example.py
```

You should see the following output:
```bash
(nos-py38) examples/llm-function-calling  [ python 00-json-response-example.py                      [14/50]
TimeoutError (timeout=10s), retrying ...
1 name='Dothraki' age=18 armor=<Armor.leather: 'leather'> strength=3
2 name='GENERIC049' age=60 armor=<Armor.leather: 'leather'> strength=23
3 name='Benny' age=37 armor=<Armor.plate: 'plate'> strength=35
4 name='Bill' age=24 armor=<Armor.leather: 'leather'> strength=9
5 name='sunshine' age=15 armor=<Armor.plate: 'plate'> strength=2
6 name='Hanna' age=12 armor=<Armor.leather: 'leather'> strength=4
7 name='Stark' age=40 armor=<Armor.plate: 'plate'> strength=3
8 name='Boris' age=66 armor=<Armor.leather: 'leather'> strength=100
9 name='harald northpole' age=27 armor=<Armor.chainmail: 'chainmail'> strength=18
10 name='Geralt' age=121 armor=<Armor.chainmail: 'chainmail'> strength=80
11 name='Pusher' age=32 armor=<Armor.chainmail: 'chainmail'> strength=1
12 name='Karel' age=12 armor=<Armor.plate: 'plate'> strength=138
13 name='Severus Snape[..]' age=76 armor=<Armor.plate: 'plate'> strength=21
14 name='lalo' age=82 armor=<Armor.plate: 'plate'> strength=16
...
``````
