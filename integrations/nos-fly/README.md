# [NOS](https://github.com/autonomi-ai/nos) Inference Server on Fly.io.

First deploy with:
```bash
fly apps create nos-cpu
fly deploy --vm-size performance-8x --vm-memory 16384
```

You can now work with the [gRPC interface](https://github.com/autonomi-ai/nos?tab=readme-ov-file#-text--image-embedding-clip-as-a-service) (50051) at https://nos-cpu.fly.dev.
