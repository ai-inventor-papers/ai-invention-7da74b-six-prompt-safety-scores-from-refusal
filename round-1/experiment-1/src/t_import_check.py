import transformers, torch
print("transformers", transformers.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
from transformers import AutoModelForCausalLM, AutoTokenizer
print("Qwen3 classes OK")
import numpy, scipy, sklearn, loguru, psutil, peft, huggingface_hub, safetensors, accelerate
print("all deps OK")