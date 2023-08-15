import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_utils import set_seed
from tqdm import tqdm

seed = 1024
max_experiment_times = 1
context_length_per_experiment = 1
generate_length_per_experiment = 2048
# context_length_per_experiment = 1
# generate_length_per_experiment = 8192
# context_length_per_experiment = 2048
# generate_length_per_experiment = 1
use_flash_attn = True
quant_type = "bf16" # bf16, int8 or nf4
model_path = "/ark-contexts/data/huggingface/Qwen/Qwen-7B"

set_seed(seed)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

if quant_type == "nf4":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )
elif quant_type == "int8":
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
else:
    quantization_config = None

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0", trust_remote_code=True, bf16=True, use_flash_attn=use_flash_attn, quantization_config=quantization_config).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
model.generation_config.min_length = generate_length_per_experiment + context_length_per_experiment
model.generation_config.max_new_tokens = generate_length_per_experiment

time_costs = []
context_str = '我' * context_length_per_experiment
max_gpu_memory_cost = 0
for _ in tqdm(range(max_experiment_times)):
    inputs = tokenizer(context_str, return_tensors='pt')
    inputs = inputs.to(model.device)
    t1 = time.time()
    pred = model.generate(**inputs)
    time_costs.append(time.time() - t1)
    assert pred.shape[1] == model.generation_config.min_length
    max_gpu_memory_cost = max(max_gpu_memory_cost, torch.cuda.max_memory_allocated())
    torch.cuda.empty_cache()

print("Average generate speed (tokens/s): {}".format((max_experiment_times * generate_length_per_experiment) / sum(time_costs)))
print(f"GPU Memory cost: {max_gpu_memory_cost / 1024 / 1024 / 1024}GB")
print("Experiment setting: ")
print(f"seed = {seed}")
print(f"max_experiment_times = {max_experiment_times}")
print(f"context_length_per_experiment = {context_length_per_experiment}")
print(f"generate_length_per_experiment = {generate_length_per_experiment}")
print(f"use_flash_attn = {use_flash_attn}")
print(f"quant_type = {quant_type}")