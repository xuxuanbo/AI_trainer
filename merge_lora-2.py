from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import os
import math
from loguru import logger



def merge_lora_to_base_model():
    model_name_or_path = "/root/autodl-tmp/gemma-2b"
    model_name = "/root/autodl-tmp/gemma-2b"
    adapter_name_or_path = '/root/autodl-tmp/output/peft_model'
    save_path = '/root/autodl-tmp/lora_model'

    config = AutoConfig.from_pretrained(model_name_or_path)
    model_max_length = 4096

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto',
    )
    trainable_params_file = os.path.join(adapter_name_or_path, "trainable_params.bin")
    if os.path.isfile(trainable_params_file):
        model.load_state_dict(torch.load(trainable_params_file, map_location=model.device), strict=False)
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map={'': 'cpu'})
    model = model.merge_and_unload()
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


if __name__ == '__main__':
    merge_lora_to_base_model()
