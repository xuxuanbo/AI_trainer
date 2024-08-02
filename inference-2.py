from transformers import AutoTokenizer, TextIteratorStreamer, AutoConfig
import torch
import json
from threading import Thread
import math
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback

def main():

    model_name_or_path = "/root/autodl-tmp/lora_model"
    model_name = "/root/autodl-tmp/lora_model"

    load_in_4bit = False
    
    gen_kwargs = {
        'max_new_tokens': 1,
        'num_beams': 1,
        'do_sample': False
    }
    
    config = AutoConfig.from_pretrained(model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto',
    )
    
    config = AutoConfig.from_pretrained(model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    data_path = "/root/instruct_rte_test_data.jsonl"
    #data_path = "/root/instruct_mrpc_test_data.jsonl"
    
    answerlist = []
    with open(data_path, 'r', encoding='utf-8') as file:
        data_lines = [json.loads(line) for line in file]
        for i,line in enumerate(tqdm(data_lines, desc="Processing lines")):
            # if i >= 100: #只测试100条数据
            #     break
            question = line["instruct"]
            input_ids = tokenizer(question, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            gen_kwargs["input_ids"] = input_ids
            with torch.no_grad():
                output = model.generate(**gen_kwargs)
                answer = ""
                for new_text in tokenizer.decode(output[0], skip_special_tokens=True):
                    answer += new_text
                answerlist.append(answer)

    with open('Inference_result.txt', 'w', encoding='utf-8') as f:
        for answer in answerlist:
            f.write(answer + '\n')
        
if __name__ == '__main__':
    main()
