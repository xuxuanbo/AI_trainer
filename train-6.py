import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import default_data_collator, Trainer, TrainingArguments,BitsAndBytesConfig
from torch.utils.data import Dataset
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM


class TaskDataset(Dataset):
    '''
    Dataset for PlanCoder
    '''
    def __init__(self, tokenizer,data, max_length=4096):
        '''
        Args:
            tokenizer: the tokenizer used to tokenize the data
            data: the data to tokenize
            max_length: the maximum length of the input ids
        '''

        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        IGNORE_INDEX = -100

        question = self.data[idx]["instruct"]
        answer = str(self.data[idx]["label"])

        # question_ids are the token ids of the question text.
        # target_ids are the token ids of the target text.
        question_ids = self.tokenizer.encode(question, add_special_tokens=False)
        target_ids = self.tokenizer.encode(answer, add_special_tokens=False)

        input_ids = question_ids + target_ids # input ids for the model
        attention_mask = [1] * len(input_ids) # attention mask for the input
        labels = [IGNORE_INDEX] * len(question_ids) + target_ids # ignore the input tokens

        # pad inputs and labels to the same length
        padding_length = self.max_length - len(input_ids) 
        if padding_length > 0:  # pad on the right
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [IGNORE_INDEX] * padding_length
        elif padding_length < 0:
            raise Exception("Max length exceeded!")
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def create_peft_config(model):
    '''
    Create a PEFT model from a pretrained model
    '''

    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_kbit_training,
    )

    # Define the PEFT config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Prepare the model for int8 training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

if not os.path.exists('/root/autodl-tmp/output'):
    os.mkdir('/root/autodl-tmp/output')

if __name__ == '__main__':

    IGNORE_INDEX = -100

    model_name = "/root/autodl-tmp/gemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f'Loading model from: {model_name}')
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map='auto',
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    ) 

    # load the training data
    datafile = 'instruct_train_data.jsonl'
    logger.info('Loading data: {}'.format(datafile))
    with open(datafile, 'r', encoding='utf-8') as file:
        data_lines = [json.loads(line) for line in file]
    logger.info("there are {} data in dataset".format(len(data_lines)))
    train_dataset = TaskDataset(tokenizer, data_lines)

    model.train() # set the model to train mode

    model, lora_config = create_peft_config(model)

    output_dir = "/root/autodl-tmp/output"

    # Define training config
    config = {
        'lora_config': lora_config,
        'learning_rate': 1e-4,
        'num_train_epochs': 2, # train the model for 2 epoch
        'per_device_train_batch_size': 1, # batch size of 1, you can increase this number to 2 or 4 to if you have enough GPU memory
        'gradient_accumulation_steps': 4, # accumulate gradients for 4 steps, so the effective batch size is 4 * 1 = 4. You may try to increase this number to 8 or 16 to improve the performance
        'gradient_checkpointing': True,
    }


    # Define training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        bf16=True,  # Use BF16 if available
        # logging strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=5,
        save_strategy="no",
        optim="adamw_torch",
        **{k: v for k, v in config.items() if k != 'lora_config'}
    )

    # Train the model
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )

    # Start training
    logger.info("*** starting training ***")
    trainer.train()

    # Save the model to the output directory
    save_dir = "/root/autodl-tmp/output/peft_model"
    model.save_pretrained(save_dir)
