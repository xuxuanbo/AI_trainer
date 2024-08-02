from datasets import load_dataset,load_from_disk
import json
import random

def load_mrpc(path='autodl-tmp/mrpc/msr_paraphrase_test.txt',save_path='autodl-tmp/mrpc_processed/msr_paraphrase_test.jsonl'):
    with open(path,'r', encoding='utf-8') as f, open(save_path,'w', encoding='utf-8') as o:
        next(f)
        for l in f:
            ele_list = l.strip().split('\t')
            label = ele_list[0]
            sen1 = ele_list[3]
            sen2 = ele_list[4]
            template = '''You are given two sentences below, Sentence 1 and Sentence 2. If the two senetences are semantically equivalent, please return 1. Otherwise, please return 0.
            ### Sentence 1: {}
            ### Sentence 2: {}
            ### Return:'''.format(sen1.strip(),sen2.strip()).lstrip()

            new_entry = {
                    'instruct':template,
                    'label':label
            }

            json_line = json.dumps(new_entry, ensure_ascii=False)
            o.write(json_line+'\n')
    print('数据已写入',save_path)


def load_mrpc_jsonl(path):
    with open(path,'r',encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    print(len(data))
    return data

def split_dataset(data,ratio = 0.7):
    random.seed(2024)
    random.shuffle(data)
    train_size = int(ratio*len(data))
    train = data[:train_size]
    test = data[train_size:]
    return train,test

if __name__ == "__main__":
    # load_mrpc()
    # load_mrpc(path='autodl-tmp/mrpc/msr_paraphrase_train.txt',save_path='autodl-tmp/mrpc_processed/msr_paraphrase_train.jsonl')
    train = load_mrpc_jsonl('autodl-tmp/mrpc_processed/msr_paraphrase_train.jsonl')
    test = load_mrpc_jsonl('autodl-tmp/mrpc_processed/msr_paraphrase_test.jsonl')
    full = train + test
    train,test = split_dataset(full)
    print(len(full),len(train),len(test))