import datasets
import pandas as pd
import math
import json
import os
import random
dir_key = "uids1103/data"

prompt_dict = {
    "math" : {
        "system_prompt" : "You are a math problem assistant." , 
        "user_prompt" : '''Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            Your response should include your thought process enclosed within <think></think> tags
            and the final answer enclosed within <answer></answer> tags (Just put a number between the tags).\n
            ### Instruction:\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}'''
    }
}
class DataCollator:
    def __init__(self,tokenizer,system_prompt,user_prompt):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.tokenizer = tokenizer
    def __call__(self, examples):
        text = [
            self.tokenizer.apply_chat_template([
                {"role" : "system" , "content": self.system_prompt} , 
                {"role" : "user" , "content": self.user_prompt.format_map({"instruction" : example['question']})}   
            ] , tokenize = False , add_generation_prompt = True ) for example in examples
        ]

        inputs = self.tokenizer(text,return_tensors = "pt",padding = True,add_special_tokens=False)

        answers = [
            example['answer'] for example in examples
        ]

        return {
            "input_ids" : inputs["input_ids"] ,
            "attention_mask" : inputs["attention_mask"] , 
            "answer" : answers
        }

info = {"amc-23" : "math" , "gsm8k" : "math" , "math-500" : "math" , "simplelr_qwen_level3to5" : "math" ,
        "simplelr_abel_level3to5" : "math" , "simplelr_abel_level3to5_smoke" : "math" , "DAPO-math" : "math"}


def _load_simplelr_parquet(path):
    ds = pd.read_parquet(path)
    ds = pd.DataFrame({
        'question': ds['prompt'].apply(lambda x: x[0]['content']),
        'answer': ds['reward_model'].apply(lambda y: f"\\boxed{{{y['ground_truth']}}}")
    })
    return ds.to_dict(orient='records')


def select_train_subset(examples, fraction=1.0, max_samples=0, seed=42):
    total = len(examples)
    if total == 0:
        return examples
    fraction = float(1.0 if fraction is None else fraction)
    if fraction <= 0:
        raise ValueError(f"train_data_fraction must be > 0, got {fraction}")
    target = total if fraction >= 1.0 else max(1, int(math.ceil(total * fraction)))
    max_samples = int(max_samples or 0)
    if max_samples > 0:
        target = min(target, max_samples)
    if target >= total:
        return examples
    rng = random.Random(int(seed))
    indices = list(range(total))
    rng.shuffle(indices)
    indices = sorted(indices[:target])
    return [examples[idx] for idx in indices]

def get_test_QAs(option,tokenizer=None):
    
    if option == "amc-23":
        eval_data_path = f"./data/amc-23"
        dataset = datasets.load_dataset(eval_data_path,split='test') 
        QAs = [{'question':x, 'answer':f"\\boxed{{{y.split('####')[-1].strip()}}}"} for x,y in zip(dataset['question'], dataset['answer'])]
    elif option == "gsm8k":
        data_path = f"./data/gsm8k/main"
        dataset = datasets.load_dataset(data_path,split='test') 
        QAs = [{'question':x, 'answer':f"\\boxed{{{y.split('####')[-1].strip()}}}"} for x,y in zip(dataset['question'], dataset['answer'])]
    elif option == "math-500":
        eval_data_path = f"./data/math-500/"
        dataset = datasets.load_dataset(eval_data_path,split='test') 
        QAs = [{'question':x, 'answer':f"\\boxed{{{y}}}"} for x,y in zip(dataset['problem'], dataset['answer'])]
    elif option == "simplelr_qwen_level3to5":
        eval_data_path = f"./data/simplelr_qwen_level3to5/test.parquet"
        ds = pd.read_parquet(eval_data_path)
        ds = pd.DataFrame({
        'question': ds['prompt'].apply(lambda x: x[0]['content']),
        'answer': ds['reward_model'].apply(lambda y: f"\\boxed{{{y['ground_truth']}}}")
        })
        QAs = ds.to_dict(orient='records')
    elif option == "simplelr_abel_level3to5":
        data_path = f"./data/simplelr_abel_level3to5/test.parquet"
        QAs = _load_simplelr_parquet(data_path)
    elif option == "simplelr_abel_level3to5_smoke":
        data_path = f"./data/simplelr_abel_level3to5_smoke/test.parquet"
        QAs = _load_simplelr_parquet(data_path)
    elif option == 'DAPO-math':
        dataset = datasets.load_dataset(r"./data/DAPO-Math-17k-Processed",split='train')
        QAs = [{'question' : item['prompt'] , 'answer' : f"\\boxed{{{item['reward_model']['ground_truth']}}}" }for item in dataset]
    if tokenizer is None:
        return QAs
    return QAs , DataCollator(tokenizer,prompt_dict[info[option]]['system_prompt'],prompt_dict[info[option]]['system_prompt'])


def get_train_QAs(option,tokenizer=None):
    if option == "gsm8k":
        data_path = f"./data/gsm8k/main"
        dataset = datasets.load_dataset(data_path,split='train') 
        QAs = [{'question':x, 'answer':f"\\boxed{{{y.split('####')[-1].strip()}}}"} for x,y in zip(dataset['question'], dataset['answer'])]
    elif option == "simplelr_qwen_level3to5":
        data_path = f"./data/simplelr_qwen_level3to5/train.parquet"
        ds = pd.read_parquet(data_path)
        ds = pd.DataFrame({
        'question': ds['prompt'].apply(lambda x: x[0]['content']),
        'answer': ds['reward_model'].apply(lambda y: f"\\boxed{{{y['ground_truth']}}}")
        })
        QAs = ds.to_dict(orient='records')
    elif option == "simplelr_abel_level3to5":
        data_path = f"./data/simplelr_abel_level3to5/train.parquet"
        QAs = _load_simplelr_parquet(data_path)
    elif option == "simplelr_abel_level3to5_smoke":
        data_path = f"./data/simplelr_abel_level3to5_smoke/train.parquet"
        QAs = _load_simplelr_parquet(data_path)
    elif option == 'DAPO-math':
        dataset = datasets.load_dataset(f"./data/DAPO-Math-17k-Processed",split='train')
        QAs = [{'question' : item['prompt'] , 'answer' : f"\\boxed{{{item['reward_model']['ground_truth']}}}" }for item in dataset]
    if tokenizer is None:
        return QAs
    return QAs , DataCollator(tokenizer,prompt_dict[info[option]]['system_prompt'],prompt_dict[info[option]]['system_prompt'])
