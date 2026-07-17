import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from transformers import AutoTokenizer, AutoProcessor,AutoConfig,AutoModelForCausalLM
from helper.modeling_draft import Model
import torch
import datasets
import os

from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import time
import importlib.util

from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
from tqdm import tqdm
from torch.nn.attention import SDPBackend, sdpa_kernel
import datasets
from transformers import get_cosine_schedule_with_warmup,get_scheduler
from transformers import DynamicCache
import json
import pandas as pd
import re
import signal
import torch
import argparse


def handle_signal(signum, frame):
    print("Received signal, cleaning up...")
    if torch.cuda.is_available():
        del model
        torch.cuda.empty_cache()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


def _dtype_from_name(name):
    name = str(name or "auto").lower()
    if name == "auto":
        return "auto"
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype={name}")


def _resolve_attn_implementation(requested):
    requested = str(requested or "")
    if not requested:
        return None
    if requested == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        print(
            "Warning: attn_implementation=flash_attention_2 was requested, "
            "but flash_attn is not installed. Falling back to eager."
        )
        return "eager"
    return requested


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}



parser = argparse.ArgumentParser(description="Training configuration") 
parser.add_argument('--model_dir',type=str,) 
parser.add_argument('--version_name', type=str,help='Version name for saving checkpoints')
parser.add_argument('--model_type',type=str,default='qwen2')
parser.add_argument('--dtype', type=str, default='auto', choices=['auto', 'bf16', 'fp16', 'fp32'])
parser.add_argument('--attn_implementation', type=str, default='')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--accumulation_steps', type=int, default=16)
parser.add_argument('--warmup_ratio', type=float, default=0.05)
parser.add_argument('--sample_num', type=int, default=200)
parser.add_argument('--max_seq_len', type=int, default=4096)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--persistent_workers', default=True)
parser.add_argument('--log_dir',type=str,required=True)
parser.add_argument('--saved_model_dir',type=str,required=True)
parser.add_argument('--dataset_dir',type=str,required=True)

args = parser.parse_args()
model_dir=args.model_dir
version_name=args.version_name
batch_size = args.batch_size
num_epochs = args.num_epochs
lr = args.lr
accumulation_steps = args.accumulation_steps
warmup_ratio = args.warmup_ratio
sample_num = args.sample_num
max_seq_len = args.max_seq_len
num_workers = args.num_workers
persistent_workers = _as_bool(args.persistent_workers)
log_dir=args.log_dir
saved_model_dir=args.saved_model_dir
dataset_dir = args.dataset_dir
model_torch_dtype = _dtype_from_name(args.dtype)
attn_impl = _resolve_attn_implementation(args.attn_implementation)

if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

print(version_name,os.getenv('CUDA_VISIBLE_DEVICES'))

with open(dataset_dir,'r',encoding='utf-8') as f:
    sharegpt_dataset=json.load(f)
df=pd.DataFrame(sharegpt_dataset)
dataset=datasets.Dataset.from_pandas(df)
print(dataset)

config=AutoConfig.from_pretrained(model_dir)
if model_torch_dtype != "auto":
    config.torch_dtype = model_torch_dtype
model_type=args.model_type
target_model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype=model_torch_dtype, config=config, attn_implementation=attn_impl)
target_model.eval()

config.rope_scaling=None
config.num_hidden_layers=1
if model_torch_dtype != "auto":
    config.torch_dtype = model_torch_dtype
model=Model(config, target_model=target_model).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side = "right")

count=0
for param in model.parameters():
    if param.requires_grad==True:
        print(param.shape)
        count+=param.numel()
        
print(count/1000/1000,'M')


class DataCollator:
    def __init__(self, tokenizer, max_length=4096):
        self.tokenizer=tokenizer
        self.max_length=max_length
        
    def __call__(self, batch):
        batch_input_ids=[]
        batch_attention_mask=[]
        batch_loss_mask=[]
        max_length=0
        
        for example in batch:
            
            input_ids=[]
            attention_mask=[]
            loss_mask=[]
            
            if model_type == 'qwen2':
                text='<|im_start|>'+'system'+'\n'+'You are a helpful assistant.'+'<|im_end|>'+'\n'
            elif model_type == 'llama':
                text= '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'+'You are a helpful assistant.'+'<|eot_id|>'
            elif model_type == 'deepseek':
                text = "<｜begin▁of▁sentence｜>You are a helpful assistant."
            the_input_ids=self.tokenizer.encode(text,add_special_tokens=False)
            input_ids+=the_input_ids
            attention_mask+=[1]*len(the_input_ids)
            loss_mask+=[0]*len(the_input_ids)

            for idx, conversation in enumerate(example['conversations']):
                role=conversation['from']
                content=conversation['value']
                if role == 'human':
                    role = 'user'
                if role == 'gpt':
                    role = 'assistant'
                
                if model_type == 'qwen2':
                    text='<|im_start|>'+role+'\n'+content+'<|im_end|>'+'\n'
                elif model_type == 'llama':
                    text='<|start_header_id|>'+role+'<|end_header_id|>\n\n'+content+'<|eot_id|>'
                elif model_type == 'deepseek':
                    if role == 'user':
                        text = "<｜User｜>" + content
                    else:
                        text = "<｜Assistant｜>" + content + "<｜end▁of▁sentence｜>"
                the_input_ids=self.tokenizer.encode(text,add_special_tokens=False)
                input_ids+=the_input_ids
                attention_mask+=[1]*len(the_input_ids)

                if role == 'assistant' or role == 'ASSISTANT':
                    loss_mask+=[1]*len(the_input_ids)
                else:
                    loss_mask+=[0]*len(the_input_ids)


            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_loss_mask.append(loss_mask)
            max_length=max(max_length,len(input_ids))

        max_length=min(max_length,self.max_length)
        for idx in range(len(batch)):
            if len(batch_input_ids[idx])>=max_length:
                batch_input_ids[idx]=batch_input_ids[idx][:max_length]
                batch_attention_mask[idx]=batch_attention_mask[idx][:max_length]
                batch_loss_mask[idx]=batch_loss_mask[idx][:max_length]
            
            else:
                the_length=len(batch_input_ids[idx])
                batch_input_ids[idx]=batch_input_ids[idx]+[self.tokenizer.eos_token_id]*(max_length-the_length)
                batch_attention_mask[idx]=batch_attention_mask[idx]+[0]*(max_length-the_length)
                batch_loss_mask[idx]=batch_loss_mask[idx]+[0]*(max_length-the_length)
        
        return {
            'input_ids':torch.tensor(batch_input_ids),
            'attention_mask':torch.tensor(batch_attention_mask),
            'loss_mask':torch.tensor(batch_loss_mask)
        }


datacollator=DataCollator(tokenizer, max_length=max_seq_len)
dataloader=DataLoader(
    dataset,
    collate_fn=datacollator,
    num_workers=num_workers,
    persistent_workers=persistent_workers and num_workers > 0,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
)


def compute_acc(target_logits,draft_logits,valid_positions,k=2):

    target_indices = torch.argmax(target_logits, dim=-1)
    draft_topk_values, draft_topk_indices = torch.topk(draft_logits, k=k, dim=-1)

    top1_hit = draft_topk_indices[..., 0] == target_indices             
    topk_hit = (draft_topk_indices == target_indices.unsqueeze(-1)).any(dim=-1) 

    correct_top1 = (top1_hit & valid_positions).sum().item()
    correct_topk = (topk_hit & valid_positions).sum().item()
    total_valid_tokens = valid_positions.sum().item()
    
    return correct_top1,correct_topk,total_valid_tokens

def compute_normalized_gradient_l2_norm(model):
    gradient_l2_norm = torch.norm(
        torch.cat([param.grad.view(-1) for param in model.parameters() if param.grad is not None])
    )
    num_grad_params = sum(
        param.grad.numel() for param in model.parameters() if param.grad is not None
    )
    normalized_gradient_l2_norm = gradient_l2_norm / num_grad_params
    
    return normalized_gradient_l2_norm

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
l1_loss=nn.SmoothL1Loss(reduction='none')

num_training_steps = num_epochs * ((len(dataloader)+accumulation_steps-1)//accumulation_steps)
num_warmup_steps = min(int(warmup_ratio * num_training_steps), 500)
print(num_training_steps)
lr_scheduler = get_scheduler(
    name="cosine_with_min_lr",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
    scheduler_specific_kwargs={'min_lr_rate':0.0}, 
)

total_correct_top1=[]
total_correct_topk=[]
total_token_nums=[]

step=0
accumulated_step=0
batch_logs=[]
start_time=time.time()

for epoch in range(num_epochs):

    log_file = log_dir + f"/epoch_{epoch}.log"
    with open(log_file,'w',encoding='utf-8') as f:
        pass

    for i,batch in enumerate(dataloader):

        input_ids=batch['input_ids'].to('cuda')
        attention_mask=batch['attention_mask'].to('cuda')
        loss_mask=batch['loss_mask'].to('cuda')
        
        if not torch.any(loss_mask==1):
            continue
        
        with torch.no_grad():
            target_outputs=model.target_model.model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            output_hidden_states=False)

            last_hidden_state=target_outputs.last_hidden_state
            feature_states=last_hidden_state
            target_logits=model.target_model.lm_head(last_hidden_state)

        
        target_logits=target_logits[:,:-1,:]
        feature_states=feature_states[:,:-1,:].to(model.dtype)

        input_ids=input_ids[:,1:]
        attention_mask=attention_mask[:,:-1]
        loss_mask=loss_mask[:,:-1]

        draft_outputs=model(hidden_states=feature_states,input_ids=input_ids,attention_mask=attention_mask,use_cache=False)
        next_feature_states=draft_outputs['next_feature_states']
        draft_hidden_states=draft_outputs['hidden_states'].to(model.target_model.dtype)
        draft_logits=model.lm_head(draft_hidden_states)

        loss1=l1_loss(next_feature_states[:,:-1,:].float(),feature_states[:,1:,:].float())

        loss1=torch.mean(loss1,dim=-1)*loss_mask[:,:-1] 
        loss1=torch.sum(loss1, dim=-1) / torch.sum(loss_mask[:,:-1], dim=-1)
        loss1=loss1.mean()
        loss1=loss1*2

        with torch.no_grad():
            target_logits=target_logits[:,1:,:].float().softmax(dim=-1).detach()
        draft_logits=draft_logits[:,:-1,:].float().softmax(dim=-1)

        plogp=target_logits*torch.log(draft_logits)
        loss2=torch.sum(plogp,dim=-1)*loss_mask[:,:-1]
        loss2=torch.sum(loss2, dim=-1) / torch.sum(loss_mask[:,:-1], dim=-1)
        loss2= - loss2.mean()

        loss2=loss2*0.1
        
        loss=loss1+loss2
        
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            if feature_states.grad is not None:
                feature_states.grad.zero_()
            
            loss = loss.detach()
            del loss
            del feature_states,next_feature_states,target_logits,draft_logits
            torch.cuda.empty_cache()
        
        else:
            accumulated_step+=1
            
            if accumulated_step%accumulation_steps==1: 
                
                optimizer.zero_grad(set_to_none=True)
                loss2.backward(retain_graph=True)
                loss2_norm=compute_normalized_gradient_l2_norm(model.draft_model.layers[0])
                optimizer.zero_grad(set_to_none=True)
                
                loss1.backward(retain_graph=True)
                loss1_norm=compute_normalized_gradient_l2_norm(model.draft_model.layers[0])
                optimizer.zero_grad(set_to_none=True)
            
            
            loss/=accumulation_steps
            loss.backward()

            valid_positions=loss_mask[:,:-1]
            with torch.no_grad():
                correct_top1,correct_topk,total_valid_tokens=compute_acc(target_logits,draft_logits,valid_positions,k=2)
            
            total_correct_top1.append(correct_top1)
            total_correct_topk.append(correct_topk)
            total_token_nums.append(total_valid_tokens)

            batch_logs.append({
                'loss':loss.item()*accumulation_steps,
                'loss1':loss1.item(),
                'loss2':loss2.item(),
                'loss1_norm':loss1_norm.item(),
                'loss2_norm':loss2_norm.item(),
                'correct_top1':correct_top1,
                'correct_topk':correct_topk,
                'total_valid_tokens':total_valid_tokens
            })
        
            if accumulated_step%accumulation_steps==0:
                step+=1
                real_sample_num=sample_num*accumulation_steps

                avg_logs = {
                    "step": step,
                    "loss": round(sum(log["loss"] for log in batch_logs)/len(batch_logs),4),
                    "used_time": round((time.time()-start_time)/60, 3),
                    "loss1": round(sum(log["loss1"] for log in batch_logs)/len(batch_logs),4),
                    "loss2": round(sum(log["loss2"] for log in batch_logs)/len(batch_logs),4),
                    "loss1_norm": sum(log["loss1_norm"] for log in batch_logs),
                    "loss2_norm": sum(log["loss2_norm"] for log in batch_logs),
                    "top1_acc": round(sum(log['correct_top1'] for log in batch_logs)/sum(log['total_valid_tokens'] for log in batch_logs),4),
                    "topk_acc": round(sum(log['correct_topk'] for log in batch_logs)/sum(log['total_valid_tokens'] for log in batch_logs),4),
                    f"last{sample_num}_top1_acc": round(sum(total_correct_top1[-real_sample_num:])/sum(total_token_nums[-real_sample_num:]),4),
                    f"last{sample_num}_topk_acc": round(sum(total_correct_topk[-real_sample_num:])/sum(total_token_nums[-real_sample_num:]),4),
                }
                
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(avg_logs) + '\n')

                total_correct_top1=total_correct_top1[-real_sample_num:]
                total_correct_topk=total_correct_topk[-real_sample_num:]
                total_token_nums=total_token_nums[-real_sample_num:]
                    
                batch_logs.clear()
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                if step%8000==0 and step!=0:
                    model.save_model(f'{saved_model_dir}/step{step}.pth')
                
                if (step*accumulation_steps)%16==0:
                    torch.cuda.empty_cache()
    

model.save_model(f'{saved_model_dir}/step{step}.pth')  
