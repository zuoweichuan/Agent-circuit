import random

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from PPO_ import PPO
import rl_utils
import env

import torch._dynamo
torch._dynamo.config.suppress_errors = True
#-------------------------------------------
max_iters = 320000
log_interval = 160
eval_interval = 3200
eval_iters = 3200
decay_lr = True # whether to decay the learning rate
warmup_iters = 9600 # how many steps to warm up for
lr_decay_iters = 280000 # should be ~= max_iters per Chinchilla
min_lr = 1e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
grad_clip = 1.0
learning_rate = 5e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
gradient_accumulation_steps = 1
batch_size = 32
block_size = 512
n_layer = 16
n_head = 32
n_embd = 512
dropout = 0.2 # for pretraining 0 is good, for finetuning try 0.1+
eval_only = False # whether to only run eval and then stop
bias = False # do we use bias inside LayerNorm and Linear layers?
always_save_checkpoint = False # whether to save a checkpoint on every eval
compile = False # whether to compile the model using torch.jit.script
init_from = 'scratch' # 'scratch', 'resume', or 'gpt2-medium'
meta_path = r'/home/aic711/nanoLAMG/GPT-PPO/data/meta.pkl'
GPT_out_dir = r'/home/aic711/nanoLAMG/GPT-PPO/GPT_out'
Train_model = 'PPO'    # 'GPT' or 'PPO'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
#-------------------------------------------
gpt_model_path = r'/home/aic711/nanoLAMG/GPT-PPO/GPT_out/ckpt.pt'
load_checkpoint = True
max_steps = 2048
line_range = 200
mse_max = 8743
actor_lr = 1e-4
critic_lr = 1e-4
num_episodes = 100
hidden_dim = [64,128,256,512]
dropout_rate = 0.2
gamma = 0.90
lmbda = 0.95
epochs = 2
eps = 0.2
action_dim = 8
#-------------------------------------------
iter_num = 0
best_val_loss = 1000000
seed_offset = 0
local_iter_num = 0 
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
#-------------------------------------------
# 计算每轮的token数量
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if Train_model == 'GPT':
    os.makedirs(GPT_out_dir, exist_ok=True)
else:
    os.makedirs('PPO_out', exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 获取词典
load_meta = os.path.exists(meta_path)
meta_vocab_size = None
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

#加载数据
data_trian = np.memmap(os.path.join(f'data', f'train.bin'), dtype=np.uint16, mode='r')
data_val = np.memmap(os.path.join('data', f'val.bin'), dtype=np.uint16, mode='r')
idx_b_train = [i for i, x in enumerate(data_trian) if x == stoi['\n<b>\n']]
idx_b_val = [i for i, x in enumerate(data_val) if x == stoi['\n<b>\n']]
def get_batch(split):
    if split == 'train':
        idx_q = random.randint(0, len(idx_b_train) -2) if len(idx_b_train) > 1 else 0
        pre = data_trian[idx_b_train[idx_q]:idx_b_train[idx_q] + 32]
        ix = torch.randint(idx_b_train[idx_q] - 50, idx_b_train[idx_q + 1] - block_size +  32,(batch_size,)) if len(idx_b_train) > 1 else torch.randint(len(data_trian) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((np.concatenate((pre,data_trian[i:i+block_size - 32]))).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((np.concatenate((pre,data_trian[i + 1:i + 1 +block_size - 32]))).astype(np.int64)) for i in ix])
    else:
        idx_q = random.randint(0, len(idx_b_val) -2) if len(idx_b_val) > 1 else 0
        pre = data_val[idx_b_val[idx_q]:idx_b_val[idx_q] + 32]
        ix = torch.randint(idx_b_val[idx_q] -50, idx_b_val[idx_q + 1] - block_size  + 32,(batch_size,)) if len(idx_b_val) > 1 else torch.randint(len(data_trian) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((np.concatenate((pre,data_val[i:i+block_size - 32]))).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((np.concatenate((pre,data_val[i + 1:i+1+block_size -32]))).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) 

if Train_model == 'GPT':
    # 初始化模型
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        print(f"Resuming training from {GPT_out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(GPT_out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
    model.to(device)
else:
    state_dim = meta_vocab_size
    model_args['vocab_size'] = state_dim  # 词典大小
    model = PPO(model_args,state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device, gpt_model_path)

if Train_model == 'GPT':
    # 初始化优化器
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            accrucys = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss,accrucy = model(X, Y)
                losses[k] = loss.item()
                accrucys[k] = accrucy.item()
            out[split] = accrucys.mean()
        model.train()
        return out

    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    X, Y = get_batch('train')
    t0 = time.time()
    raw_model = model
    running_mfu = -1.0

    while(True):
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter_num}: train losses {losses['train']:.4f}, val losses {losses['val']:.4f}")
            if losses['val'] < best_val_loss or always_save_checkpoint and iter_num != 0:
                if iter_num > 0:
                    best_val_loss = losses['val']
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {GPT_out_dir}")
                    torch.save(checkpoint, os.path.join(GPT_out_dir, 'ckpt.pt'))
        if iter_num == 0 and eval_only:
            break
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss,accrucy = model(X, Y)
                loss = loss / gradient_accumulation_steps
            X, Y = get_batch('train')
            scaler.scale(loss).backward()
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0:
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1
        if iter_num > max_iters:
            break
else:
    with open('temp.v', 'r') as f:
        state = f.read()
    Envi = env.Env(state = state, line_range = line_range, mse_max = mse_max,stoi = stoi, block_size=block_size)
    agent = PPO(model_args,state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
             epochs, eps, gamma, device, gpt_model_path,dropout_rate)
    
    if load_checkpoint:
        agent.load('PPO_out/ckpt.pt')
    
    return_list = rl_utils.train_on_policy_agent(Envi, agent, num_episodes, max_steps)
    episodes_list = list(range(len(return_list)))
    mv_return = rl_utils.moving_average(return_list, 9)