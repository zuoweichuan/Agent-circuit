import random
import torch
import numpy as np


def get_batch(split, data_trian, data_val, idx_b_train, idx_b_val, block_size, batch_size, device):
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
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, ctx, eval_iters):
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
        out[split] = losses.mean()
    model.train()
    return out

import re

def EVO_To_C(path, i):
    with open(path,'r') as f:
        Top_module = f.read()
    while 'S_' in Top_module:
        x = re.search(r'S_\d+_\d{2}', Top_module)
        if x is None:
            x = re.search(r'S_\d+_\d', Top_module)
        Top_module = Top_module.replace(x.group(), 'N[' + str(i) + ']')
        i += 1
    while 'C_' in Top_module:
        x = re.search(r'C_\d+_\d{2}', Top_module)
        if x is None:
            x = re.search(r'C_\d+_\d+', Top_module)
        Top_module = Top_module.replace(x.group(), 'N[' + str(i) + ']')
        i += 1
    with open(path,'w') as f:
        f.write(Top_module)
    print(i)

def ABC_To_C(path):
    with open (path,'r') as f:
        Top_module = f.read()
    Top_module = re.findall(r'  assign.*;', Top_module)
    Top_module = '\n'.join(Top_module)
    Top_module = re.sub(r'\\','',Top_module)
    Top_module = re.sub(r'\~','!',Top_module)
    Top_module = re.sub(r'new_n(\d+)', r'N[\1]', Top_module)

    with open(path,'w') as f:
        f.write(Top_module)

def C_To_ABC(path):
    with open (path,'r') as f:
        Top_module = f.read()

    Top_module = re.sub(r'A',r'\\A',Top_module)
    Top_module = re.sub(r'B',r'\\B',Top_module)
    Top_module = re.sub(r'O',r'\\O',Top_module)
    Top_module = re.sub(r'Z',r'\\Z',Top_module)
    Top_module = re.sub(r'!',r'~',Top_module)
    Top_module = re.sub(r'N\[(\d+)\]', r'new_n\1', Top_module)

    with open(path,'w') as f:
        f.write(Top_module)

def Save(path, save_path):
    with open (path,'r') as f:
        content = f.read()
    with open(save_path,'w') as f:
        f.write("module mul8s (clock,A,B,O);\n  input clock;\n  input [7:0] A;\n  input [7:0] B;\n  output [15:0] O;\n  wire [600:0] N;\n")
        f.write(content)
        f.write("\nendmodule")

if __name__ == "__main__":
    Path = 'temp.v'
    # Save_path = 'Result/11x11/mul11u_mse2.3e7_ar427.56.v'
    ABC_To_C(Path)
    #C_To_ABC(Path)
    # Save(Path,Save_path)
    #EVO_To_C(Path,972)