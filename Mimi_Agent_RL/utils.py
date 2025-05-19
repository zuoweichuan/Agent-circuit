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