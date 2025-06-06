import torch

# GPT 训练参数
max_iters = 320
log_interval = 10
eval_interval = 320
eval_iters = 320
decay_lr = True  # 是否衰减学习率
warmup_iters = 9  # 预热步骤数
lr_decay_iters = 28  # 应该约等于 max_iters (根据 Chinchilla 论文)
min_lr = 1e-5  # 最小学习率，应约为 learning_rate/10
grad_clip = 1.0
learning_rate = 5e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
gradient_accumulation_steps = 1
batch_size = 16
block_size = 512
n_layer = 8
n_head = 8
n_embd = 128
dropout = 0.2  # 预训练时使用 0，微调时尝试 0.1+
eval_only = False  # 是否只运行评估
bias = False  # 是否在 LayerNorm 和 Linear 层中使用偏置
always_save_checkpoint = False  # 是否在每次评估时保存检查点
compile = False  # 是否使用 torch.jit.script 编译模型
init_from = 'scratch'  # 'scratch', 'resume', 或 'gpt2-medium'
meta_path = r'data\meta.pkl'
GPT_out_dir = r'GPT_out'


Train_model = 'PPO'  # 'GPT' 或 'PPO'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# PPO 参数
gpt_model_path = r'GPT_out\ckpt.pt'
load_checkpoint = False
max_steps = 2048
line_range = 200
mse_max = 20000
actor_lr = 5e-5
critic_lr = 1e-4
num_episodes = 1000
hidden_dim = [2, 4, 8, 16]
dropout_rate = 0.2
gamma = 0.90
lmbda = 0.95
epochs = 2
eps = 0.05
action_dim = 8

# 内部状态
iter_num = 0
best_val_loss = 1000000
seed_offset = 0
local_iter_num = 0