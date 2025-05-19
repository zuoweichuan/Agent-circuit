
import torch
import torch.nn.functional as F
import numpy as np

import rl_utils
import random
from model import GPT, GPTConfig  # 导入GPT模型和配置


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim, dropout_prob=0.5):
        super(PolicyNet, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.LayerNorm(hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout_prob))
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, action_dim))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return F.softmax(self.model(x), dim=1)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims, dropout_prob=0.5):
        super(ValueNet, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.LayerNorm(hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout_prob))
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, model_args,state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, gpt_model_path,dropout_rate=0.5):
        if gpt_model_path:
            checkpoint = torch.load(gpt_model_path, map_location=device)
            checkpoint_model_args = checkpoint['model_args']
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                model_args[k] = checkpoint_model_args[k]
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.transformer_config = GPTConfig(**model_args)
        self.transformer = GPT(self.transformer_config).to(device)
        if gpt_model_path:
            self.transformer.load_state_dict(state_dict)  # 加载预训练的GPT模型参数
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim,dropout_rate).to(device)
        self.critic = ValueNet(state_dim, hidden_dim,dropout_rate).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.transformer_optimizer = torch.optim.Adam(self.transformer.parameters(),
                                                      lr=actor_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state, mask_, epsilon=0.1):
            state = torch.tensor([state], dtype=torch.long).to(self.device)
            state, _, _ = self.transformer(state)  # 使用Transformer提取特征
            state = state.squeeze(0)
            probs = self.actor(state)
            for i in mask_:
                probs[0][i] = 0
            
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            if random.random() < epsilon:
                # 随机选择一个动作
                action = random.choice([i for i in range(probs.size(1)) if i not in mask_])
            else:
                # 选择概率最大的动作
                action = torch.argmax(probs, dim=-1).item()
            
            return action

    def process_in_batches(self, states, next_states, rewards, dones, actions, batch_size=32):
        td_target_f = []
        advantage_f = []
        # td_delta_f = []
        old_log_probs_f = []

        for i in range(0, states.size(0)):
            batch_states = states[i:i + batch_size] if i + batch_size < states.size(0) else states[i:]
            batch_next_states = next_states[i:i + batch_size] if i + batch_size < next_states.size(0) else next_states[i:]
            batch_rewards = rewards[i:i + batch_size] if i + batch_size < rewards.size(0) else rewards[i:]
            batch_dones = dones[i:i + batch_size] if i + batch_size < dones.size(0) else dones[i:]
            batch_actions = actions[i:i + batch_size] if i + batch_size < actions.size(0) else actions[i:]

            with torch.no_grad():
                batch_states_f, _, _ = self.transformer(batch_states)
                batch_states_f = F.softmax(batch_states_f, dim=-1)
                batch_states_f = batch_states_f.squeeze(1)
                batch_next_states_f, _, _ = self.transformer(batch_next_states)
                batch_next_states_f = F.softmax(batch_next_states_f, dim=-1)
                batch_next_states_f = batch_next_states_f.squeeze(1)

                batch_td_target = batch_rewards + self.gamma * self.critic(batch_next_states_f) * (1 - batch_dones)
                batch_td_delta = batch_td_target - self.critic(batch_states_f)

                batch_advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, batch_td_delta.cpu()).to(self.device)
                batch_old_log_probs = torch.log(self.actor(batch_states_f).gather(1, batch_actions) + 1e-5).detach()

            td_target_f.append(batch_td_target)
            advantage_f.append(batch_advantage)
            # td_delta_f.append(batch_td_delta)
            old_log_probs_f.append(batch_old_log_probs)

            # 显式地释放不再使用的显存
            torch.cuda.empty_cache()

        td_target = torch.cat(td_target_f, dim=0)
        # td_delta = torch.cat(td_delta_f, dim=0)
        advantage = torch.cat(advantage_f, dim=0)
        # advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.cat(old_log_probs_f, dim=0)

        return td_target, advantage, old_log_probs

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.long).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.long).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.long).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.long).view(-1, 1).to(self.device)
        
        # 使用Transformer提取特征
        td_target, advantage, old_log_probs = self.process_in_batches(states, next_states,rewards,dones,actions, batch_size=32)
        
        # td_target = rewards + self.gamma * self.critic(next_states_f) * (1 - dones)
        # td_delta = td_target - self.critic(states_f)
        # advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        # old_log_probs = torch.log(self.actor(states_f).gather(1, actions)).detach()

        batch_size = 32  # 设置批次大小
        num_samples = states.size(0)
        indices = list(range(num_samples))

        for eco in range(self.epochs):
            random.shuffle(indices)
            print(f'epoch {eco + 1} begin !')
            for start in range(0, num_samples - batch_size + 1):
                # print(start)
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantage = advantage[batch_indices]
                if torch.isnan(batch_advantage).any() or torch.isinf(batch_advantage).any():
                    print(start)
                    raise ValueError("Batch advantage contains NaN or Inf values")
                batch_old_log_probs = old_log_probs[batch_indices]
                if torch.isnan(batch_old_log_probs).any() or torch.isinf(batch_old_log_probs).any():
                    print(start)
                    raise ValueError("Batch old log probs contains NaN or Inf values")
                batch_td_target = td_target[batch_indices]
                if torch.isnan(batch_td_target).any() or torch.isinf(batch_td_target).any():
                    print(start)
                    raise ValueError("Batch td_target contains NaN or Inf values")

                if torch.isnan(batch_states).any() or torch.isinf(batch_states).any():
                    print(start)
                    raise ValueError("Batch states contains NaN or Inf values")
                    
                
                
                # 使用Transformer提取特征
                batch_states_f, _, _ = self.transformer(batch_states)
                batch_states_f = F.softmax(batch_states_f, dim=-1)
                batch_states_f = batch_states_f.squeeze(1)

                if torch.isnan(batch_states_f).any() or torch.isinf(batch_states_f).any():
                    print(start)
                    raise ValueError("Batch states f contains NaN or Inf values")
                
                batch_log_probs = torch.log(self.actor(batch_states_f).gather(1, batch_actions) + 1e-5)

                if torch.isnan(batch_log_probs).any() or torch.isinf(batch_log_probs).any():
                    print(start)
                    raise ValueError("Batch log probs contains NaN or Inf values")

                ratio = torch.exp(batch_log_probs - batch_old_log_probs)
                if torch.isnan(ratio).any() or torch.isinf(ratio).any():
                    print(start)
                    print(batch_old_log_probs)
                    print(batch_log_probs)
                    raise ValueError("Ratio contains NaN or Inf values")
                surr1 = ratio * batch_advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * batch_advantage  # 截断

                # batch_critic_values = self.critic(batch_states_f)
                # critic_values.append(batch_critic_values)

            # critic_values = torch.cat(critic_values, dim=0)

                actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
                critic_loss = torch.mean(F.mse_loss(self.critic(batch_states_f), batch_td_target.detach()))
                total_loss = actor_loss + critic_loss   
                
                if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                    print(start)
                    raise ValueError("Tatal loss contains NaN or Inf values")                
            
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                self.transformer_optimizer.zero_grad()
            
                total_loss.backward()

                # 添加梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)

                self.actor_optimizer.step()
                self.critic_optimizer.step()
                self.transformer_optimizer.step()

                torch.cuda.empty_cache()
        
            self.save('PPO_out/ckpt.pt')
        torch.cuda.empty_cache()
    
    def save(self, path):
        torch.save({
            'transformer': self.transformer.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'transformer_optimizer': self.transformer_optimizer.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.transformer.load_state_dict(checkpoint['transformer'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.transformer_optimizer.load_state_dict(checkpoint['transformer_optimizer'])

# actor_lr = 1e-3
# critic_lr = 1e-2
# num_episodes = 500
# hidden_dim = 128
# gamma = 0.98
# lmbda = 0.95
# epochs = 10
# eps = 0.2
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# env_name = 'CartPole-v0'
# env = gym.make(env_name)
# env.seed(0)
# torch.manual_seed(0)
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
# gpt_model_path = 'path_to_pretrained_gpt_model.pth'  # 预训练GPT模型的路径
# agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
#             epochs, eps, gamma, device, gpt_model_path)

# return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

# episodes_list = list(range(len(return_list)))
# plt.plot(episodes_list, return_list)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('PPO on {}'.format(env_name))
# plt.show()

# mv_return = rl_utils.moving_average(return_list, 9)
# plt.plot(episodes_list, mv_return)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('PPO on {}'.format(env_name))
# plt.show()