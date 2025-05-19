from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import pickle

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def encode(s,stoi):
    encoded = []
    chars = stoi.keys()
    i = 0
    while i < len(s):
        match = None
        for char in chars:
            if s[i:i+len(char)] == char:
                match = char
                break
        if match:
            encoded.append(stoi[match])
            i += len(match)
        else:
            print(f"no match for {s[i:i+10]}")
            exit()
    return encoded

def mask(env):
    mask = [] 
    tg = env.lines[env.now]
    if '&' not in tg and '|' not in tg and '^' not in tg:
        mask = [1,2,3,4,5,6]
        if 'A' in tg or 'B' in tg or 'b' in tg:
            mask.append(7)
    if env.count_not == 2:
        mask.extend([6,7])
    if env.count_replace == 1:
        mask.extend([3,4,5])
    if env.last_action in [3,4,5]:
        mask.extend([3,4,5])
    elif env.last_action == 6:
        mask.append(6)
    elif env.last_action == 7:
        mask.append(7)
    
    if '&' in tg :
        mask.append(3)
    elif '|' in tg :
        mask.append(4)
    elif '^' in tg:
        mask.append(5)
    
    
    mask = list(set(mask))

    return mask
        

def train_on_policy_agent(env, agent, num_episodes, max_steps=1000):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                step = 0
                while not done and step < max_steps:
                    # state_t = agent.transformer.get_state(state)
                    action = agent.take_action(state,mask_=mask(env),epsilon=0.7/(i_episode+1))
                    env.last_action = action
                    next_state, reward, done= env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                    step += 1
                    print(f"  step {step}")
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
            agent.save('PPO_out/ckpt_re.pt')
    return return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()

    advantage_array = np.array(advantage_list)
    return torch.tensor(advantage_array, dtype=torch.float)
