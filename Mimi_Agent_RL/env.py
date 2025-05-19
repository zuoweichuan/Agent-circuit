from To_cpp import V2c, run_cpp_file
import re
import numpy as np
from rl_utils import encode
import torch
import pickle

class Env():
    def __init__(self, state, line_range, mse_max, stoi, block_size):
        self.line_range = line_range  
        self.state = state
        self.next_state = state
        self.initial_state = state
        self.block_size = block_size
        self.stoi = stoi
        self.len = 0
        self.now = 0
        self.area = 0
        self.mse = mse_max
        self.done = False
        self.reward = 0
        self.last_action = 0
        self.last_mse =  0
        self.count_not = 0
        self.count_replace = 0
        

    def count_area(self, state):
        return state.count('&') * 2 + state.count('|') *2 + state.count('!') + state.count('^') * 4

    def Trans(self, state):
        # state = state.replace('>>', '')
        state = state.replace('assign', '')
        state = state.replace(' ', '')
        area, mse = self.Area_Mse()
        self.area = area
        pre = f'\n<b>\n_MSE_{mse}_area_{area}@{self.mse};\n'
        print(f'  area:{area}, mse:{mse}')
        pre = pre + ' '*(32 - len(pre) + 14)
        state = state + '\n'
        state = encode(state,self.stoi)
        if len(state) < self.block_size:
            exitend = [self.stoi[' '] for i in range(self.block_size - len(state))]
            state.extend(exitend)
        pre = encode(pre,self.stoi)
        # state = (torch.tensor(state, dtype=torch.long, device='cuda')[None, ...])
        # pre = (torch.tensor(pre, dtype=torch.long, device='cuda')[None, ...])
        # state = torch.cat((pre, state), dim=1) if state.size(1) <= self.block_size - 32 else torch.cat((pre, state[:, :self.block_size - 32]), dim=1) # 裁剪
        state = pre + state[0: self.block_size - 32]
        return state
    
    def reset(self):
        self.lines = self.initial_state.splitlines()
        state = '\n'.join(self.lines[0:self.line_range])
        self.next_state = self.initial_state
        state = self.Trans(state)
        self.lines[0] = self.lines[0].replace('assign', '>>assign')
        self.now = 0
        self.last_action = 0
        self.done = False
        self.count_not = 0
        self.count_replace = 0
        self.last_mse = 0
        self.len = len(self.lines)
        self.area = self.count_area(self.initial_state)
        return state

    def Area_Mse(self):
        state_mse = '\n'.join(self.lines[0:]) 
        area = self.count_area(state_mse)
        with open('temp.v', 'w') as f:
            f.write(state_mse.replace('>>', ''))
        V2c('temp.v', 'multi.cpp')
        Mse = run_cpp_file('multi.cpp')
        mse = re.findall(r'//MSE\s*=\s*(.*)',Mse).pop()
        if int(mse) > 2 * self.mse:
            self.done = True
        
        if int(mse) > self.mse:
            self.reward = -0.2
        elif area < self.area:
            self.reward = (8.5 - np.log(area))/3.4246 + (3668 - area)/1500
        elif area == self.area:
            self.reward = 0
        else:
            self.reward = -0.09
        
        if int(self.last_mse) > int(mse) :
            self.reward += 0.05
        self.last_mse = mse
        print(f'  reward:{self.reward:.2f}')
        return area, mse

    
    def select_next(self):
        self.lines[self.now] = self.lines[self.now].replace('>>', '')
        self.now += 1
        if self.now == self.len:
            self.done = True
        self.lines[self.now] = self.lines[self.now].replace('assign', '>>assign')

    def step(self, action):
        print('*'*80)
        print(self.lines[self.now],end='  ==>')
        # count not action to prevent take none too many times
        # 什么都不做，直接下一行
        if action == 0:
            self.select_next()
            self.count_not = 0
            self.count_replace = 0
        # 删除门和左边信号：
        elif action == 1:
            self.lines[self.now] = re.sub(r'!?\w\[\d*\]\s*(&|\||\^)\s*', '', self.lines[self.now])
        # 删除门和右边信号
        elif action == 2:
            self.lines[self.now] = re.sub(r'\s*(&|\||\^)\s*!?\w\[\d*\]\s*', '', self.lines[self.now])
        # 替换为与门：
        elif action == 3:
            self.lines[self.now] = re.sub(r'(&|\||\^)', '&', self.lines[self.now])
            self.count_replace += 1
        # 替换为或门：
        elif action == 4:
            self.lines[self.now] = re.sub(r'(&|\||\^)', '|', self.lines[self.now])
            self.count_replace += 1
        # 替换为异或门：
        elif action == 5:
            self.lines[self.now] = re.sub(r'(&|\||\^)', '^', self.lines[self.now])
            self.count_replace += 1
        # 左取非：
        elif action == 6:
            self.count_not += 1
            left_signal,_ = re.findall(r'(!?\w\[\d*\])\s*(&|\||\^)', self.lines[self.now]).pop()
            if '!' in left_signal:
                self.lines[self.now] = self.lines[self.now].replace(left_signal, left_signal.replace('!',''))
            else:
                self.lines[self.now] = self.lines[self.now].replace(left_signal, '!'+left_signal)
        # 右取非或单取非：
        elif action == 7:
            self.count_not += 1
            right_signal = re.findall(r'(!?\w\[\d*\]);', self.lines[self.now]).pop()
            if '!' in right_signal:
                self.lines[self.now] = self.lines[self.now].replace(right_signal, right_signal.replace('!',''))
            else:
                self.lines[self.now] = self.lines[self.now].replace(right_signal, '!'+right_signal)

        start = int(max(0, self.now - self.line_range/2))
        # if self.now > int(len(self.lines)) - int(self.line_range/2):
        #     start = int(len(self.lines)) - int(self.line_range)
        end = int(min(len(self.lines), self.now + self.line_range/2))

        self.next_state = '\n'.join(self.lines[start:end])
        if(len(self.next_state)<self.block_size):
            self.next_state = self.next_state + ' '*(self.block_size - len(self.next_state))
            print('Time to stop!')
        print(self.lines[self.now])
        self.next_state = self.Trans(self.next_state)
        print(f'  action:  {action}')
        print('*'*80)
        print('\n\n')

        return self.next_state, self.reward, self.done

if __name__ == '__main__':
    line_range = 400
    with open('temp.v', 'r') as f:
        state = f.read()
    with open('data\meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    envi = Env(state=state, line_range=line_range, mse_max= 1000,stoi=stoi, block_size=256)
    envi.reset()
    for i in [1,7,0,7]:
        next_state, reward, done = envi.step(i)
        start = int(max(0, envi.now - 1))
        print(envi.lines[start:envi.now+1])
        print(reward)
        print(done)
        print('-----------------')
        
        