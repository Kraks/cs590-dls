import argparse
import gym
import numpy as np
from itertools import count
import os
import random
import itertools
import bisect
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import glob
from common import *

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

def apply_backtrack(s: State) -> State:
    #print("backtrack {}".format(s))
    v, f, asn = s.cont[0]
    b = False if v>0 else True
    return State(f.assign(v, b), (-v,)+asn, s.cont[1:])

def apply_unit(s: State) -> State:
    #print("unit {}".format(s))
    f, asn, cont = s
    new_f, new_asn = f.elimUnit()
    return State(new_f, new_asn + asn, cont)

class SatEnv():
    def __init__(self, filename):
        self.filename = filename
        self.reset()
    def reset(self, filename=None):
        if filename: self.formula = parse_dimacs(filename)
        else: self.formula = parse_dimacs(self.filename)
        self.state = State(self.formula, (), ())
        return self.emb(shuffle_state(self.state))
    def emb(self, state):
        def clause_reshape(x):
            if len(x) < 3: return x + ([0.0] * (3-len(x)))
            else: return x
        cs = [clause_reshape(c.xs) for c in self.state.formula.cs]
        if len(cs) < 91: return torch.tensor(cs + ([([0.0]*3)] * (91-len(cs)))).flatten()
        else: return torch.tensor(cs).flatten()
    def action_space(self) -> List[Lit]:
        return self.state.formula.allVars
    def step_tr(self):
        f, asn, cont = self.state
        if f.isEmpty(): return True
        elif len(cont) == 0 and f.hasUnsat():
            return True
        elif f.hasUnsat():
            self.state = apply_backtrack(self.state)
            return False
        elif f.hasUnit():
            self.state = apply_unit(self.state)
            return self.step_tr()
        v = f.pick()
        self.state = State(f.assign(v, True), (v,)+asn, (Cont(v, f, asn),)+cont)
        return False

    def step(self, action):
        """
        Returns (state, reward, done, info),
        where the info is the assignment.
        """
        f, asn, cont = self.state
        if f.isEmpty(): 
            reward = 1.0
            print("pick {}, done: {}, backtrack/unsat: {}".format(action, True, False))
            return None, reward, True, asn
        elif len(cont) == 0 and f.hasUnsat():
            reward = -1.0
            print("pick {}, done: {}, backtrack/unsat: {}".format(action, True, True))
            return None, reward, True, None
        elif f.hasUnsat():
            self.state = apply_backtrack(self.state)
            reward = -1.0
            done = True if len(self.state.formula.cs) == 0 else False
            print("pick {}, done: {}, backtrack/unsat: {}".format(action, done, True))
            return self.emb(self.state.formula), reward, done, self.state.assignment
        elif f.hasUnit():
            self.state = apply_unit(self.state)
            return self.step(action)
        v = abs(action)
        b = True if action>0 else False
        pre_nvars = len(self.state.formula.allVars)
        self.state = State(f.assign(v, b), (action,)+asn, (Cont(action, f, asn),)+cont)
        cur_nvars = len(self.state.formula.allVars)
        reward = ((pre_nvars - cur_nvars) / float(pre_nvars))
        done = True if len(self.state.formula.cs) == 0 else False
        print("pick {}, done: {}, backtrack/unsat: {}".format(action, done, False))
        return self.emb(self.state.formula), reward, done, self.state.assignment

# TODO: replace it as a GNN
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(3*91, 128)
        self.affine2 = nn.Linear(128, 64)
        self.affine3 = nn.Linear(64, 40)
        self.saved_log_probs = []
        self.rewards = []
    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        scores = self.affine3(x)
        return F.softmax(scores, dim=1)

#env = gym.make('CartPole-v0')
#env.seed(args.seed)
#env = SatEnv('../src/main/resources/uf20-91/uf20-02.cnf')
torch.manual_seed(args.seed)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

def select_action(state, env): 
    # state now is a list of 4 double
    #state = torch.from_numpy(state).float().unsqueeze(0) # a tensor of 4 double
    state = state.float().unsqueeze(0)
    probs = policy(state) #output the probs of each variable
    print(probs)
    m = Categorical(probs)
    sp = list(env.action_space())
    ws = [probs[0][(abs(a)-1)*2] for a in sp]
    cumdist = list(itertools.accumulate(ws))
    x = random.random() * cumdist[-1]
    action = sp[bisect.bisect(cumdist, x)]
    idx = torch.tensor([(abs(action)-1)*2])
    policy.saved_log_probs.append(m.log_prob(idx))
    
    return action

def select_action_test(state, env):
    state = state.float().unsqueeze(0)
    probs = policy(state) #output the probs of each variable
    #print(probs)
    error = all([x.data.item() == 0.0 for x in probs[0]])
    sp = list(env.action_space())
    if error: return sp[0]

    m = Categorical(probs)
    ws = [probs[0][(abs(a)-1)*2] for a in sp]
    cumdist = list(itertools.accumulate(ws))
    x = random.random() * cumdist[-1]
    try:
        action = sp[bisect.bisect(cumdist, x)]
        return action
    except IndexError:
        return sp[0]

def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

dump_name = 'policy.pkl'

def main():
    global policy
    sats = glob.glob("../src/main/resources/uf20-91/*.cnf")
    if os.path.isfile(dump_name):
        print("load existing mode")
        policy = torch.load(dump_name)
        policy.eval()
    else:
        for filename in sats[0:100]:
            print('Training with {}'.format(filename))
            env = SatEnv(filename)
            for i_episode in range(1, 10):
                print("Start episode {}".format(i_episode))
                try:
                    state = env.reset()
                    for t in range(20 ** 2):
                        action = select_action(state, env)
                        state, reward, done, _ = env.step(action)
                        policy.rewards.append(reward)
                        if done: break

                    finish_episode()
                    print('Episode {}\tLast length: {:5d}'.format( i_episode, t))
                except IndexError:
                    print('catch index error')
                    continue
        torch.save(policy, dump_name)
        print("training done...")

    for filename in sats[100:150]:
        env = SatEnv(filename)
        state = env.reset()
        steps = 0
        while True:
            steps += 1
            action = select_action_test(state, env)
            _, _, done, _ = env.step(action)
            if done: break
        print("{}, NN steps {}".format(filename, steps))

        env.reset()
        steps = 0
        while True:
            steps += 1
            done = env.step_tr()
            if done: break
        print("{}, TR steps {}".format(filename, steps))

if __name__ == "__main__":
    main()
