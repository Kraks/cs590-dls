import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

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
    return State(f.assign(v, False), (-v,)+asn, s.cont[1:])

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
        return self.emb(self.state.shuffle())
    def emb(self, state):
        def clause_reshape(x):
            if len(x) < 3: return x + ([0.0] * (3-len(x)))
            else: return x
        cs = [clause_reshape(c.xs) for c in self.state.formula.cs]
        if len(cs) < 91: return torch.tensor(cs + ([([0.0]*3)] * (91-len(cs)))).flatten()
        else: return torch.tensor(cs).flatten()
    def action_space(self) -> List[Lit]:
        return [abs(x) for x in self.state.formula.allVars]
    def step(self, action):
        """
        Returns (state, reward, done, info),
        where the info is the assignment.
        """
        f, asn, cont = self.state
        if f.isEmpty(): 
            reward = 0.0
            return None, reward, True, asn
        elif len(cont) == 0 and f.hasUnsat():
            reward = 0.0
            return None, reward, True, None
        elif f.hasUnsat():
            self.state = apply_backtrack(self.state)
            reward = 0.0
            return self.emb(self.state.formula), reward, False, self.state.assignment
        elif f.hasUnit():
            self.state = apply_unit(self.state)
            return self.step(action)
        v = action
        pre_nvars = len(self.state.formula.allVars)
        self.state = State(f.assign(v, True), (v,)+asn, (Cont(v, f, asn),)+cont)
        cur_nvars = len(self.state.formula.allVars)
        reward = ((pre_nvars - cur_nvars) / pre_nvars) * 100.0
        return self.emb(self.state.formula), reward, False, self.state.assignment

# TODO: replace it as a GNN
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(3*91, 256)
        self.affine2 = nn.Linear(256, 128)
        self.affine3 = nn.Linear(128, 20)
        self.saved_log_probs = []
        self.rewards = []
    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        scores = self.affine3(x)
        return F.softmax(scores, dim=1)

#env = gym.make('CartPole-v0')
#env.seed(args.seed)
env = SatEnv('/home/kraks/research/sat/src/main/resources/uf20-91/uf20-02.cnf')
torch.manual_seed(args.seed)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

# TODO: select a variable
# TODO: how to represent this action as a tensor? i.e., which branching variable.
# TODO: how to represent a state?
#       formula, decision history
# TODO: invarant
def select_action(state): 
    # state now is a list of 4 double
    #state = torch.from_numpy(state).float().unsqueeze(0) # a tensor of 4 double
    state = state.float().unsqueeze(0)
    probs = policy(state) #output the probs of each variable
    m = Categorical(probs) #transform to categories

    print(probs)
    action = m.sample() + 1 #choose one
    while action.item() not in env.action_space():
        action = m.sample() + 1

    policy.saved_log_probs.append(m.log_prob(action-1))
    return action.item()

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

def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(20 ** 2):
            action = select_action(state)
            print("Picking {}".format(action))
            state, reward, done, _ = env.step(action)
            #if args.render: env.render()
            policy.rewards.append(reward)
            if done: break
        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        """
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break
        """

if __name__ == "__main__":
    main()
