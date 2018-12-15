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
from gnn import *

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--task_id', type=int, default=4, help='bAbI task id')
parser.add_argument('--question_id', type=int, default=0, help='question types')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=20, help='input batch size')
parser.add_argument('--state_dim', type=int, default=40, help='GGNN hidden state size')
parser.add_argument('--n_steps', type=int, default=5, help='propogation steps number of GGNN')
parser.add_argument('--niter', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--verbal', action='store_true', help='print training info or not')
parser.add_argument('--seed', type=int, help='manual seed', default=543, metavar='N')
opt = parser.parse_args()


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

def create_adjacency_matrix(edges, n_nodes, n_edge_types):
    a = np.zeros([n_nodes, n_nodes * n_edge_types * 2])
    for edge in edges:
        src_idx = edge[0]
        e_type = edge[1]
        tgt_idx = edge[2]
        a[tgt_idx-1][(e_type - 1) * n_nodes + src_idx - 1] =  1
        a[src_idx-1][(e_type - 1 + n_edge_types) * n_nodes + tgt_idx - 1] =  1
    return a

class SatEnv():
    def __init__(self, filename):
        self.filename = filename
        self.map = {}
        self.reset()
    def reset(self, filename=None):
        if filename: self.formula = parse_dimacs(filename)
        else: self.formula = parse_dimacs(self.filename)
        self.state = State(self.formula, (), ())
        #self.state = shuffle_state(self.state)
        return self.emb()
    def emb(self):
        return self.graph_emb()
        """
        def clause_reshape(x):
            if len(x) < 3: return x + ([0.0] * (3-len(x)))
            else: return x
        cs = [clause_reshape(c.xs) for c in self.state.formula.cs]
        if len(cs) < 91: return torch.tensor(cs + ([([0.0]*3)] * (91-len(cs)))).flatten()
        else: return torch.tensor(cs).flatten()
        """
    def numbering(self):
        self.map = {}
        self.revmap = {}

    def node2idx(self, n): 
        pol = 0 if n > 0 else 1
        return (abs(n)-1) * 2 + pol
        
    def graph_emb(self):
        self.numbering()
        all_var = self.state.formula.allVars
        all_var_uniq = list(set([abs(x) for x in all_var]))
        #print("len all var: {}".format(len(all_var)))

        gnn = GatedGraphNeuralNetwork(hidden_size=32, num_edge_types=2,
                                      layer_timesteps=[3, 5, 7, 2], residual_connections={2: [0], 3: [0, 1]})
        n_nodes = len(all_var_uniq) * 2 + len(self.state.formula.cs)

        lit_edges = []
        for v in range(1, len(all_var_uniq)+1):
            lit_edges.append((self.node2idx(v), self.node2idx(-v)))
            self.map[self.node2idx(v)] = all_var_uniq[v-1]
            self.map[self.node2idx(-v)] = -all_var_uniq[v-1]
            self.revmap[all_var_uniq[v-1]] = self.node2idx(v)
            self.revmap[-all_var_uniq[v-1]] = self.node2idx(-v)

        #print(lit_edges)
        lit_edges = AdjacencyList(node_num=n_nodes, adj_list=lit_edges, device=gnn.device)

        clause_edges = []
        for i, c in enumerate(self.state.formula.cs):
            for x in c.xs: 
                clause_edges.append((self.revmap[x], i+(len(all_var_uniq)*2)))
        #print(clause_edges)
        clause_edges = AdjacencyList(node_num=n_nodes, adj_list=clause_edges, device=gnn.device)

        output = gnn.compute_node_representations(initial_node_representation=torch.randn(n_nodes, 32),
                                                  adjacency_lists=[lit_edges, clause_edges])
        print("output shape: {}".format(output.shape))
        return output[:len(all_var_uniq)*2]

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
            emb = None if done else self.emb()
            return emb, reward, done, self.state.assignment
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
        emb = None if done else self.emb()
        return emb, reward, done, self.state.assignment

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(32, 128)
        self.affine2 = nn.Linear(128, 64)
        self.affine3 = nn.Linear(64, 1)
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
torch.manual_seed(opt.seed)

policy = Policy()

optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

def select_action(state, env): 
    # state now is a list of 4 double
    #state = torch.from_numpy(state).float().unsqueeze(0) # a tensor of 4 double
    state = state.float().unsqueeze(0)
    probs = policy(state)[0].squeeze(1) #output the probs of each variable
    #print(probs)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    
    """
    sp = list(env.action_space())
    ws = [probs[(abs(a)-1)*2] for a in sp]
    cumdist = list(itertools.accumulate(ws))
    x = random.random() * cumdist[-1]
    action = sp[bisect.bisect(cumdist, x)]
    idx = torch.tensor([(abs(action)-1)*2])
    policy.saved_log_probs.append(m.log_prob(idx))
    """
    return env.map[action.data.item()]

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
        R = r + opt.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    for pl in policy_loss:
        optimizer.zero_grad()
        pl.backward()
        optimizer.step()
    """
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    """
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
                except IndexError as e:
                    raise e
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
