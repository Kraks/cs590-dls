import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

#TODO: batch for a NNet

dtype = torch.FloatTensor

# Reference: https://github.com/priba/nmp_qc/

class NNet(nn.Module):
    def __init__(self, n_in, n_out, hlayers = (128, 258, 128)):
        super(NNet, self).__init__()
        self.n_hlayers = len(hlayers)
        self.fcs = nn.ModuleList([nn.Linear(n_in, hlayers[i]) if i == 0 else
                                  nn.Linear(hlayers[i-1], n_out) if i == self.n_hlayers else
                                  nn.Linear(hlayers[i-1], hlayers[i])
                                  for i in range(self.n_hlayers+1)])
    def forward(self, x):
        x = x.contiguous().view(-1, self.num_flat_features(x))
        for i in range(self.n_hlayers):
            x = F.relu(self.fcs[i](x))
        x = self.fcs[-1](x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size: num_features *= s
        return num_features

class MessageFunction(nn.Module):
    def __init__(self, message_def='mpnn', args={}):
        super(MessageFunction, self).__init__()
        self.m_definition = ''
        self.m_function = None
        self.args = {}
        self.__set_message(message_def, args)
    def forward(self, h_v, h_w, e_vw, args=None):
        return self.m_function(h_v, h_w, e_vw, args)
    def __set_message(self, message_def, args={}):
        self.m_definition = message_def.lower()
        self.m_function = {'mpnn': self.m_mpnn }.get(self.m_definition, None)
        if self.m_function is None:
            raise RuntimeError("Can not find message function for {}".format(self.m_definition))
        init_parameters = { 'mpnn': self.init_mpnn }.get(self.m_definition, lambda x: (nn.ParameterList([]),
                                                                                       nn.ModuleList([]),
                                                                                       {}))
        self.learn_args, self.learn_modules, self.args = init_parameters(args)
        self.m_size = { 'mpnn': self.out_mpnn }.get(self.m_definition, None)
    def get_definition(self): return self.m_definition
    def get_args(self): return self.args
    def get_out_size(self, size_h, size_e, args=None): return self.m_size(size_h, size_e, args)
    # Gilmer et al., Neural Message Passing for Quantum Chemistry
    def m_mpnn(self, h_v, h_w, e_vw, opt={}):
        edge_output = self.learn_modules[0](e_vw)
        edge_output = edge_output.view(-1, self.args['out'], self.args['in'])
        h_w_rows = h_w[..., None].expand(h_w.size(0), h_v.size(1), h_w.size(1)).contiguous()
        h_w_rows = h_w_rows.view(-1, self.args['in'])
        h_multiply = torch.bmm(edge_output, torch.unsqueeze(h_w_rows, 2))
        n_mew = torch.squeeze(h_multiply)
        return m_new
    def out_mpnn(self, size_h, size_e, args):
        return self.args['out']
    def init_mpnn(self, params):
        learn_args, learn_modules = [], []
        args = {}
        args['in'] = params['in']
        args['out'] = params['out']
        learn_modules.append(NNet(n_in = params['edge_feat'], n_out = (params['in'] * params['out'])))
        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args

class UpdateFunction(nn.Module):
    def __init__(self, update_def='mpnn', args={}):
        super(UpdateFunction, self).__init__()
        self.u_definition = ''
        self.u_function = None
        self.args = {}
        self.__set_message(update_def, args)
    def forward(self, h_v, m_v, opt={}):
        return self.u_function(h_v, m_v, opt)
    def __set_message(self, update_def, args):
        self.u_definition = update_def.lower()
        self.u_function = { 'mpnn': self.u_mpnn }.get(self.u_definition, None)
        if self.u_function is None:
            raise RuntimeError("Can not find update function for {}".format(self.u_definition))
        init_parameters = {
            'mpnn': self.init_mpnn
        }.get(self.u_definition, lambda x: (nn.ParameterList([]), nn.ModuleList([]), {}))
        self.learn_args, self.learn_modules, self.args = init_parameters(args)
    def get_definition(self): return self.u_definition
    def get_args(self): return self.args
    def u_mpnn(self, h_v, m_v, opt={}):
        h_in = h_v.view(-1, h_v.size(2))
        m_in = m_v.view(-1, m_v.size(2))
        h_new = self.learn_modules[0](m_in[None, ...], h_in[None,...])[0]
        return torch.squeeze(h_new).view(h_v.size())
    def init_mpnn(self, params):
        learn_args, learn_modules = [], []
        args = []
        args['in_m'] = params['in_m']
        args['out'] = params['out']
        learn_modules.append(nn.GRU(params['in_m'], params['out_m']))
        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args

class ReadoutFunction(nn.Module):
    def __init__(self, readout_def='mpnn', args={}):
        super(ReadoutFunction, self).__init__()
        self.r_definition = ''
        self.r_function = None
        self.args = {}
        self.__set_readout(readout_def, args)
    def forward(self, h_v):
        return self.r_function(h_v)
    def __set_readout(self, readout_def, args):
        self.r_definition = readout_def.lower()
        self.r_function = { 'mpnn': self.r_mpnn }.get(self.r_definition, None)
        if self.r_function is None:
            raise RuntimeError("Can not find readout function for {}".format(self.r_definition))
        init_parameters = {
            'mpnn': self.init_mpnn
        }.get(self.r_definition, lambda x: (nn.ParameterList([]), nn.ModuleList([]), {}))
        self.learn_args, self.learn_modules, self.args = init_parameters(args)
    def get_definition(self): return self.r_definition
    def r_mpnn(self, h):
        aux = Variable(torch.Tensor(h[0].size(0), self.args['out']).type_as(h[0].data).zero_())
        for i in range(h[0].size(0)):
            nn_res = nn.Sigmoid()(self.learn_modules[0](torch.cat([h[0][i, :, :], h[-1][i, :, :]], 1))) * self.learn_modules[1](h[-1][i,:,:])
            nn_res = (torch.sum(h[0][i, :, :], 1).expand_as(nn_res) > 0).type_as(nn_res) * nn_res
            aux[i, :] = torch.sum(nn_res, 0)
        return aux
    def init_mpnn(self, params):
        learn_args, learn_modules = [], []
        args = {}
        learn_modules.append(NNet(n_in = 2 * params['in'], n_out = params['target']))
        learn_modules.append(NNet(n_in = params['in'], n_out = params['target']))
        args['out'] = params['target']
        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args

class MPNN(nn.Module):
    """
    in_n : int list
        Size for the node and edge features.
    hidden_state_size: int
        Size of the hidden states (the input will be padded with 0's to this size).
    message_size : int
        Message function output vector size.
    n_layers : int
        Number of iterations Message+Update (weight trying).
    l_target : int
        Size of the output.
    type : str (Optional)
        Classificication | Regression
        If classificication, LogSoftmax layer is applied to the output vector.
    """
    def __init__(self, in_n, hidden_state_size, message_size, n_layers, l_target, type='regression'):
        super(MPNN, self).__init__()
        self.m = nn.ModuleList(
            [MessageFunction('mpnn', args={'edge_feat': in_n[1],
                                           'in': hidden_state_size,
                                           'out': message_size})])
        self.u = nn.ModuleList(
            [UpdateFunction('mpnn', args={'in_m': message_size,
                                          'out': hidden_state_size})])
        self.r = ReadoutFunction('mpnn', args={'in': hidden_state_size,
                                               'target': l_target})
        self.type = type
        self.args = {}
        self.args['out'] = hidden_state_size
        self.n_layers = n_layers

    def forward(self, g, h_in, e):
        h = []
        h_t = torch.cat([h_in, Variable(
            torch.zeros(h_in.size(0), h_in.size(1), self.args['out'] - h_in.size(2)).type_as(h_in.data))], 2)
        h.append(h_t.clone())
        for t in range(self.n_layers):
            e_aux = e.view(-1, e.size(3))
            h_aux = h[t].view(-1, h[t].size(2))
            m = self.m[0].forward(h[t], h_aux, e_aux)
            m = m.view(h[0].view(0), h[0].size(1), -1, m.size(1))
            # Nodes without edge set message to 0
            m = torch.unsqueeze(g, 3).expand_as(m) * m
            m = torch.squeeze(torch.sum(m, 1))
            h_t = self.u[0].forward(h[t], m)
            # Delete virtual nodes
            h_t = (torch.sum(h_in, 2).expand_as(h_t) > 0).type_as(h_t) * h_t
            h.append(h_t)
        # Readout
        res = self.r.forward(h)
        if self.type == 'classificication': res = nn.LogSoftmax()(res)
        return res

if __name__ == '__main__':
    net = NNet(n_in = 100, n_out = 1)
    rin = torch.rand(10, 100)
    print(net)
    print(rin)
    print(net(rin))
