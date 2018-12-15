import torch
import torch.nn as nn

# https://github.com/JamesChuanggg/ggnn.pytorch/blob/master/model.py

class AttrProxy():
    """
    Translates index lookups into attribute lookups.
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix
    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class Propogator(nn.Module):
    """
    Gated propogator for GGNN, using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()
        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid()
        )
        self.tranform = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Tanh()
        )
    def forward(self, state_in, state_out, state_cur, A):
        A_in = A[:, :, :self.n_node*self.n_edge_types]
        A_out = A[:, :, self.n_node*self.n_edge_types:]
        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)
        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r*state_cur), 2)
        h_hat = self.tranform(joined_input)
        output = (1 - z) * state_cur + z * h_hat
        return output

class GGNN(nn.Module):
    def __init__(self, opt):
        super(GGNN, self).__init__()
        assert(opt.state_dim >= opt.annotation_dim)
        self.state_dim = opt.state_dim
        self.annotation_dim = opt.annotation_dim
        self.n_edge_types = opt.n_edge_types
        self.n_node = opt.n_node
        self.n_steps = opt.n_steps

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propogation model
        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

        # Output model
        self.out = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, 1)
        )

        self.__initialize()

    def __initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, annotation, A):
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_node * self.n_edge_types, self.state_dim)
            prop_state = self.propogator(in_states, out_states, prop_state, A)
        join_state = torch.cat((prop_state, annotation), 2)
        output = self.out(join_state)
        output = output.sum(2)
        return output
