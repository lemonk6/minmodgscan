import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import numpy.random as rnd


class ESN(nn.Module):
    '''
    Echo state network (reservoir computer) used to create command embeddings. Code for the Echo State Network taken
    directly from Vadim Alperovich's (no affiliation) TextGenESN project on Github (github.com/VirtualRoyalty/TextGenESN)
    '''

    def __init__(self,
                 n_in, n_res, n_out,
                 ro_hidden_layers=1,
                 lin_size=64,
                 density=0.2,
                 spec_radius=0.99,
                 leaking_rate=1.0,
                 in_scaling=1.0,
                 dropout_rate=0.1,
                 batch_size=300,
                 device=torch.device('cuda'),
                 is_feedback=False,
                 fb_scaling=1.0,
                 **params):
        super(ESN, self).__init__()

        self.n_in = n_in
        self.n_res = n_res
        self.n_out = n_out
        self.device = device
        self.batch_size = batch_size
        self.is_feedback = is_feedback
        self.in_scaling = in_scaling
        self.leaking_rate = leaking_rate
        # readout layers
        self.readout_in = nn.Linear(n_res, lin_size)
        self.hidden_ro_layers = [nn.Linear(lin_size, lin_size).to(device) for i in range(ro_hidden_layers - 1)]
        self.readout_out = nn.Linear(lin_size, self.n_out)

        self.dropout = nn.Dropout(p=dropout_rate)
        if self.is_feedback:
            self.prev_input = torch.zeros(batch_size, n_out, requires_grad=False).to(self.device)

        # reservoir initiation
        try:
            self.w_input = params['w_input']
            self.w_res = params['w_res']
            self.w_fb = params['w_fb']
            self.readout_in.weight = nn.Parameter(params['readout_in.weight'])
            self.readout_in.bias = nn.Parameter(params['readout_in.bias'])
            self.readout_out.weight = nn.Parameter(params['readout_out.weight'])
            self.readout_out.bias = nn.Parameter(params['readout_out.bias'])
            print('External reservoir set')

        except:
            self.w_input = nn.Parameter(self.initiate_in_reservoir(n_res, n_in, scaling=in_scaling).to(device))
            self.w_res = nn.Parameter(self.initiate_reservoir(density, n_res, spec_radius, device).float())
            self.w_fb = nn.Parameter(self.initiate_fb_reservoir(n_res, n_out, scaling=fb_scaling).to(device))

            print('Internal reservoir set')
            n_non_zero = self.w_res[self.w_res > 0.01].shape[0]
            print('Reservoir has {} non zero values ({:.2%})' \
                  .format(n_non_zero, n_non_zero / (n_res ** 2)))

        return

    def forward(self, input, hidden_state):
        state = torch.mm(self.w_input, input.T) + torch.mm(self.w_res, hidden_state)
        hidden_state = hidden_state * (1 - self.leaking_rate)
        if self.is_feedback:
            hidden_state += self.leaking_rate * torch.tanh(state + torch.mm(self.w_fb, self.prev_input.T))
            self.prev_input = input
        else:
            hidden_state += self.leaking_rate * torch.tanh(state)

        output = self.readout_in(hidden_state.T)

        return hidden_state, output

    def init_hidden(self):
        return Variable(torch.zeros(self.n_res, self.batch_size, requires_grad=False)).to(self.device)

    def initiate_in_reservoir(self, n_reservoir, n_input, scaling):

        w_input = np.random.rand(n_reservoir, n_input) - 0.5
        w_input = w_input * scaling
        w_input = torch.tensor(w_input, dtype=torch.float32, requires_grad=False)

        return w_input

    def initiate_fb_reservoir(self, n_reservoir, n_output, scaling=1.0):

        w_fb = np.random.rand(n_reservoir, n_output) - 0.5
        w_fb *= scaling
        w_fb = torch.tensor(w_fb, dtype=torch.float32, requires_grad=False)
        return w_fb

    def initiate_reservoir(self, density, n_res, spec_radius, device):

        w_res = np.identity(n_res)
        w_res = np.random.permutation(w_res)
        w_res = torch.tensor(w_res, requires_grad=False).to(device)

        number_nonzero_elements = density * n_res * n_res
        while np.count_nonzero(w_res.cpu().data) < number_nonzero_elements:
            q = torch.tensor(self.create_rotation_matrix(n_res), requires_grad=False).to(device)
            w_res = torch.mm(q, w_res)

        w_res *= spec_radius

        return w_res

    def create_rotation_matrix(self, n_reservoir):

        h = rnd.randint(0, n_reservoir)
        k = rnd.randint(0, n_reservoir)
        phi = rnd.rand(1) * 2 * np.pi

        Q = np.identity(n_reservoir)
        Q[h, h] = np.cos(phi)
        Q[k, k] = np.cos(phi)
        Q[h, k] = - np.sin(phi)
        Q[k, h] = np.sin(phi)

        return Q

    def add_hidden_emb(self, sequence):
        self.embedding = torch.cat((self.embedding, sequence.cpu()), 0)

    def finalize(self):
        self.init_embedding = False


def unravel_index(indices: torch.LongTensor, shape, device):
    '''
    Converts the index of the most-attended grid cell into grid coordinates. Function by François Rozet,
    posted on Dec 20th 2020 at
    https://stackoverflow.com/questions/53212507/how-to-efficiently-retrieve-the-indices-of-maximum-values-in-a-torch-tensor/65168284#65168284
    :param indices: index of the most-attended grid cell
    :param shape: grid dimensions
    :param device: cpu or cuda
    :return: grid coordinates of the most-attended grid cell
    '''
    shape = torch.tensor(shape).to(device)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int).to(device)
    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = torch.div(indices, dim, rounding_mode='trunc')
    return coord.flip(-1)


class MLP(nn.Module):
    '''
    The agent's controller network, outputs prediction for the next step in an action sequence.
    '''

    def __init__(self, hidden_dim=100, out_dim=6, language_in_dim=64, feature_dim=16, grid_size=6, device="cpu",
                 selective_attention=True, action_attention=True, sub_pos=False):
        super(MLP, self).__init__()
        if selective_attention and not sub_pos:  # selective attention and absolute agent positions as input (standard)
            self.in_dim = 270 + feature_dim + 2
        elif selective_attention and sub_pos:  # selective attention and relative target distances as input
            self.in_dim = 270 + feature_dim
        else:  # no selective attention, full grid as input
            self.in_dim = 264 + grid_size * grid_size * feature_dim
        # when full grid is input, increase number of neurons to account for higher dimensionality of input features
        if not selective_attention:
            self.hidden_dim = 500
        else:
            self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.language_in_dim = language_in_dim
        self.feature_dim = feature_dim
        self.grid_size = grid_size
        self.device = torch.device(device)
        self.action_attention = action_attention
        self.selective_attention = selective_attention
        self.sub_pos = sub_pos

        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim, bias=False),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim, bias=False)
        )
        self.att = nn.Linear(self.language_in_dim, self.feature_dim, bias=False)  # selective attention key matrix
        if self.action_attention:
            self.att_lang = nn.Linear(self.language_in_dim, self.language_in_dim,
                                      bias=False)  # self-attention key matrix
            self.att_action = nn.Linear(self.language_in_dim, 200,
                                        bias=False)  # attention key matrix for action attention
        if selective_attention:  # if selective attention optimized with CMA-ES, no grad needed
            self.att.requires_grad_ = False

    def forward(self, states, partial_input_features, most_attended_loc=None):
        batch_size = partial_input_features.shape[0]
        states = states.permute(0, 3, 1, 2)
        # If action attention is used, command embedding goes through a self-attention key matrix and an action attention
        # key matrix. The result is multiplied with the agent's past 20 actions and orientations and used as input to
        # the controller
        if self.action_attention:
            self_att_output = self.att_lang(
                partial_input_features[:, :, :self.language_in_dim].reshape(batch_size, -1).float())
            weighted_lang = partial_input_features[:, :, :self.language_in_dim].reshape(batch_size,
                                                                                        -1) * self_att_output

            act_att_output = self.att_action(weighted_lang)
            weighted_acts = partial_input_features[:, :, self.language_in_dim:self.language_in_dim + 200].reshape(
                batch_size, -1) * act_att_output
        else:
            weighted_acts = partial_input_features[:, :, self.language_in_dim:self.language_in_dim + 200]
        # if no target location is passed to model, identify the most-attended grid cell
        if not torch.is_tensor(most_attended_loc):
            weighted_channels = self.att(
                partial_input_features[:, :, :self.language_in_dim].reshape(batch_size, -1).float())
            weighted_grid = torch.matmul(weighted_channels.reshape(batch_size, 1, -1),
                                         states.reshape(batch_size, self.feature_dim, -1))
            most_attended_loc = torch.argmax(weighted_grid, dim=2)
            most_attended_loc = unravel_index(most_attended_loc, (self.grid_size, self.grid_size), self.device).reshape(
                -1, 2)
        most_attended_features = states[torch.arange(batch_size), :, most_attended_loc[:, 0].view(-1),
                                 most_attended_loc[:, 1].view(-1)]
        if self.selective_attention and not self.sub_pos: # slightly different inputs for absolute/relative target locations
            combined_input = torch.cat([most_attended_loc / (self.grid_size - 1),
                                        partial_input_features.reshape(batch_size, -1)[:, :self.language_in_dim],
                                        weighted_acts.reshape(batch_size, -1),
                                        partial_input_features.reshape(batch_size, -1)[:, self.language_in_dim + 200:],
                                        most_attended_features], dim=1)
        elif self.selective_attention and self.sub_pos:
            combined_input = torch.cat([most_attended_loc / (self.grid_size - 1) - partial_input_features.reshape(
                batch_size, -1)[:, self.language_in_dim + 200:self.language_in_dim + 202],
                                        partial_input_features.reshape(batch_size, -1)[:, :self.language_in_dim],
                                        weighted_acts.reshape(batch_size, -1),
                                        partial_input_features.reshape(batch_size, -1)[:, self.language_in_dim + 202:],
                                        most_attended_features], dim=1)
        else: # version without selective attention
            full_grid = weighted_grid * states.reshape(batch_size, self.feature_dim, -1)
            combined_input = torch.cat(
                [full_grid.reshape(batch_size, -1),
                 partial_input_features.reshape(batch_size, -1)[:, :self.language_in_dim],
                 weighted_acts.reshape(batch_size, -1)], dim=1)
        predicted_action = self.layers(combined_input)
        return predicted_action, most_attended_loc, states
