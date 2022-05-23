import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import numpy.random as rnd


# Code for the Echo State Network taken from Vadim Alperovich's TextGenESN project on Github
# (https://github.com/VirtualRoyalty/TextGenESN)
class ESN(nn.Module):

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
        self.softmax = nn.LogSoftmax()
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


# function by François Rozet, posted on Dec 20th 2020 at
# https://stackoverflow.com/questions/53212507/how-to-efficiently-retrieve-the-indices-of-maximum-values-in-a-torch-tensor/65168284#65168284
def unravel_index(indices: torch.LongTensor, shape, device):
    shape = torch.tensor(shape).to(device)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int).to(device)
    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = torch.div(indices, dim, rounding_mode='trunc')
    return coord.flip(-1)


class MLP(nn.Module):
    def __init__(self, hidden_dim=64, out_dim=6, language_in_dim=64, feature_dim=16, grid_size=6,
                 image_dim=60, CNN=False, device="cpu", selective_attention=True):
        super(MLP, self).__init__()
        if selective_attention:
            self.in_dim = 136 + feature_dim
        else:
            self.in_dim = 134 + grid_size * grid_size
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.language_in_dim = language_in_dim
        self.feature_dim = feature_dim
        self.grid_size = grid_size
        self.kernel_size = int(image_dim / grid_size)
        self.CNN = CNN
        self.device = torch.device(device)
        self.selective_attention = selective_attention

        if self.CNN:
            self.conv = nn.Conv2d(in_channels=3, out_channels=self.feature_dim,
                                  kernel_size=(self.kernel_size, self.kernel_size), stride=self.kernel_size)
            self.conv.requires_grad_(False)

        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim, bias=False),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim, bias=False)
        )

        self.att = nn.Parameter(data=torch.zeros(self.language_in_dim, self.feature_dim))
        if selective_attention:
            self.att.requires_grad_ = False
        self.register_parameter("att", self.att)

    def forward(self, states, partial_input_features, most_attended_loc=None):
        batch_size = partial_input_features.shape[0]
        states = states.permute(0, 3, 1, 2)
        if self.CNN:
            states = self.conv(states / 255)
        if not torch.is_tensor(most_attended_loc):
            att = torch.cat(batch_size * [self.att.unsqueeze(0)])
            weighted_channels = torch.bmm(partial_input_features[:, :, :self.language_in_dim].reshape(batch_size, 1, -1).float(), att)
            weighted_grid = torch.matmul(weighted_channels, states.reshape(batch_size, self.feature_dim, -1)) 
            #print(weighted_grid.reshape(-1, 6, 6))
            most_attended_loc = torch.argmax(weighted_grid, dim=2)
            most_attended_loc = unravel_index(most_attended_loc, (self.grid_size, self.grid_size), self.device).reshape(-1, 2)
        most_attended_features = states[torch.arange(batch_size), :, most_attended_loc[:, 0].view(-1),
                                 most_attended_loc[:, 1].view(-1)]
        if self.selective_attention:
            combined_input = torch.cat([most_attended_loc / (self.grid_size - 1), partial_input_features.reshape(batch_size, -1),
                                    most_attended_features], dim=1)
        else:
            combined_input = torch.cat(
                [weighted_grid.reshape(batch_size, -1), partial_input_features.reshape(batch_size, -1)], dim=1)
        predicted_action = self.layers(combined_input)
        return predicted_action, most_attended_loc, states
