import torch as th
from torch import nn

from wacky.modules import WackyLayer, AdderLayer, WackyModule


class RecurrentLayer(WackyModule):

    def __init__(self, in_features, out_features, recurrent_cell, num_cells=1, *args, **kwargs):
        super(RecurrentLayer, self).__init__()

        self.num_cells = num_cells
        self.cells = nn.ModuleList()
        for _ in range(num_cells):
            self.cells.append(recurrent_cell(in_features=in_features, out_features=out_features, *args, **kwargs))

        self.hidden_state = th.zeros((out_features))

    def forward(self, x):
        for i in range(self.num_cells):
            self.hidden_state = self.cells[i](x[i], self.hidden_state)


class WackyLSTMCell(WackyModule):

    def __init__(self, in_features, out_features):
        super(WackyLSTMCell, self).__init__()

        cell_features = out_features  # cell_state is the same size as out_features
        self.cell_state = th.zeros((cell_features))

        # current in_features and "last" out_features (from prev cell)
        layer_in_features = in_features + out_features
        self.forget_gate = WackyLayer(layer_in_features, cell_features, module=nn.Linear, activation=nn.Sigmoid())
        self.new_candidates = WackyLayer(layer_in_features, cell_features, module=nn.Linear, activation=nn.Sigmoid())
        self.candidate_scale = WackyLayer(layer_in_features, cell_features, module=nn.Linear, activation=nn.Tanh())
        self.scale_cell_state = nn.Tanh()
        self.filter_cell_state = WackyLayer(layer_in_features, cell_features, module=nn.Linear, activation=nn.Sigmoid())

    def forward(self, x, h):
        x = th.cat([x, h], -1)
        self.cell_state = self.cell_state * self.forget_gate(x)
        self.cell_state = self.cell_state + self.new_candidates(x) * self.candidate_scale(x)
        scaled_cell_state = self.scale_cell_state(self.cell_state)
        return scaled_cell_state * self.filter_cell_state(x)


class WackyGatedRecurrentUnit(WackyModule):

    def __init__(self, in_features, out_features, hidden_features=None):
        super(WackyGatedRecurrentUnit, self).__init__()

        hidden_features = hidden_features if hidden_features is not None else out_features

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.reset_gate = AdderLayer(in_features, hidden_features, hidden_features, module=nn.Linear, activation=nn.Sigmoid())
        self.update_gate = AdderLayer(in_features, hidden_features, hidden_features, module=nn.Linear, activation=nn.Sigmoid())
        self.contents_x = WackyLayer(in_features, hidden_features, module=nn.Linear, activation=nn.Sigmoid())
        self.contents_h = WackyLayer(hidden_features, hidden_features, module=nn.Linear, activation=nn.Sigmoid())
        self.contents_activation = nn.Tanh()

        if out_features != hidden_features:
            self.out_encoder = WackyLayer(hidden_features, out_features, module=nn.Linear, activation=nn.ReLU())
        else:
            self.out_encoder = None

    def forward(self, x, h):
        #print(x.shape)
        #print(h.shape)
        z = self.update_gate(x, h)
        h_new = self.contents_activation(
            self.contents_x(x) + self.reset_gate(x, h) * self.contents_h(h)
        )

        h = z * h + (1 - z) * h_new

        #print(h.shape)

        if self.out_encoder is not None:
            return self.out_encoder(h)
        else:
            return h


class ImportanceConnector(WackyModule):

    def __init__(self, in_features, out_features, n_connections=1):
        super(ImportanceConnector, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_connections = n_connections

        self.importance = WackyLayer(
            #in_features,
            n_connections,
            n_connections,
            module=nn.Linear,
            activation=nn.Softmax()
        )

        self.recurrent_module = WackyGatedRecurrentUnit(
            in_features,
            out_features,
            hidden_features=out_features*n_connections
        )

    def forward(self, x, h):
        #importance = self.importance(x).reshape(-1,1)
        importance = self.importance(th.ones(self.n_connections)).reshape(-1,1)
        x = self.recurrent_module(x, h)
        #print('x', x.shape)

        outs = []
        for i in range(self.n_connections):
            #print(importance[i])
            outs.append(importance[i] * x)
        return x, outs


class WackyNeuronConnections(WackyModule):

    def __init__(self, in_features, out_features, n_neurons=16, hidden_features=8):
        super(WackyNeuronConnections, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_neurons = n_neurons
        self.hidden_features = hidden_features

        self.neurons = nn.ModuleList()
        for _ in range(n_neurons):
            self.neurons.append(ImportanceConnector(
                in_features=in_features,
                out_features=hidden_features,
                n_connections=n_neurons
            ))

        self.output_encoder = WackyLayer(
            in_features=hidden_features*n_neurons,
            out_features=out_features,
            module=nn.Linear,
            activation=None
        )

        self.reset()

    def reset(self):
        self.h_list = [
            th.zeros(1, self.hidden_features*self.n_neurons) for _ in range(self.n_neurons)
        ]

    def forward(self, x):

        outs_h = []
        outs_x = []
        for n, h in zip(self.neurons, self.h_list):
            new_x, new_h = n(x, h)
            outs_h.append(new_h)
            outs_x.append(new_x)

        #print([elem.shape for elem in self.h_list])

        self.h_list = []
        for elem in list(map(list, zip(*outs_h))):
            self.h_list.append(th.stack(elem).reshape(1,-1))

        #print([elem.shape for elem in self.h_list])
        return self.output_encoder(th.stack(outs_x).reshape(1,-1))










