import torch.nn as nn
import torch.nn.functional as F
import torch

from ..layer.transformer import Transformer


"""
TODO: 
To make it more general
- get the number of entities and number of agents directly from the args
"""


class TransformerAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(TransformerAgent, self).__init__()
        self.args = args
        self.n_agents   = args.env_args["num_agents"]
        self.n_entities = args.env_args["num_agents"]+args.env_args["num_landmarks"]
        self.feat_dim  = args.feat_dim
        self.emb_dim    = args.emb
        self.transformer = Transformer(args.feat_dim, args.emb, args.heads, args.depth, args.emb)
        self.q_basic = nn.Linear(args.emb, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.args.emb).cuda()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()
        inputs  = inputs.view(-1, self.n_entities, self.feat_dim)
        hidden_state = hidden_state.view(-1, 1, self.emb_dim)
        outputs, _ = self.transformer.forward(inputs, hidden_state, None)
        # first output for 3 action 
        q = self.q_basic(outputs[:, -1, :])

        # last dim for hidden state
        h = outputs[:, -1:, :]

        return q.view(b, a, -1), h.view(b, a, -1)






