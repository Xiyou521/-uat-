from turtle import forward
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

from ..layer.graph_att_cov_module import GraphAttEmbeddingConv

"""
graph_emb_dim: 64
graph_encoder_hidden_sizes: []
graph_attention_type: "general"
grap_n_gcn_layers: 2
gcn_bias: True
residual: True
"""


class GraphQMixer(nn.Module):
    def __init__(self, args, abs=True):
        super(GraphQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.graph_emb_dim = args.graph_emb_dim
        
        self.graph_input_dim = args.obs_shape
        self.hypernet_input_dim = args.state_shape

        self.abs = abs # monotonicity constraint
        self.qmix_pos_func = getattr(self.args, "qmix_pos_func", "abs")

        # self-attention graph convolution
        self.graph_conv = GraphAttEmbeddingConv(self.n_agents,
                                                input_dim=self.graph_input_dim,
                                                emb_dim=self.graph_emb_dim,
                                                encoder_hidden_sizes=args.graph_encoder_hidden_sizes,
                                                attention_type=args.graph_attention_type,
                                                n_gcn_layers=args.grap_n_gcn_layers)
        
        # hyper w1 b1
        """"
        self.hyper_w1 = nn.Sequential(nn.Linear(self.hypernet_input_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.n_agents * self.embed_dim))
        """
                                 
        self.hyper_b1 = nn.Sequential(nn.Linear(self.hypernet_input_dim, self.graph_emb_dim))
        
        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(nn.Linear(self.hypernet_input_dim, args.hypernet_embed),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(args.hypernet_embed, self.graph_emb_dim))

        self.hyper_b2 = nn.Sequential(nn.Linear(self.hypernet_input_dim, args.hypernet_embed),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(args.hypernet_embed, 1))

        if getattr(args, "use_orthogonal", False):
            for m in self.modules():
                orthogonal_init_(m)

    def forward(self, qvals, states, obs):
        # reshape
        b, t, _ = qvals.size()

        qvals = qvals.reshape(b * t, 1, self.n_agents)
        states = states.reshape(-1, self.hypernet_input_dim)

        #(n_envs, max_path_length, n_agents, obs_size)
        embs = self.graph_conv(obs)

        # First layer
        w1 = embs.reshape(-1, self.n_agents, self.graph_emb_dim) # use the graph embeddings as the hidden layer
        #w1 = self.hyper_w1(states).view(-1, self.n_agents, self.embed_dim) # b * t, n_agents, emb
        b1 = self.hyper_b1(states).view(-1, 1, self.graph_emb_dim)
        
        # Second layer
        w2 = self.hyper_w2(states).view(-1, self.graph_emb_dim, 1) # b * t, emb, 1
        b2= self.hyper_b2(states).view(-1, 1, 1)
        
        if self.abs:
            w1 = self.pos_func(w1)
            w2 = self.pos_func(w2)
            
        # Forward
        hidden = F.elu(th.matmul(qvals, w1) + b1) # b * t, 1, emb
        y = th.matmul(hidden, w2) + b2 # b * t, 1, 1
        
        return y.view(b, t, -1)

    def pos_func(self, x):
        if self.qmix_pos_func == "softplus":
            return th.nn.Softplus(beta=self.args.qmix_pos_func_beta)(x)
        elif self.qmix_pos_func == "quadratic":
            return 0.5 * x ** 2
        else:
            return th.abs(x)
        
