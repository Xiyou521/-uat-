from turtle import forward
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

from ..layer.transformer import TransformerBlock


class TransQMixer(nn.Module):
    def __init__(self, args, abs=True):
        super(TransQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents

        self.n_agents   = args.env_args["num_agents"]
        self.n_entities = args.env_args["num_agents"]+args.env_args["num_landmarks"]
        self.feat_dim   = args.feat_dim
        self.emb_dim    = args.mixer_emb
        self.transformer = TransformerMixer(args.feat_dim, self.emb_dim, args.mixer_heads, args.mixer_depth, self.emb_dim )

        self.abs = abs # monotonicity constraint
        self.qmix_pos_func = getattr(self.args, "qmix_pos_func", "abs")
        self.custom_space = args.env_args.get("use_custom_state", False)
        
        # The hypernet weights are given by the embeddings of the transformer
        self.hyper_b1 = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(self.emb_dim, 1))

        if getattr(args, "use_orthogonal", False):
            for m in self.modules():
                orthogonal_init_(m)

    def forward(self, qvals, states, obs):
        # reshape
        b, t, _ = qvals.size()

        qvals = qvals.reshape(b * t, 1, self.n_agents)

        #(time_steps, state_entities, n_tokens)
        if self.custom_space:
            inputs = states.reshape(-1, self.n_entities, self.feat_dim)
        else:
            inputs = obs.reshape(-1, self.n_agents*self.n_entities, self.feat_dim)

        #(time_steps, state_entities, emb)
        embs, _ = self.transformer(inputs, None)


        # decompose the embeddings if possible
        if inputs.shape[1] >= self.n_agents + 3:

            # First layer
            w1 = embs[:, :self.n_agents, :].view(-1, self.n_agents, self.emb_dim)
            b1 = embs[:, -1, :].view(-1, 1, self.emb_dim) # start from the bottom for the other hypernet embeddings
            
            # Second layer
            w2 = embs[:, -2, :].view(-1, self.emb_dim, 1) # b * t, emb, 1
            b2 = F.relu(self.hyper_b2(embs[:, -3, :])).view(-1, 1, 1)
            
            if self.abs:
                w1 = self.pos_func(w1)
                w2 = self.pos_func(w2)

        # otherwise average the embeddings
        else:

            e_mean = embs.mean(dim=1, keepdim=True)

            # First layer
            w1 = embs[:, :self.n_agents, :].view(-1, self.n_agents, self.emb_dim)
            b1 = self.hyper_b1(e_mean).view(-1, 1, self.emb_dim) # start from the bottom for the other hypernet embeddings
            
            # Second layer
            w2 = e_mean.view(-1, self.emb_dim, 1) # b * t, emb, 1
            b2 = F.relu(self.hyper_b2(embs[:, -3, :])).view(-1, 1, 1)
            
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

class TransformerMixer(nn.Module):

    def __init__(self, input_dim, emb, heads, depth, output_dim):
        super().__init__()

        self.num_tokens = output_dim

        self.token_embedding = nn.Linear(input_dim, emb)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, mask=False))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, output_dim)

    def forward(self, x, mask):

        tokens = self.token_embedding(x)

        b, t, e = tokens.size()

        x, mask = self.tblocks((tokens, mask))

        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)

        return x, tokens