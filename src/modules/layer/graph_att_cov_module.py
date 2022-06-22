import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .graph_conv_layer import GraphConvolutionModule
from .attention_module import AttentionModule
from .mlp_encoder_module import MLPEncoderModule

class GraphAttEmbeddingConv(nn.Module):
    def __init__(
        self,
        n_agents,
        input_dim,
        emb_dim=32,
        encoder_hidden_sizes=(64,),
        attention_type="general",
        n_gcn_layers=2,
        gcn_bias = True,
        residual = True
    ):
        super(GraphAttEmbeddingConv, self).__init__()

        self.n_agents = n_agents
        self.input_shape = input_dim
        self.emb_dim = emb_dim
        self.attention_type = attention_type
        self.n_gcn_layers = n_gcn_layers
        self.gcn_bias = gcn_bias
        self.residual = residual

        self.encoder = MLPEncoderModule(input_dim=self.input_shape,
                                        output_dim=self.emb_dim,
                                        hidden_sizes=encoder_hidden_sizes,
                                        output_nonlinearity=th.tanh)

        self.attention_layer = AttentionModule(dimensions=self.emb_dim, 
                                               attention_type=self.attention_type)

        self.gcn_layers = th.nn.ModuleList([
            GraphConvolutionModule(in_features=self.emb_dim, 
                                   out_features=self.emb_dim, 
                                   bias=self.gcn_bias, 
                                   id=i) for i in range(self.n_gcn_layers)
        ])

    def forward(self, inputs):
        embeddings_collection = []
        embeddings_0 = self.encoder.forward(inputs)
        embeddings_collection.append(embeddings_0)

        # (n_paths, max_path_length, n_agents, n_agents)
        # or (n_agents, n_agents)
        attention_weights = self.attention_layer.forward(embeddings_0)

        for i_layer, gcn_layer in enumerate(self.gcn_layers):
            # (n_paths, max_path_length, n_agents, emb_feat_dim)
            # or (n_agents, emb_feat_dim)
            embeddings_gcn = gcn_layer.forward(embeddings_collection[i_layer], 
                                               attention_weights)
            embeddings_collection.append(embeddings_gcn)

        if self.residual:
            emb = embeddings_collection[0] + embeddings_collection[-1]
        else:
            emb = embeddings_collection[-1]

        return emb