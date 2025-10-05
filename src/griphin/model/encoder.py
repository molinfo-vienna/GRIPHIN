from abc import ABC

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    BatchNorm,
    GATConv,
    GINEConv,
    TransformerConv,
    GATv2Conv,
    MLP,
)
from torch_geometric.data import Data


class PositionalEncodingMixin(ABC):
    def sinusoidal_positional_encoding(self, positions, pos_embedding_dim):
        assert pos_embedding_dim % 2 == 0, "Embedding dimension must be even."

        n_nodes, coord_dim = positions.shape
        assert coord_dim == 3, "Input positions must have shape (n_nodes, 3)."

        # Ensure the embedding dimension is divisible by 3
        if pos_embedding_dim % 3 != 0:
            raise ValueError(
                f"Embedding dimension ({pos_embedding_dim}) must be divisible by 3 for x, y, z coordinates."
            )

        if pos_embedding_dim == 0:
            return torch.empty((n_nodes, 0), device=positions.device)

        # Create a range of frequencies for the sinusoidal encoding
        div_term = torch.exp(
            torch.arange(0, pos_embedding_dim // 2, device=positions.device)
            * -(torch.log(torch.tensor(10000.0)) / (pos_embedding_dim // 2))
        )

        # Initialize the embedding matrix
        embeddings = torch.zeros((n_nodes, pos_embedding_dim), device=positions.device)

        # Number of dimensions allocated to each coordinate
        coord_embedding_dim = pos_embedding_dim // 3

        # Apply sinusoidal encoding for each coordinate (x, y, z)
        for i in range(3):  # Loop over the 3 coordinates
            pos = positions[:, i][:, torch.newaxis]  # Shape: (n_nodes, 1)
            sinusoidal = pos * div_term[: coord_embedding_dim // 2]  # Match the size
            embeddings[
                :,
                i * coord_embedding_dim : (i + 1) * coord_embedding_dim,
            ] = torch.cat([torch.sin(sinusoidal), torch.cos(sinusoidal)], axis=-1)

        return embeddings


class Voxel3DCNN(nn.Module):
    def __init__(
        self,
        first_layer_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_exclusion_channel: bool = False,
    ):
        super(Voxel3DCNN, self).__init__()

        self.first_layer_channels = first_layer_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_exclusion_channel = use_exclusion_channel
        if use_exclusion_channel:
            input_channels = 10
        else:
            input_channels = 9
        # Init module lists
        self.gamma = nn.Parameter(torch.ones(1, input_channels, 1, 1, 1))
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_channels = 9 if not use_exclusion_channel else 10
        out_channels = first_layer_channels

        for i in range(num_layers):
            self.convs.append(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            self.bns.append(nn.BatchNorm3d(out_channels))
            self.dropouts.append(nn.Dropout3d(p=dropout))
            in_channels = out_channels
            out_channels = out_channels * 2 if i < num_layers - 1 else out_channels

        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.out_dim = out_channels

    def forward(self, data):
        if not self.use_exclusion_channel:
            x = data.vals[..., :9]
        else:
            x = data.vals[..., :10]

        x = x.permute(
            0, 4, 1, 2, 3
        )  # Permute to (batch_size, channels, depth, height, width)
        x = torch.nan_to_num(x, nan=0.0)
        x = x.to(torch.float32)  # / 255.0
        x = self.gamma * x  # Apply learnable scaling
        for i in range(len(self.convs)):
            # Apply convolution, batch normalization, ReLU activation, and dropout
            x = self.pool(F.relu(self.bns[i](self.convs[i](x))))
            x = self.dropouts[i](x)

        x = F.adaptive_avg_pool3d(x, output_size=1)

        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, start_dim=1)  # Output: (batch_size, 1024)

        return x


class GNNEncoder(torch.nn.Module, PositionalEncodingMixin):
    def __init__(
        self,
        pos_embedding_dim: int = 240,
        context_vector_dim: int = 2048,
        types_embedding_dim: int = 128,
        output_dim: int = 1024,
        num_layers: int = 3,
        num_edge_features: int = 5,
        dropout: float = 0.3,
        residual_connection: str = "res",
        node_pooling: str = "add",
        gnn_conv: str = "GAT",
    ) -> None:
        super().__init__()
        hidden_dim = pos_embedding_dim + context_vector_dim + types_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.context_vector_dim = context_vector_dim
        self.types_embedding_dim = types_embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_edge_features = num_edge_features
        self.dropout = dropout
        self.residual_connection = residual_connection
        self.node_pooling = node_pooling
        self.gnn_conv = gnn_conv

        # Initialize embeddings
        self.initial_embedding_layer = nn.Linear(118, self.types_embedding_dim)
        input_dim = (
            self.types_embedding_dim + self.pos_embedding_dim + self.context_vector_dim
        )

        # Convolutional layers
        self.convolution = torch.nn.ModuleList()
        self.convolution_batch_norm = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            if self.gnn_conv == "GAT":
                self.convolution.append(
                    GATConv(
                        input_dim,
                        hidden_dim,
                        heads=4,
                        concat=False,
                    )
                )
            elif self.gnn_conv == "GINE":
                self.convolution.append(
                    GINEConv(
                        input_dim,
                        hidden_dim,
                        heads=4,
                        concat=False,
                    )
                )
            self.convolution_batch_norm.append(BatchNorm(hidden_dim))
            if self.residual_connection == "dense":
                input_dim += hidden_dim
            if self.residual_connection == "res":
                input_dim = hidden_dim

        self.linear = nn.Linear(input_dim, output_dim)
        if self.node_pooling == "add":
            self.pooling = global_add_pool
        elif self.node_pooling == "mean":
            self.pooling = global_mean_pool
        elif self.node_pooling == None:
            self.pooling = torch.nn.Identity()
        else:
            raise ValueError(f"Unknown pooling method: {self.node_pooling}")

    def forward(self, data: Data, context_vector) -> Tensor:
        types = F.one_hot(data.x.flatten(), num_classes=118).type(torch.float)
        pos_embedding = self.sinusoidal_positional_encoding(
            data.pos, self.pos_embedding_dim
        )
        types_embedding = self.initial_embedding_layer(types)
        x = torch.cat(
            (types_embedding, pos_embedding, context_vector[data.batch]), dim=1
        )

        for i, conv in enumerate(self.convolution):
            x_conv = conv(x, data.edge_index, data.edge_attr.type(torch.float))
            x_conv = self.convolution_batch_norm[i](x_conv)
            x_conv = F.gelu(x_conv)
            if self.residual_connection == "dense":
                x = torch.cat((x, x_conv), dim=1)
            if self.residual_connection == "res":
                x = x + x_conv
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linear(x)
        representation = self.pooling(x, data.batch)

        return representation


class GNNEncoder2(torch.nn.Module, PositionalEncodingMixin):
    def __init__(
        self,
        pos_embedding_dim: int = 240,
        context_vector_dim: int = 2048,
        types_embedding_dim: int = 128,
        output_dim: int = 1024,
        num_layers: int = 3,
        num_edge_features: int = 5,
        dropout: float = 0.3,
        residual_connection: str = "res",
        node_pooling: str = "add",
        gnn_conv: str = "GAT",
    ) -> None:
        super().__init__()
        self.pos_embedding_dim = pos_embedding_dim
        self.context_vector_dim = context_vector_dim
        self.types_embedding_dim = types_embedding_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_edge_features = num_edge_features
        self.dropout = dropout
        self.residual_connection = residual_connection
        hidden_dim = types_embedding_dim
        self.node_pooling = node_pooling
        self.gnn_conv = gnn_conv

        # Initialize embeddings
        self.initial_embedding_layer = nn.Linear(118, hidden_dim)

        # Convolutional layers
        self.convolution = torch.nn.ModuleList()
        self.convolution_batch_norm = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            input_dim = hidden_dim + self.pos_embedding_dim + self.context_vector_dim
            if self.gnn_conv == "GAT":
                self.convolution.append(
                    GATConv(
                        input_dim,
                        hidden_dim,
                        heads=4,
                        concat=False,
                    )
                )
            elif self.gnn_conv == "GATv1":
                self.convolution.append(
                    GATConv(
                        input_dim,
                        hidden_dim,
                        heads=4,
                        concat=False,
                        edge_dim=5,
                    )
                )
            elif self.gnn_conv == "GATv2":
                self.convolution.append(
                    GATv2Conv(
                        input_dim,
                        hidden_dim,
                        heads=4,
                        concat=False,
                        edge_dim=5,
                    )
                )
            elif self.gnn_conv == "GINE":
                self.convolution.append(
                    GINEConv(
                        nn=nn.Linear(input_dim, hidden_dim),
                        edge_dim=5,
                    )
                )
            elif self.gnn_conv == "TransformerConv":
                self.convolution.append(
                    TransformerConv(
                        input_dim,
                        hidden_dim,
                        heads=4,
                        concat=False,
                        edge_dim=5,
                    )
                )
            self.convolution_batch_norm.append(BatchNorm(hidden_dim))
            if self.residual_connection == "dense":
                input_dim += hidden_dim
            if self.residual_connection == "res":
                input_dim = hidden_dim

        self.linear = nn.Linear(input_dim, output_dim)
        if self.node_pooling == "add":
            self.pooling = global_add_pool
        elif self.node_pooling == "mean":
            self.pooling = global_mean_pool
        elif self.node_pooling == None:
            self.pooling = None
        else:
            raise ValueError(f"Unknown pooling method: {self.node_pooling}")

    def forward(self, data: Data, context_vector) -> Tensor:
        x = F.one_hot(data.x.flatten(), num_classes=118).type(torch.float)
        x = self.initial_embedding_layer(x)
        pos_embedding = self.sinusoidal_positional_encoding(
            data.pos, self.pos_embedding_dim
        )

        for i, conv in enumerate(self.convolution):
            x_conv = torch.cat((x, pos_embedding, context_vector[data.batch]), dim=1)
            x_conv = conv(x_conv, data.edge_index, data.edge_attr.type(torch.float))
            x_conv = self.convolution_batch_norm[i](x_conv)
            x_conv = F.gelu(x_conv)
            if self.residual_connection == "dense":
                x = torch.cat((x, x_conv), dim=1)
            if self.residual_connection == "res":
                x = x + x_conv
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linear(x)
        if self.pooling is None:
            representation = x
        else:
            representation = self.pooling(x, data.batch)

        return representation


class RbfPositionalEncoding(nn.Module):
    def __init__(
        self, out_dim, max_dist: float = 30.0, num_bins: int = 300, gamma: float = 0.5
    ):
        super(RbfPositionalEncoding, self).__init__()
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.gamma = gamma
        self.mlp = MLP((3 * num_bins, 128, out_dim), batch_norm=True)

    def forward(self, positions):
        rbf = self.get_rbf_positional_encoding(positions)
        embedding = self.mlp(rbf)
        return embedding

    def get_rbf_positional_encoding(self, positions):
        device = positions.device
        gamma = self.gamma
        positions = positions.float().unsqueeze(-1)
        mu = torch.linspace(
            -self.max_dist, self.max_dist, self.num_bins, device=device
        ).view(1, 1, -1)
        rbf = torch.exp(-gamma * (positions - mu) ** 2)
        return rbf.view(positions.size(0), -1)


class SinusoidalPositionalEncoding(nn.Module, PositionalEncodingMixin):
    def __init__(self, out_dim: int, num_freq: int = 300):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.num_freq = num_freq
        self.out_dim = out_dim
        self.mlp = MLP((3 * num_freq, 128, out_dim), batch_norm=True)

    def forward(self, positions):
        sinusoidal_encoding = self.sinusoidal_positional_encoding(
            positions, self.num_freq * 3
        )
        return self.mlp(sinusoidal_encoding)


class GNNEncoder1(torch.nn.Module, PositionalEncodingMixin):
    def __init__(
        self,
        pos_embedding_dim: int = 240,
        context_vector_dim: int = 2048,
        types_embedding_dim: int = 128,
        output_dim: int = 1024,
        num_layers: int = 3,
        num_edge_features: int = 5,
        dropout: float = 0.3,
        residual_connection: str = "res",
        node_pooling: str = "add",
        gnn_conv: str = "GAT",
        pos_info: str = "add",
        pos_embedding: str = "rbf",
    ) -> None:
        super().__init__()
        self.pos_embedding_dim = pos_embedding_dim
        self.context_vector_dim = context_vector_dim
        self.types_embedding_dim = types_embedding_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_edge_features = num_edge_features
        self.dropout = dropout
        self.residual_connection = residual_connection
        hidden_dim = types_embedding_dim
        self.node_pooling = node_pooling
        self.gnn_conv = gnn_conv
        self.pos_info = pos_info
        self.pos_embedding = pos_embedding

        # Initialize embeddings
        self.initial_embedding_layer = nn.Linear(118, hidden_dim)
        if pos_embedding == "sinusoidal":
            self.positional_embedding_layer = SinusoidalPositionalEncoding(
                context_vector_dim, num_freq=pos_embedding_dim
            )
        elif pos_embedding == "rbf":
            self.positional_embedding_layer = RbfPositionalEncoding(
                context_vector_dim, num_bins=pos_embedding_dim
            )

        # Convolutional layers
        self.convolution = torch.nn.ModuleList()
        self.convolution_batch_norm = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            input_dim = hidden_dim + self.context_vector_dim
            # input_dim = hidden_dim + self.pos_embedding_dim + self.context_vector_dim
            if self.gnn_conv == "GAT":
                self.convolution.append(
                    GATConv(
                        input_dim,
                        hidden_dim,
                        heads=4,
                        concat=False,
                    )
                )
            elif self.gnn_conv == "GATv1":
                self.convolution.append(
                    GATConv(
                        input_dim,
                        hidden_dim,
                        heads=4,
                        concat=False,
                        edge_dim=5,
                    )
                )
            elif self.gnn_conv == "GATv2":
                self.convolution.append(
                    GATv2Conv(
                        input_dim,
                        hidden_dim,
                        heads=4,
                        concat=False,
                        edge_dim=5,
                    )
                )
            elif self.gnn_conv == "GINE":
                self.convolution.append(
                    GINEConv(
                        nn=nn.Linear(input_dim, hidden_dim),
                        edge_dim=5,
                    )
                )
            elif self.gnn_conv == "TransformerConv":
                self.convolution.append(
                    TransformerConv(
                        input_dim,
                        hidden_dim,
                        heads=4,
                        concat=False,
                        edge_dim=5,
                    )
                )
            self.convolution_batch_norm.append(BatchNorm(hidden_dim))
            if self.residual_connection == "dense":
                input_dim += hidden_dim
            if self.residual_connection == "res":
                input_dim = hidden_dim

        self.linear = nn.Linear(input_dim, output_dim)
        if self.node_pooling == "add":
            self.pooling = global_add_pool
        elif self.node_pooling == "mean":
            self.pooling = global_mean_pool
        elif self.node_pooling == None:
            self.pooling = None
        else:
            raise ValueError(f"Unknown pooling method: {self.node_pooling}")

    def forward(self, data: Data, context_vector) -> Tensor:
        x = F.one_hot(data.x.flatten(), num_classes=118).type(torch.float)
        x = self.initial_embedding_layer(x)
        pos_embedding = self.positional_embedding_layer(data.pos)
        if self.pos_info == "add":
            context = context_vector[data.batch] + pos_embedding
        elif self.pos_info == "mul":
            context = context_vector[data.batch] * pos_embedding

        for i, conv in enumerate(self.convolution):
            x_conv = torch.cat((x, context), dim=1)
            x_conv = conv(x_conv, data.edge_index, data.edge_attr.type(torch.float))
            x_conv = self.convolution_batch_norm[i](x_conv)
            x_conv = F.gelu(x_conv)
            if self.residual_connection == "dense":
                x = torch.cat((x, x_conv), dim=1)
            if self.residual_connection == "res":
                x = x + x_conv
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linear(x)
        if self.pooling is None:
            representation = x
        else:
            representation = self.pooling(x, data.batch)

        return representation
