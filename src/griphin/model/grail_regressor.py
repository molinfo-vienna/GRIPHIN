from typing import Union

from lightning import LightningModule
import torch
from torch import Tensor
from torch.optim import Optimizer, AdamW
from torch_geometric.data import Data
from torch_geometric.nn import MLP

from .encoder import (
    GNNEncoder,
    GNNEncoder1,
    GNNEncoder2,
    Voxel3DCNN,
)


class GrailRegressor(LightningModule):
    def __init__(self, **params) -> None:
        super(GrailRegressor, self).__init__()
        self.save_hyperparameters()
        self.hparams.setdefault("CNN_encoder", "Voxel3DCNN")
        self.hparams.setdefault("node_pooling", "add")
        self.hparams.setdefault("exclude_hydrogens", False)
        self.hparams.setdefault("MLP_hidden_layers", 2)
        self.hparams.setdefault("data_split", "random")
        self.hparams.setdefault("data_set", "PDBbind_2024")
        self.hparams.setdefault("MLP_pyramidal", False)
        self.hparams.setdefault("context", "before")
        self.hparams.setdefault("GNN_conv", "GAT")
        self.hparams.setdefault("pos_info", "add")
        self.hparams.setdefault("pos_embedding", "rbf")

        if self.hparams.context is not None:
            if self.hparams.CNN_encoder == "Voxel3DCNN":
                self.grail_encoder = Voxel3DCNN(
                    first_layer_channels=self.hparams.CNN_first_layer_channels,
                    num_layers=self.hparams.CNN_layers,
                    dropout=self.hparams.dropout,
                    use_exclusion_channel=self.hparams.use_exclusion_channels,
                )
        if self.hparams.context == "before":
            gnn_context_input_dim = self.grail_encoder.out_dim
        else:
            gnn_context_input_dim = 0

        if self.hparams.GNN_encoder == "GNNEncoder":
            self.ligand_encoder = GNNEncoder(
                pos_embedding_dim=self.hparams.pos_embedding_dim,
                context_vector_dim=gnn_context_input_dim,
                types_embedding_dim=self.hparams.types_embedding_dim,
                output_dim=self.hparams.GNN_output_dim,
                num_layers=self.hparams.GNN_layers,
                dropout=self.hparams.GNN_dropout,
                node_pooling=self.hparams.node_pooling,
                gnn_conv=self.hparams.GNN_conv,
            )
        if self.hparams.GNN_encoder == "GNNEncoder1":
            self.ligand_encoder = GNNEncoder1(
                pos_embedding_dim=self.hparams.pos_embedding_dim,
                context_vector_dim=gnn_context_input_dim,
                types_embedding_dim=self.hparams.types_embedding_dim,
                output_dim=self.hparams.GNN_output_dim,
                num_layers=self.hparams.GNN_layers,
                dropout=self.hparams.GNN_dropout,
                node_pooling=self.hparams.node_pooling,
                gnn_conv=self.hparams.GNN_conv,
                pos_info=self.hparams.pos_info,
                pos_embedding=self.hparams.pos_embedding,
            )
        if self.hparams.GNN_encoder == "GNNEncoder2":
            self.ligand_encoder = GNNEncoder2(
                pos_embedding_dim=self.hparams.pos_embedding_dim,
                context_vector_dim=gnn_context_input_dim,
                types_embedding_dim=self.hparams.types_embedding_dim,
                output_dim=self.hparams.GNN_output_dim,
                num_layers=self.hparams.GNN_layers,
                dropout=self.hparams.GNN_dropout,
                node_pooling=self.hparams.node_pooling,
                gnn_conv=self.hparams.GNN_conv,
            )
        if self.hparams.MLP_projector:
            if self.hparams.context == "after":
                dims = [self.ligand_encoder.output_dim + self.grail_encoder.out_dim]
            else:
                dims = [self.ligand_encoder.output_dim]

            for _ in range(self.hparams.MLP_hidden_layers):
                if self.hparams.MLP_pyramidal:
                    dims.append(int(dims[-1] // 2))
                else:
                    dims.append(self.hparams.MLP_hidden_dim)
            dims.append(1)
            self.mlp = MLP(
                dims,
                dropout=self.hparams.dropout,
                batch_norm=True,
            )
        self.loss = torch.nn.MSELoss()
        self.representation_output = None

    def forward(self, data: Data) -> Tensor:
        if self.hparams.context is not None:
            grail_representation = self.grail_encoder(data)

            if self.representation_output == "grail":
                return grail_representation

        if self.hparams.context == "before":
            ligand_representation = self.ligand_encoder(data, grail_representation)
        else:
            ligand_representation = self.ligand_encoder(
                data, torch.empty((self.hparams.batch_size, 0), device=self.device)
            )

        if self.representation_output == "interaction":
            return ligand_representation

        if self.hparams.MLP_projector:
            if self.hparams.context == "after":
                ligand_representation = torch.cat(
                    (ligand_representation, grail_representation), dim=1
                )
            y_hat = self.mlp(ligand_representation).flatten()
            return y_hat
        else:
            return ligand_representation.flatten()

    def flip_sign_and_voxel(self, data):
        # No cloning
        axis_map = {0: 1, 1: 2, 2: 3}  # pos axis -> voxel axis

        for axis in [0, 1, 2]:
            if torch.rand(1) < 0.5:
                data.pos[:, axis] *= -1
                data.vals = torch.flip(data.vals, dims=[axis_map[axis]])

        return data

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict]]:
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        return [optimizer]

    def shared_step(self, batch: Data, batch_idx: int) -> Tensor:
        y_hat = self(batch)
        y = batch.y

        return self.loss(y_hat, y)

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        """Training step and logging"""
        batch = self.flip_sign_and_voxel(batch)
        loss = self.shared_step(batch, batch_idx)

        self.log(
            "train/train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=len(batch),
            reduce_fx="mean",
        )

        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> Tensor:
        loss = self.shared_step(batch, batch_idx)

        self.log(
            "val/val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )

        return loss

    def predict_step(
        self, batch: Data, batch_idx: int = 0
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        return self(batch)

    def set_representation_output(self, representation_output: str = None) -> None:
        self.representation_output = representation_output
