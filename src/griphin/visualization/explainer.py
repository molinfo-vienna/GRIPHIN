from captum.attr import IntegratedGradients
import torch
from torch_geometric.data import Data
import numpy as np


class Explainer:
    def __init__(self, model, device="cuda:0"):
        self.device = torch.device(device)
        self.model = model
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.set_representation_output(False)

    @staticmethod
    def model_forward(vals, x, edge_index, edge_attr, pos, batch, model):
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            batch=batch,
            vals=vals,
        )
        return model(data)

    def calculate_attributions(self, data_point):
        print(data_point.code)
        # Prepare data point
        num_nodes = data_point.x.shape[0]
        data_point = data_point.to(self.device)
        data_point.pos = data_point.pos.float()
        data_point.batch = torch.zeros(num_nodes, dtype=torch.long, device=self.device)

        # Prepare Captum inputs
        input_values = data_point.vals.float().requires_grad_(
            True
        )  # Ensure input is differentiable
        baseline = torch.zeros_like(data_point.vals)  # Zero baseline
        additional_forward_args = (
            data_point.x,
            data_point.edge_index,
            data_point.edge_attr,
            data_point.pos,
            data_point.batch,
            self.model,
        )

        # Compute attributions using Integrated Gradients
        ig = IntegratedGradients(self.model_forward)
        attributions = ig.attribute(
            input_values,
            baseline,
            target=None,
            additional_forward_args=additional_forward_args,
            internal_batch_size=1,
        )
        attributions = attributions.cpu().detach().numpy()
        attributions_normalized = attributions / np.abs(attributions).max()
        return attributions_normalized
