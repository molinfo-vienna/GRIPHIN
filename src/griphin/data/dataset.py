import os
from tqdm import tqdm

import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import ToUndirected

import CDPL.Grid as Grid
import CDPL.Math as Math
import CDPL.Chem as Chem


class GrailLigandData(InMemoryDataset):
    def __init__(
        self, root, bbox_size, transform=None, pre_transform=None, pre_filter=None
    ):
        self.root = root
        self.bbox_size = bbox_size
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [""]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        raw_folder = os.path.join(self.root, "raw")
        affinities = pd.read_csv(
            os.path.join(self.root, "binding_affinities.csv"),
            header=None,
            names=["code", "affinity"],
        )
        affinity_dict = dict(
            zip(affinities["code"].str.lower(), affinities["affinity"])
        )

        data_list = []
        for target in tqdm(os.listdir((raw_folder))):
            target_path = os.path.join(raw_folder, target)
            if target not in affinity_dict.keys():
                print(f"Binding affinity not found for target {target}. Skipping.")
                continue
            affinity = float(affinity_dict[target])
            data = self.process_data_point(
                target_path, bbox_size=self.bbox_size, affinity=affinity
            )
            if data is not None:
                data_list.append(data)

        data_list = [data for data in data_list if data is not None]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    @staticmethod
    def process_data_point(target_path, bbox_size, affinity):
        try:
            ligand_path = os.path.join(target_path, "ligand.sdf")
            grail_path = os.path.join(target_path, "map.cdf")
            code = os.path.basename(target_path)

            grid_set = Grid.DRegularGridSet()
            grid_set_reader = Grid.FileCDFDRegularGridSetReader(grail_path)
            grid_set_reader.read(grid_set)
            vals, center = GrailLigandData.grid_set_to_padded_tensor(
                grid_set, bbox_size
            )

            mol_reader = Chem.MoleculeReader(ligand_path)
            mol = Chem.BasicMolecule()
            mol_reader.read(mol)
            pos, types, edge_index, edge_attr = GrailLigandData.ligand_to_tensor(mol)
            pos = pos - center  # Center the ligand coordinates

            # Create a PyG Data object
            data = Data(
                x=types,
                pos=pos,
                edge_index=edge_index,
                edge_attr=edge_attr,
                vals=vals,
                y=torch.tensor([affinity], dtype=torch.float),
                code=code,
            )
            transform = ToUndirected()
            data = transform(data)
            return data

        except Exception as e:
            print(f"Error processing {target_path}: {e}")
            return None

    @staticmethod
    def grid_set_to_padded_tensor(grid_set, box_range):
        grid = grid_set[0]
        num_i = grid.getSize1()
        num_j = grid.getSize2()
        num_k = grid.getSize3()

        center_i = num_i // 2
        center_j = num_j // 2
        center_k = num_k // 2

        channels = 11
        box_size = 2 * box_range
        vals = torch.zeros(
            (box_size, box_size, box_size, channels), dtype=torch.float32
        )
        vals[..., 9] = 1.0  # The pocket density channel is padded with ones

        for channel in range(channels):
            grid_data = torch.tensor(
                grid_set[channel].getData().toArray()
            )  # shape: (num_i, num_j, num_k)

            # Compute valid ranges (clipping to avoid going out of bounds)
            i_start = max(center_i - box_range, 0)
            j_start = max(center_j - box_range, 0)
            k_start = max(center_k - box_range, 0)

            i_end = min(center_i + box_range, num_i)
            j_end = min(center_j + box_range, num_j)
            k_end = min(center_k + box_range, num_k)

            # Compute where to insert this slice into the result volume
            i_offset = i_start - (center_i - box_range)
            j_offset = j_start - (center_j - box_range)
            k_offset = k_start - (center_k - box_range)

            # Extract the valid subarray
            subgrid = grid_data[i_start:i_end, j_start:j_end, k_start:k_end]

            # Place it into the result tensor
            vals[
                i_offset : i_offset + subgrid.shape[0],
                j_offset : j_offset + subgrid.shape[1],
                k_offset : k_offset + subgrid.shape[2],
                channel,
            ] = subgrid

        coord = Math.Vector3D()
        grid.getCoordinates(center_i, center_j, center_k, coord)
        min_coord = coord.toArray()
        grid.getCoordinates(
            center_i - 1,
            center_j - 1,
            center_k - 1,
            coord,
        )
        max_coord = coord.toArray()

        center = (min_coord + max_coord) / 2

        vals = vals.to(torch.float16)

        return torch.unsqueeze(vals, dim=0), torch.tensor(center, dtype=torch.float32)

    @staticmethod
    def ligand_to_tensor(mol, conf_idx=None):
        num_atoms = mol.getNumAtoms()
        num_bonds = mol.getNumBonds()
        pos = torch.zeros((num_atoms, 3), dtype=torch.float32)
        types = torch.zeros((num_atoms, 1), dtype=torch.long)
        edge_index = torch.zeros((2, num_bonds), dtype=torch.long)
        edge_attr = torch.zeros((num_bonds, 5), dtype=torch.long)

        Chem.calcBasicProperties(mol, False)

        for (
            atom
        ) in (
            mol.atoms
        ):  # iterate of structure data entries consisting of a header line and the actual data
            idx = atom.getIndex()
            if conf_idx is not None:
                pos[idx] = torch.tensor(
                    Chem.getConformer3DCoordinates(atom, conf_idx).toArray(),
                    dtype=torch.float32,
                )
            else:
                pos[idx] = torch.tensor(
                    Chem.get3DCoordinates(atom).toArray(), dtype=torch.float32
                )
            types[idx] = Chem.getType(atom)

        for i, bond in enumerate(mol.bonds):
            edge_index[0, i] = bond.getBegin().getIndex()
            edge_index[1, i] = bond.getEnd().getIndex()
            edge_attr[i, Chem.getOrder(bond) - 1] = 1
            edge_attr[i, 3] = Chem.getAromaticityFlag(bond)
            edge_attr[i, 4] = Chem.getRingFlag(bond)

        return pos, types, edge_index, edge_attr


# def plot_3d_molecule_with_voxels(
#     pos,
#     types,
#     edge_index,
#     edge_attr,
#     voxelbox,
#     voxel_origin=(-7.5, -7.5, -7.5),
#     voxel_extent=15.0,
#     channel=0,
#     voxel_threshold=0.1,
#     show_labels=True,
# ):
#     """
#     Plots a 3D molecular structure with a voxel grid overlay.

#     Args:
#         pos (Tensor): (num_atoms, 3)
#         types (Tensor): (num_atoms, 1) atomic numbers
#         edge_index (Tensor): (2, num_bonds)
#         edge_attr (Tensor): (num_bonds, D)
#         voxelbox (Tensor): (1, X, Y, Z, C)
#         voxel_origin (tuple): Origin of the voxel grid (default: (-5, -5, -5))
#         voxel_extent (float): Total length of the cube along one axis (default: 10)
#         channel (int): Channel index to visualize from the voxel grid
#         voxel_threshold (float): Minimum value for a voxel to be plotted
#         show_labels (bool): Whether to annotate atoms
#     """

#     ELEMENTS = [
#         "",
#         "H",
#         "He",
#         "Li",
#         "Be",
#         "B",
#         "C",
#         "N",
#         "O",
#         "F",
#         "Ne",
#         "Na",
#         "Mg",
#         "Al",
#         "Si",
#         "P",
#         "S",
#         "Cl",
#         "Ar",
#         "K",
#         "Ca",
#     ]
#     ATOM_COLORS = {
#         1: "white",
#         6: "black",
#         7: "blue",
#         8: "red",
#         9: "green",
#         15: "orange",
#         16: "yellow",
#         17: "lime",
#     }

#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection="3d")

#     # Plot molecule
#     pos_np = pos.detach().cpu().numpy()
#     types_np = types.view(-1).detach().cpu().numpy()
#     edge_index_np = edge_index.detach().cpu().numpy()

#     for i, atom_type in enumerate(types_np):
#         color = ATOM_COLORS.get(atom_type, "gray")
#         ax.scatter(*pos_np[i], color=color, s=100, edgecolors="k")
#         if show_labels:
#             label = ELEMENTS[atom_type] if atom_type < len(ELEMENTS) else str(atom_type)
#             ax.text(*pos_np[i], label, fontsize=8, ha="center", va="center")

#     for i in range(edge_index_np.shape[1]):
#         src, dst = edge_index_np[:, i]
#         xs, ys, zs = zip(pos_np[src], pos_np[dst])
#         ax.plot(xs, ys, zs, color="gray", linewidth=1.5)

#     # Plot voxel box
#     vox = (
#         voxelbox[0, ..., channel].detach().cpu().to(torch.float32).numpy() / 255.0
#     )  # shape: (X, Y, Z)
#     X, Y, Z = vox.shape
#     grid_lin = torch.linspace(voxel_origin[0], voxel_origin[0] + voxel_extent, steps=X)
#     xx, yy, zz = torch.meshgrid(grid_lin, grid_lin, grid_lin, indexing="ij")

#     # Flatten voxel data and select high-value voxels
#     xx = xx.numpy().flatten()
#     yy = yy.numpy().flatten()
#     zz = zz.numpy().flatten()
#     vals = vox.flatten()

#     mask = vals > voxel_threshold
#     ax.scatter(xx[mask], yy[mask], zz[mask], c=vals[mask], cmap="Reds", alpha=0.3, s=20)

#     ax.set_box_aspect([1, 1, 1])
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     plt.tight_layout()
#     plt.savefig("test.png")
