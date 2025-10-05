import os

import numpy as np
import py3Dmol
import ipywidgets
from ipywidgets import interact, fixed

import CDPL.Grid as Grid
import CDPL.Math as Math


class Py3DmolViewer:
    # Convert a CDPL grid to a 3D numpy array and coordinates
    @staticmethod
    def get_nd_array(grid):
        num_i = grid.getSize1()
        num_j = grid.getSize2()
        num_k = grid.getSize3()
        shape = (num_i, num_j, num_k)
        # print(f"Grid size: {num_i} x {num_j} x {num_k}")

        vals = np.ndarray((num_i, num_j, num_k), dtype=np.float32)
        coords = np.ndarray((num_i, num_j, num_k, 3), dtype=np.float32)

        for i in range(num_i):
            for j in range(num_j):
                for k in range(num_k):
                    vals[i, j, k] = grid.getElement(i, j, k)
                    coord = Math.Vector3D()
                    grid.getCoordinates(i, j, k, coord)
                    coords[i, j, k, :] = coord.toArray()

        return vals, coords, shape

    @staticmethod
    def grid_to_cube(grid, coords, grid_shape):
        """
        Convert a 3D grid and its coordinates to a Cube file format string.
        """
        bohr = 0.52917721092  # Bohr radius in angstroms
        angstrom = 1.0  # Angstroms
        scaling_factor = angstrom / bohr
        num_i, num_j, num_k = grid_shape
        origin = coords[0, 0, 0] * scaling_factor  # Origin of the grid
        voxel_size = (
            coords[1, 1, 1] - coords[0, 0, 0]
        ) * scaling_factor  # Voxel size (assumes uniform grid)

        # Header for Cube file format
        cube_str = "Py3Dmol volumetric data\n"
        cube_str += "Generated from voxel grid\n"
        cube_str += f"0 {origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}\n"  # No atoms, origin of the grid
        cube_str += f"{num_i} {voxel_size[0]:.6f} 0.0 0.0\n"  # X-dimension
        cube_str += f"{num_j} 0.0 {voxel_size[1]:.6f} 0.0\n"  # Y-dimension
        cube_str += f"{num_k} 0.0 0.0 {voxel_size[2]:.6f}\n"  # Z-dimension

        # Add the grid data
        for i in range(num_i):
            for j in range(num_j):
                for k in range(num_k):
                    value = grid[i, j, k]
                    string = f"{value:.6f} "
                    cube_str += string
                    if (k + 1) % 6 == 0:  # Add a newline every 6 values
                        cube_str += "\n"
                cube_str += "\n"

        return cube_str

    def fit_padded_tensor_to_grid(self, tensor):
        num_i, num_j, num_k = self.grid_shape
        box_range = tensor.shape[1] // 2

        center_i = num_i // 2
        center_j = num_j // 2
        center_k = num_k // 2

        channels = 10
        grid = np.zeros((num_i, num_j, num_k, channels))

        for channel in range(channels):
            for i_, i in enumerate(range(center_i - box_range, center_i + box_range)):
                for j_, j in enumerate(
                    range(center_j - box_range, center_j + box_range)
                ):
                    for k_, k in enumerate(
                        range(center_k - box_range, center_k + box_range)
                    ):
                        if (
                            i < 0
                            or j < 0
                            or k < 0
                            or i >= num_i
                            or j >= num_j
                            or k >= num_k
                        ):
                            continue

                        grid[i, j, k, channel] = tensor[i_, j_, k_, channel]

        return grid

    def retrieve_files(self, data_point, data_root):
        data_point.code
        # sub_dirs = ["1981-2000", "2001-2010", "2011-2020", "2021-2023"]
        # PDB_BIND_ROOT = "/data/sharedXL/projects/Daniel/FlowMol/data/PDBbind_2024"
        # for sub_dir in sub_dirs:
        #     path = os.path.join(PDB_BIND_ROOT, sub_dir, data_point.code)
        #     if os.path.exists(path):
        #         pdb_file = os.path.join(path, f"{data_point.code}_protein.pdb")

        pdb_file = os.path.join(data_root, "raw", data_point.code, "protein.pdb")
        sdf_file = os.path.join(data_root, "raw", data_point.code, "ligand.sdf")
        grail_file = os.path.join(data_root, "raw", data_point.code, "map.cdf")

        # Load the PDB structure (e.g., protein)
        with open(pdb_file, "r") as f:
            pdb_data = f.read()

        # Load the SDF ligand
        with open(sdf_file, "r") as f:
            sdf_data = f.read()

        in_grid_set = Grid.DRegularGridSet()
        grid_set_reader = Grid.FileCDFDRegularGridSetReader(grail_file)
        grid_set_reader.read(in_grid_set)
        channels = 11
        grid_list = []

        for channel in range(channels):
            grail_grid, coords, grid_shape = Py3DmolViewer.get_nd_array(
                in_grid_set[channel]
            )
            grid_list.append(grail_grid)
        self.coords = coords
        self.grid_shape = grid_shape

        self.pdb_data = pdb_data
        self.sdf_data = sdf_data
        self.grid_list = grid_list

    def add_protein(self, pdb_data):
        self.view.addModel(pdb_data, "pdb")

    def toggle_protein_visibility(self, visible=True):
        if visible:
            self.view.setStyle(
                {"model": 0, "hetflag": False},
                {
                    "cartoon": {"color": "white"},
                    "stick": {"colorscheme": "element", "radius": 0.2},
                },
            )
        else:
            self.view.setStyle({"model": 0}, {})

    def toggle_ligand_visibility(self, visible=True):
        if visible:
            self.view.setStyle(
                {"model": 1}, {"stick": {"colorscheme": "cyanCarbon"}}
            )  # Stick style for ligand
        else:
            self.view.setStyle({"model": 1}, {})

    def add_ligand(self, sdf_data):
        self.view.addModel(sdf_data, "sdf")  # Add the ligand

    def add_grid_data(self, grail_grid, threshold=0.9, color="red"):
        grail_data = Py3DmolViewer.grid_to_cube(
            grail_grid, self.coords, self.grid_shape
        )

        # Add the volumetric data to the viewer
        self.view.addVolumetricData(
            grail_data,
            "cube",
            {
                "isoval": threshold,  # Isosurface threshold
                "color": color,  # Color of the isosurface
                "opacity": 0.9,  # Transparency of the isosurface
            },
        )

    def create_initial_view(self, zoom_level=0.7):
        self.view = py3Dmol.view(width=self.width, height=self.height)
        self.add_protein(self.pdb_data)
        self.toggle_protein_visibility(True)  # Ensure protein is visible
        self.add_ligand(self.sdf_data)
        self.toggle_ligand_visibility(True)
        self.view.zoomTo({"model": 1})  # Focus on ligand
        self.view.zoom(zoom_level)

    def add_channels(
        self,
        grid_channels=[],
        threshold=0.5,
        threshold_attr=0.5,
        attribution_mode=0,
        intersect_ligand=False,
    ):
        for channel in grid_channels:
            grid = self.grid_list[channel]
            attribution = self.attribution_data[:, :, :, channel]
            if channel == 0:
                grid = grid / grid.max()
            if intersect_ligand:
                grid = grid * self.grid_list[-1]
            if attribution_mode == 0:
                self.add_grid_data(grid, threshold=threshold, color="blue")
            # elif attribution_mode == 1:
            #     self.add_grid_data(grid, threshold=threshold, color="blue")
            #     if intersect_ligand:
            #         attribution = attribution * self.grid_list[-1]
            #     self.add_grid_data(attribution, threshold=threshold_attr, color="red")
            # elif attribution_mode == 2:
            #     self.add_grid_data(
            #         grid * attribution, threshold=threshold_attr, color="purple"
            #     )
            elif attribution_mode == 1:
                if intersect_ligand:
                    attribution = attribution * self.grid_list[-1]
                if threshold_attr >= 0:
                    self.add_grid_data(
                        attribution, threshold=threshold_attr, color="red"
                    )
                else:
                    self.add_grid_data(
                        -attribution, threshold=-threshold_attr, color="yellow"
                    )

    def remove_surfaces(self):
        self.view.removeAllShapes()

    def set_attribution_data(self, attribution_data):
        self.attribution_data = attribution_data

    def __init__(self, width=600, height=450):
        self.width = width
        self.height = height
        self.view = None
        self.pdb_data = None
        self.sdf_data = None
        self.grid_list = []
        self.coords = None
        self.grid_shape = None
        self.checkboxes = []


class InteractionWrapper:
    def __init__(self):
        # Channels:
        self.channels = {
            "H-H": 0,
            "AR-AR": 1,
            "AR-PI": 2,
            "NI-PI": 3,
            "PI-AR": 4,
            "PI-NI": 5,
            "HBD-HBA": 6,
            "HBA-HBD": 7,
            "XBD-XBA": 8,
            "ENV": 9,
        }

        self.attribution_modes = {
            "No attributions": 0,
            # "Seperate attributions": 1,
            # "Intersecting attributions": 2,
            "Attributions": 1,
        }

        # widgets:
        # slider = ipywidgets.IntSlider(min=0,max=9, step=1, value=0, description='Channel', continous_update=False)
        self.dropdown1 = ipywidgets.Dropdown(
            options=list(self.channels.keys()), value="H-H", description="Channel:"
        )

        self.dropdown2 = ipywidgets.Dropdown(
            options=list(self.attribution_modes.keys()),
            value="No attributions",
            description="Attributions:",
        )

        self.checkbox3 = ipywidgets.Checkbox(
            value=False,
            description="Intersect with Ligand",
            disabled=False,
            indent=False,
        )

        self.checkbox4 = ipywidgets.Checkbox(
            value=True, description="Protein visible", disabled=False, indent=False
        )

        self.checkbox5 = ipywidgets.Checkbox(
            value=True, description="Ligand visible", disabled=False, indent=False
        )

        self.checkbox6 = ipywidgets.Checkbox(
            value=False, description="Channels visible", disabled=False, indent=False
        )

        self.threshold_slider = ipywidgets.FloatSlider(
            value=0.9,  # Default value
            min=0.0,  # Minimum value
            max=1.0,  # Maximum value
            step=0.05,  # Step size
            description="GRAIL Threshold:",
        )

        self.attribution_slider = ipywidgets.FloatSlider(
            value=0.1,  # Default value
            min=-1.0,  # Minimum value
            max=1.0,  # Maximum value
            step=0.05,  # Step size
            description="Attribution Threshold:",
        )

    def interactive_view(
        self,
        viewer,
        channel,
        attribution_mode,
        intersect_ligand=False,
        protein_visible=True,
        ligand_visible=True,
        channels_visible=False,
        threshold=0.5,
        threshold_attr=0.5,
    ):
        idx = self.channels[channel]
        attribution_mode = self.attribution_modes[attribution_mode]
        viewer.remove_surfaces()
        if channels_visible:
            viewer.add_channels(
                grid_channels=[idx],
                threshold=threshold,
                threshold_attr=threshold_attr,
                attribution_mode=attribution_mode,
                intersect_ligand=intersect_ligand,
            )
        viewer.toggle_protein_visibility(visible=protein_visible)
        viewer.toggle_ligand_visibility(visible=ligand_visible)
        viewer.view.update()

    def interaction(self, viewer):  # , threshold=0.9, threshold_attr=0.1):
        interact(
            self.interactive_view,
            viewer=fixed(viewer),
            channel=self.dropdown1,
            attribution_mode=self.dropdown2,
            intersect_ligand=self.checkbox3,
            protein_visible=self.checkbox4,
            ligand_visible=self.checkbox5,
            channels_visible=self.checkbox6,
            threshold=self.threshold_slider,
            threshold_attr=self.attribution_slider,
        )
