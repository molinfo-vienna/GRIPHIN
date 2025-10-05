import os

from lightning import LightningDataModule
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from .dataset import GrailLigandData


class GrailDataModule(LightningDataModule):
    def __init__(
        self,
        training_data_dir: str,
        batch_size: int = None,
        bbox_size: int = 15,
        split: str = "random",
    ) -> None:
        super(GrailDataModule, self).__init__()
        self.training_data_dir = training_data_dir
        self.batch_size = batch_size
        self.num_workers = 4
        self.bbox_size = bbox_size
        self.split = split

    def setup(self, stage: str = "fit") -> None:
        if stage == "fit":
            self.full_data = GrailLigandData(
                self.training_data_dir, self.bbox_size, transform=None
            )
            if self.split == "random":
                seed = 42  # or any fixed number
                generator = torch.Generator().manual_seed(seed)
                perm = torch.randperm(len(self.full_data), generator=generator)
                self.full_data = self.full_data[perm]

                self.training_data = self.full_data[: int(0.8 * len(self.full_data))]
                self.validation_data = self.full_data[
                    int(0.8 * len(self.full_data)) : int(0.9 * len(self.full_data))
                ]
                self.test_data = self.full_data[int(0.9 * len(self.full_data)) :]

                print(f"Number of training graphs: {len(self.training_data)}")
                print(f"Number of validation graphs: {len(self.validation_data)}")
                print(f"Number of test graphs: {len(self.test_data)}")
            elif self.split == "random-train-val":
                seed = 42  # or any fixed number
                generator = torch.Generator().manual_seed(seed)
                perm = torch.randperm(len(self.full_data), generator=generator)
                self.full_data = self.full_data[perm]
                self.training_data = self.full_data[: int(0.95 * len(self.full_data))]
                self.validation_data = self.full_data[int(0.95 * len(self.full_data)) :]

                print(f"Number of training graphs: {len(self.training_data)}")
                print(f"Number of validation graphs: {len(self.validation_data)}")
            elif self.split == "time":
                train = pd.read_csv(
                    os.path.join(
                        self.training_data_dir,
                        "time-split",
                        "timesplit_no_lig_overlap_train.txt",
                    ),
                    header=None,
                )[0].tolist()
                val = pd.read_csv(
                    os.path.join(
                        self.training_data_dir,
                        "time-split",
                        "timesplit_no_lig_overlap_val.txt",
                    ),
                    header=None,
                )[0].tolist()
                test = pd.read_csv(
                    os.path.join(
                        self.training_data_dir, "time-split", "timesplit_test.txt"
                    ),
                    header=None,
                )[0].tolist()
                codes = self.full_data.code
                train_mask = torch.tensor([code in set(train) for code in codes])
                val_mask = torch.tensor([code in set(val) for code in codes])
                test_mask = torch.tensor([code in set(test) for code in codes])
                self.training_data = self.full_data[train_mask]
                self.validation_data = self.full_data[val_mask]
                self.test_data = self.full_data[test_mask]
                print(f"Number of training graphs: {len(self.training_data)}")
                print(f"Number of validation graphs: {len(self.validation_data)}")
                print(f"Number of test graphs: {len(self.test_data)}")
            elif self.split == "core":
                seed = 42  # or any fixed number
                generator = torch.Generator().manual_seed(seed)
                perm = torch.randperm(len(self.full_data), generator=generator)
                self.full_data = self.full_data[perm]
                core_set = pd.read_csv(
                    os.path.join(
                        self.training_data_dir, "core-set", "INDEX_core_name.2016"
                    ),
                    header=None,
                    delimiter="  ",
                    engine="python",
                )[0].tolist()
                codes = self.full_data.code
                test_mask = torch.tensor([code in set(core_set) for code in codes])
                self.test_data = self.full_data[test_mask]
                other_data = self.full_data[~test_mask]
                self.training_data = other_data[: int(0.9 * len(other_data))]
                self.validation_data = other_data[int(0.9 * len(other_data)) :]
                print(f"Number of training graphs: {len(self.training_data)}")
                print(f"Number of validation graphs: {len(self.validation_data)}")
                print(f"Number of test graphs: {len(self.test_data)}")
            elif self.split == "core-v2020.R1":
                seed = 42  # or any fixed number
                generator = torch.Generator().manual_seed(seed)
                perm = torch.randperm(len(self.full_data), generator=generator)
                self.full_data = self.full_data[perm]
                core_set = pd.read_csv(
                    os.path.join(
                        self.training_data_dir, "core-set", "INDEX_core_name.2016"
                    ),
                    header=None,
                    delimiter="  ",
                    engine="python",
                )[0].tolist()

                sub_set_2020 = self.parse_index_file(
                    os.path.join(
                        self.training_data_dir,
                        "v2020.R1",
                        "index",
                        "INDEX_general_PL.2020R1.lst",
                    )
                )["PDB_code"].tolist()
                codes = self.full_data.code
                subset_mask = torch.tensor(
                    [code in set(sub_set_2020) for code in codes]
                )
                test_mask = torch.tensor([code in set(core_set) for code in codes])
                test_mask = test_mask & subset_mask
                other_mask = (~test_mask) & subset_mask
                self.test_data = self.full_data[test_mask]
                other_data = self.full_data[other_mask]
                self.training_data = other_data[: int(0.9 * len(other_data))]
                self.validation_data = other_data[int(0.9 * len(other_data)) :]
                print(f"Number of training graphs: {len(self.training_data)}")
                print(f"Number of validation graphs: {len(self.validation_data)}")
                print(f"Number of test graphs: {len(self.test_data)}")
            elif self.split == "leak-proof":
                path = os.path.join(
                    self.training_data_dir, "lp-split", "LP_PDBBind.csv"
                )
                df = pd.read_csv(path, index_col=0)
                train_codes = df[df["new_split"] == "train"].index.tolist()
                val_codes = df[df["new_split"] == "val"].index.tolist()
                test_codes = df[df["new_split"] == "test"].index.tolist()
                codes = self.full_data.code
                train_mask = torch.tensor([code in set(train_codes) for code in codes])
                val_mask = torch.tensor([code in set(val_codes) for code in codes])
                test_mask = torch.tensor([code in set(test_codes) for code in codes])
                self.training_data = self.full_data[train_mask]
                self.validation_data = self.full_data[val_mask]
                self.test_data = self.full_data[test_mask]
                print(f"Number of training graphs: {len(self.training_data)}")
                print(f"Number of validation graphs: {len(self.validation_data)}")
                print(f"Number of test graphs: {len(self.test_data)}")
            elif self.split is None:
                print("No data splitting is applied.")
                print(f"Number of graphs in the dataset: {len(self.full_data)}")

            else:
                raise ValueError(f"Unknown split type: {self.split}")

    def parse_index_file(self, path):
        # Initialize an empty list to store parsed rows
        rows = []

        # Open the file and process it line by line
        with open(path, "r") as file:
            for line in file:
                # Skip header lines
                if line.startswith("#") or line.strip() == "":
                    continue

                # Split the line into parts
                parts = line.split()

                # Handle cases where the line has extra or missing fields
                if len(parts) >= 6:  # Ensure there are at least 6 fields
                    pdb_code = parts[0]
                    resolution = parts[1]
                    release_year = parts[2]
                    binding_data = parts[3]
                    reference = parts[4]
                    ligand_name = parts[5].strip(
                        "()"
                    )  # Remove parentheses from ligand name
                    comments = (
                        " ".join(parts[6:]) if len(parts) > 6 else None
                    )  # Combine remaining parts as comments

                    # Append the parsed row to the list
                    rows.append(
                        [
                            pdb_code,
                            resolution,
                            release_year,
                            binding_data,
                            reference,
                            ligand_name,
                            comments,
                        ]
                    )

        # Create a DataFrame from the parsed rows
        columns = [
            "PDB_code",
            "Resolution",
            "Release_year",
            "Binding_data",
            "Reference",
            "Ligand_name",
            "Comments",
        ]
        data = pd.DataFrame(rows, columns=columns)
        return data

    def train_dataloader(self, shuffle_data=True) -> DataLoader:
        if self.batch_size is None:
            return DataLoader(
                self.training_data,
                batch_size=len(self.training_data),
                shuffle=shuffle_data,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=True,
            )
        else:
            return DataLoader(
                self.training_data,
                batch_size=self.batch_size,
                shuffle=shuffle_data,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=True,
            )

    def val_dataloader(self) -> DataLoader:
        if self.batch_size is None:
            return DataLoader(
                self.validation_data,
                batch_size=len(self.validation_data),
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                persistent_workers=True,
            )
        else:
            return DataLoader(
                self.validation_data,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                persistent_workers=True,
            )

    def test_dataloader(self) -> DataLoader:
        if self.batch_size is None:
            return DataLoader(
                self.test_data,
                batch_size=len(self.test_data),
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                persistent_workers=True,
            )
        else:
            return DataLoader(
                self.test_data,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                persistent_workers=True,
            )

    def full_dataloader(self) -> DataLoader:
        if self.batch_size is None:
            return DataLoader(
                self.full_data,
                batch_size=len(self.full_data),
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                persistent_workers=True,
            )
        else:
            return DataLoader(
                self.full_data,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                persistent_workers=True,
            )
