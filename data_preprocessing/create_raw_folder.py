import os, sys
import argparse
from tqdm import tqdm
import shutil


def parseArguments():
    parser = argparse.ArgumentParser(
        description="Combines the ligand, protein, and GRAIL files into a standardized raw folder structure."
    )

    parser.add_argument(
        "-d",
        dest="pdb_bind_dir",
        metavar="<directory>",
        required=True,
        help="Root directory of the PDBbind v.2020 dataset.",
    )

    return parser.parse_args()


def main(args):
    root = args.pdb_bind_dir
    sub_folders = ["refined-set", "v2020-other-PL"]
    sub_folders_grail = ["refined-set-GRAIL", "v2020-other-PL-GRAIL"]
    root_raw = os.path.join(root, "raw")

    os.makedirs(root_raw, exist_ok=True)

    for i, sub_folder in enumerate(sub_folders):
        sub_path = os.path.join(root, sub_folder)
        sub_path_grail = os.path.join(root, sub_folders_grail[i])
        for target in tqdm(os.listdir(sub_path)):
            ligand_path = os.path.join(sub_path, target, f"{target}_ligand.sdf")
            protein_path = os.path.join(sub_path, target, f"{target}_protein.pdb")
            grail_path = os.path.join(sub_path_grail, f"{target}.cdf")
            if not os.path.exists(ligand_path) or not os.path.exists(grail_path):
                print(
                    f"Missing files for target {target}: ligand or GRAIL file not found."
                )
                continue
            new_dir = os.path.join(root_raw, target)
            if os.path.exists(new_dir):
                continue
            ligand_path_target = os.path.join(new_dir, "ligand.sdf")
            protein_path_target = os.path.join(new_dir, "protein.pdb")
            grail_path_target = os.path.join(new_dir, "map.cdf")
            # value = data[data["PDB_code"] == target]["Binding_Affinity_Log"]
            # if len(value) == 0:
            #     print(f"Binding affinity not found for target {target}.")
            #     continue
            os.makedirs(new_dir, exist_ok=True)
            shutil.copy(ligand_path, ligand_path_target)
            shutil.copy(protein_path, protein_path_target)
            shutil.copy(grail_path, grail_path_target)
            # with open(os.path.join(new_dir, "binding_affinity.txt"), "w") as f:
            #     f.write(str(value.values[0]) + "\n")


if __name__ == "__main__":
    main(parseArguments())
