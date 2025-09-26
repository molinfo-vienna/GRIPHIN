import pandas as pd
import re
import argparse
import numpy as np
import os


def parseArguments():
    parser = argparse.ArgumentParser(
        description="Processes the PDBbind index file to extract and standardize binding affinity data."
    )

    parser.add_argument(
        "-d",
        dest="pdb_bind_dir",
        metavar="<directory>",
        required=True,
        help="Root directory of the PDBbind v.2020 dataset.",
    )

    return parser.parse_args()


def parse_index_file(path):
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


# Define functions to extract Ki, Kd, and IC50 values
def extract_ki(binding_data):
    if "Ki=" in binding_data:
        return binding_data.split("Ki=")[1].split()[0]
    return None


def extract_kd(binding_data):
    if "Kd=" in binding_data:
        return binding_data.split("Kd=")[1].split()[0]
    return None


def extract_ic50(binding_data):
    if "IC50=" in binding_data:
        return binding_data.split("IC50=")[1].split()[0]
    return None


# Function to extract and standardize values
def extract_and_standardize(binding_data, key):
    if pd.notna(binding_data):  # Ensure the input is not NaN
        # Use a regular expression to match the key followed by =, ~, or <
        match = re.search(rf"{key}(?:[=~<>]|<=|>=)(\d*\.?\d+[a-zA-Z]*)", binding_data)
        if match:
            value_with_unit = match.group(1)  # Extract the value with the unit
            # Convert the value based on the unit
            if "mM" in value_with_unit:
                value = float(value_with_unit.replace("mM", "")) * 1e-3
            elif "uM" in value_with_unit:
                value = float(value_with_unit.replace("uM", "")) * 1e-6
            elif "nM" in value_with_unit:
                value = float(value_with_unit.replace("nM", "")) * 1e-9
            elif "pM" in value_with_unit:
                value = float(value_with_unit.replace("pM", "")) * 1e-12
            elif "fM" in value_with_unit:
                value = float(value_with_unit.replace("fM", "")) * 1e-15
            else:
                value = float(
                    value_with_unit
                )  # Assume it's already in M if no unit is specified
            return value
    return None


def main(args):
    print(args)
    path = args.pdb_bind_dir
    activity_data = os.path.join(path, "index", "INDEX_general_PL.2020")
    data = parse_index_file(activity_data)
    data["Ki"] = data["Binding_data"].apply(extract_ki)
    data["Kd"] = data["Binding_data"].apply(extract_kd)
    data["IC50"] = data["Binding_data"].apply(extract_ic50)
    # Standardize Ki, Kd, and IC50 columns
    data["Ki"] = data["Binding_data"].apply(lambda x: extract_and_standardize(x, "Ki"))
    data["Kd"] = data["Binding_data"].apply(lambda x: extract_and_standardize(x, "Kd"))
    data["IC50"] = data["Binding_data"].apply(
        lambda x: extract_and_standardize(x, "IC50")
    )

    # Combine Ki, Kd, and IC50 into a single column, prioritizing Ki > Kd > IC50
    data["Binding_Affinity"] = (
        data["Ki"].combine_first(data["Kd"]).combine_first(data["IC50"])
    )
    data["Binding_Affinity_Log"] = -np.log10(data["Binding_Affinity"])
    data[["PDB_code", "Binding_Affinity_Log"]].to_csv(
        os.path.join(path, "binding_affinities.csv"), index=False, header=False
    )


if __name__ == "__main__":
    main(parseArguments())
