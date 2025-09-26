# -*- mode: python; tab-width: 4 -*-

##
# calc_dgrails_pdb_bind.py
#
# Copyright (C) 2025 Thomas A. Seidel <thomas.seidel@univie.ac.at>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; see the file COPYING. If not, write to
# the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
##

import argparse
import os
import sys
import time
from tqdm import tqdm

from dask import delayed, compute
from dask.distributed import Client, LocalCluster

import CDPL.Chem as Chem
import CDPL.Biomol as Biomol
import CDPL.Math as Math
import CDPL.Pharm as Pharm
import CDPL.GRAIL as GRAIL
import CDPL.Grid as Grid


REMOVE_NON_STD_RESIDUES = True
NON_STD_RESIDUE_MAX_ATOM_COUNT = 8


def parseArguments():
    parser = argparse.ArgumentParser(
        description="Calculates GRAIL maps for ligand-protein complexes from PDBbind."
    )

    parser.add_argument(
        "-d",
        dest="complex_data_dir",
        metavar="<directory>",
        required=True,
        help="The directory containing the ligand-protein complexes to process, organized in PDBbind manner.",
    )
    parser.add_argument(
        "-o",
        dest="grails_out_dir",
        metavar="<directory>",
        required=True,
        help="The path of the output directory containing the calculated GRAIL maps.",
    )
    parser.add_argument(
        "-b",
        dest="bur_grid",
        required=False,
        action="store_true",
        default=False,
        help="Additionally generate a binding site buriedness grid (default: false).",
    ),
    parser.add_argument(
        "-e",
        dest="env_max_dist",
        required=False,
        metavar="<float>",
        default=7.0,
        help="Maximum distance of residues considered as environment (default: 7 Ang)",
        type=float,
    )
    parser.add_argument(
        "-r",
        dest="grid_res",
        required=False,
        metavar="<float>",
        default=0.5,
        help="Grid resolution (default: 0.5 Ang)",
        type=float,
    )
    parser.add_argument(
        "-n",
        dest="normalize",
        required=False,
        action="store_true",
        default=False,
        help="Normalize GRAIL scores (default: false).",
    ),
    parser.add_argument(
        "-w",
        dest="worker_nodes",
        required=False,
        metavar="<int>",
        default=4,
        help="Number of worker nodes (default: 4)",
        type=int,
    )

    return parser.parse_args()


def excludeResidue(pdb_code, res):
    is_std_res = Biomol.ResidueDictionary.isStdResidue(Biomol.getResidueCode(res))

    if is_std_res and res.numAtoms < 5:
        print(
            f" ! While processing complex {pdb_code}: isolated standard residue fragment of size {res.numAtoms} found",
            file=sys.stderr,
        )
        return False

    if (
        REMOVE_NON_STD_RESIDUES
        and not is_std_res
        and res.numAtoms <= NON_STD_RESIDUE_MAX_ATOM_COUNT
    ):
        if res.numAtoms == 1 and Chem.AtomDictionary.isMetal(
            Chem.getType(res.atoms[0])
        ):
            return False

        return True

    return False


def checkProtein(pdb_code, protein):
    for atom in protein.atoms:
        if Chem.getType(atom) == Chem.AtomType.H and atom.numAtoms == 0:
            print(
                f" ! While processing complex {pdb_code}: isolated hydrogen atom encountered",
                file=sys.stderr,
            )

        elif Chem.getType(atom) == Chem.AtomType.UNKNOWN:
            print(
                f" ! While processing complex {pdb_code}: atom of unknown element encountered",
                file=sys.stderr,
            )


def extractEnvironmentResidues(pdb_code, ligand, protein, env_max_dist):
    bbox_min = Math.Vector3D()
    bbox_max = Math.Vector3D()

    Chem.calcBoundingBox(ligand, bbox_min, bbox_max, True)

    for i in range(3):
        bbox_min[i] = bbox_min[i] - env_max_dist
        bbox_max[i] = bbox_max[i] + env_max_dist

    lig_env = Chem.Fragment()
    residues = Biomol.ResidueList(protein)
    num_res = 0

    for res in residues:
        if excludeResidue(pdb_code, res):
            continue

        if Chem.intersectsBoundingBox(res, bbox_min, bbox_max):
            lig_env += res
            num_res += 1

    for bond in protein.bonds:
        if lig_env.containsAtom(bond.getBegin()) and lig_env.containsAtom(
            bond.getEnd()
        ):
            lig_env.addBond(bond)

    Chem.extractSSSRSubset(protein, lig_env, True)

    return bbox_min, bbox_max, lig_env, num_res


def processComplex(
    pdb_code, comp_data_dir, out_dir, grid_res, env_max_dist, grail_calc, bur_calc
):
    print(f"* Processing complex {pdb_code}...")

    start_time = time.time()
    sdf_reader = Chem.FileSDFMoleculeReader(
        os.path.join(comp_data_dir, pdb_code + "_ligand.sdf")
    )
    ligand = Chem.BasicMolecule()

    if not sdf_reader.read(ligand):
        print(
            f" ! While processing complex {pdb_code}: reading ligand SD-file failed",
            file=sys.stderr,
        )
        return

    pdb_reader = Biomol.FilePDBMoleculeReader(
        os.path.join(comp_data_dir, pdb_code + "_protein.pdb")
    )

    Biomol.setPDBApplyDictAtomBondingToNonStdResiduesParameter(pdb_reader, True)
    Biomol.setApplyDictFormalChargesParameter(pdb_reader, False)

    if pdb_code == "6m0p":
        Biomol.setPDBIgnoreCONECTRecordsParameter(pdb_reader, True)

    protein = Chem.BasicMolecule()

    if not pdb_reader.read(protein):
        print(
            f" ! While processing complex {pdb_code}: reading protein PDB-file failed",
            file=sys.stderr,
        )
        return

    checkProtein(pdb_code, protein)

    Pharm.prepareForPharmacophoreGeneration(ligand, False, False)
    Pharm.prepareForPharmacophoreGeneration(protein, True, True)

    bbox_min, bbox_max, lig_env, num_res = extractEnvironmentResidues(
        pdb_code, ligand, protein, env_max_dist
    )

    grail_calc.setGridParamsForBoundingBox(bbox_min, bbox_max)

    print(
        f" - Grid size (points): {grail_calc.gridXSize}x{grail_calc.gridYSize}x{grail_calc.gridZSize}"
    )
    print(
        f" - Total number of grid points: {grail_calc.gridXSize * grail_calc.gridYSize * grail_calc.gridZSize}"
    )
    print(
        f" - Spatial grid dimensions (min, max): {bbox_min[0]:.3f},{bbox_min[1]:.3f},{bbox_min[2]:.3f},{bbox_max[0]:.3f},{bbox_max[1]:.3f},{bbox_max[2]:.3f}"
    )
    print(f" - Number of residues covered by grid calculation: {num_res}")

    grid_set = Grid.DRegularGridSet()
    grid_set_writer = Grid.FileCDFDRegularGridSetWriter(
        os.path.join(out_dir, pdb_code + ".cdf")
    )
    coords_func = Chem.Atom3DCoordinatesFunctor()

    grail_calc.calcInteractionGrids(lig_env, coords_func, grid_set)
    grid_set.addElement(grail_calc.calcAtomDensityGrid(ligand, coords_func, "LIG"))

    if bur_calc:
        bur_grid = Grid.DRegularGrid(grid_res)
        bur_grid.resize(
            grail_calc.gridXSize, grail_calc.gridYSize, grail_calc.gridZSize, False
        )
        bur_grid.setCoordinatesTransform(grail_calc.gridTransform)

        Grid.setName(bur_grid, "BUR")

        grid_set.addElement(bur_grid)

        bur_calc.setAtom3DCoordinatesFunction(coords_func)
        bur_calc.calculate(lig_env, bur_grid)

    if not grid_set_writer.write(grid_set):
        print(
            f" ! While processing complex {pdb_code}: error while writing output file",
            file=sys.stderr,
        )
        return

    grid_set_writer.close()

    print(f" - GRAIL grid set calculated in {int(time.time() - start_time)}s")


def setHydrophicFtrWeights(pharm):
    for feature in pharm:
        if Pharm.getType(feature) == Pharm.FeatureType.HYDROPHOBIC:
            Pharm.setWeight(feature, Pharm.getHydrophobicity(feature))


def process_complex_task(pdb_code, comp_data_dir, args):
    """
    A task to process a single complex. This function will be executed in parallel.
    """
    grail_calc = GRAIL.GRAILDataSetGenerator()
    grail_calc.setGridStepSize(args.grid_res)
    grail_calc.pharmGenerator.applyConfiguration(
        Pharm.DefaultPharmacophoreGenerator.DEFAULT_CONFIG
    )
    grail_calc.pharmProcessingFunction = setHydrophicFtrWeights
    grail_calc.normalizedScores = args.normalize


    bur_calc = None
    if args.bur_grid:
        bur_calc = GRAIL.BuriednessGridCalculator()

    try:
        df = processComplex(
            pdb_code,
            comp_data_dir,
            args.grails_out_dir,
            args.grid_res,
            args.env_max_dist,
            grail_calc,
            bur_calc,
        )
        return df
    except Exception as e:
        print(f" ! Processing complex {pdb_code} failed:", e, file=sys.stderr)
        return None


def process(args):
    """
    Main function to process all complexes in parallel using Dask.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(args.grails_out_dir):
        os.makedirs(args.grails_out_dir)

    # Set up a Dask LocalCluster with a specified number of workers
    cluster = LocalCluster(n_workers=args.worker_nodes, threads_per_worker=1)
    client = Client(cluster)

    print(f"Dask cluster initialized with {args.worker_nodes} workers.")

    # Collect all tasks
    tasks = []
    for pdb_code in os.listdir(args.complex_data_dir):
        comp_data_dir = os.path.join(args.complex_data_dir, pdb_code)

        if os.path.isfile(comp_data_dir):  # sanity check
            continue

        # Create a delayed task for each pdb_code
        task = delayed(process_complex_task)(pdb_code, comp_data_dir, args)
        tasks.append(task)

    # Use tqdm to show progress
    with tqdm(total=len(tasks), desc="Processing complexes") as pbar:
        # Compute tasks in parallel
        results = []
        for result in compute(*tasks):
            results.append(result)
            pbar.update(1)

    print("Done!")
    client.close()


if __name__ == "__main__":
    process(parseArguments())
