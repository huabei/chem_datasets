from mpi4py import MPI
from mpi4py.futures import MPICommExecutor

import warnings
from Bio.PDB import PDBParser, PPBuilder, CaPPBuilder
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Selection import unfold_entities

import numpy as np
import dask.array as da

from rdkit import Chem

import os
import re

# all punctuation
punctuation_regex  = r"""(\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""

# tokenization regex (Schwaller)
molecule_regex = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""

cutoff = 5
max_seq = 2048
max_smiles = 512
chunk_size = '1G'

def parse_complex(fn):
    try:
        name = os.path.basename(fn)

        # parse protein sequence and coordinates
        parser = PDBParser()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = parser.get_structure('protein',fn+'/'+name+'_protein.pdb')

#        ppb = PPBuilder()
        ppb = CaPPBuilder()
        seq = []
        for pp in ppb.build_peptides(structure):
            seq.append(str(pp.get_sequence()))
        seq = ''.join(seq)

        # parse ligand, convert to SMILES and map atoms
        suppl = Chem.SDMolSupplier(fn+'/'+name+'_ligand.sdf')
        mol = next(suppl)
        smi = Chem.MolToSmiles(mol)

        # position of atoms in SMILES (not counting punctuation)
        atom_order = mol.GetProp("_smilesAtomOutputOrder")
        atom_order = [int(s) for s in list(filter(None,re.sub(r'[\[\]]','',mol.GetProp("_smilesAtomOutputOrder")).split(',')))]

        # tokenize the SMILES
        tokens = list(filter(None, re.split(molecule_regex, smi)))

        # remove punctuation
        masked_tokens = [re.sub(punctuation_regex,'',s) for s in tokens]

        k = 0
        token_pos = []
        token_id = []
        for i,token in enumerate(masked_tokens):
            if token != '':
                token_pos.append(tuple(mol.GetConformer().GetAtomPosition(atom_order[k])))
                token_id.append(i)
                k += 1

        # query protein for ligand contacts
        atoms  = unfold_entities(structure, 'A')
        neighbor_search = NeighborSearch(atoms)

        close_residues = [neighbor_search.search(center=t, level='R', radius=cutoff) for t in token_pos]
        residue_id = [[c.get_id()[1]-1 for c in query] for query in close_residues] # zero-based

        # contact map
        contact_map = np.zeros((max_seq, max_smiles),dtype=np.float32)

        for query,t in zip(residue_id,token_id):
            for r in query:
                contact_map[r,t] = 1

        return name, seq, smi, contact_map
    except Exception as e:
        print(e)
        return None


if __name__ == '__main__':
    import glob

    filenames = glob.glob('raw/v2020-other-PL/*')
    filenames.extend(glob.glob('raw/refined-set/*'))
    comm = MPI.COMM_WORLD
    with MPICommExecutor(comm, root=0) as executor:
        if executor is not None:
            result = executor.map(parse_complex, filenames)
            result = list(result)
            names = [r[0] for r in result if r is not None]
            seqs = [r[1] for r in result if r is not None]
            all_smiles = [r[2] for r in result if r is not None]
            all_contacts = [r[3] for r in result if r is not None]

            import pandas as pd
            df = pd.DataFrame({'name': names, 'seq': seqs, 'smiles': all_smiles})
            all_contacts = da.from_array(all_contacts, chunks=chunk_size)
            da.to_npy_stack('pdbbind_contacts/', all_contacts)
            df.to_parquet('pdbbind_complex.parquet')
