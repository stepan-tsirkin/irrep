import os
import subprocess
from pathlib import Path
import irrep
from irrep.bandstructure import BandStructure
from irrep.gvectors import symm_matrix
from irrep.utility import get_block_indices, is_round
from monty.serialization import loadfn
import numpy as np
import pytest

TEST_FILES_PATH = Path(__file__).parents[2] / "examples"
REF_DATA_PATH = Path(__file__).parents[0] / "ref_data"
TMP_DATA_PATH = Path(__file__).parents[0] / "tmp_data"
if not os.path.exists(TMP_DATA_PATH):
    os.makedirs(TMP_DATA_PATH)


def test_bi_hoti():

    
    # Test specifying refUC in CLI
    command = [
        "irrep",
        "-spinor",
        "-code=vasp",
        "-kpnames=GM",
        "-kpoints=2",
        "-Ecut=50",
        "-EF=5.2156",
        "-IBstart=5",
        "-IBend=10",
        "-isymsep=3",
    ]
    example_dir = "Bi-hoti"
    ref_file = "ref_output_isymsep.json"
    check_isymsep(example_dir, command, ref_file)

@pytest.mark.parametrize("isym", [5,11])
def test_vasp_scalar(isym):
    # Test specifying refUC in CLI
    output_file = f"irrep-output_isymsep-{isym}.json"
    command = [
        "irrep",
        "-code=vasp",
        "-kpoints=3",
        "-Ecut=50",
        "-IBend=10",
        f"-isymsep={isym}",
        f"-json_file={output_file}",
    ]
    example_dir = "vasp_scalar"
    ref_file = f"ref_output_isymsep-{isym}.json"
    output_file = f"irrep-output_isymsep-{isym}.json"
    check_isymsep(example_dir, command, ref_file, output_file, check_irreps=False)

def check_isymsep(example_dir, command, ref_file, output_file="irrep-output.json", check_irreps=True):
    os.chdir(TEST_FILES_PATH / example_dir)

    output = subprocess.run(command, capture_output=True, text=True)
    return_code = output.returncode
    assert return_code == 0, output.stderr

    # Load generated and reference output data
    data_ref = loadfn(ref_file)
    data_run = loadfn(output_file)

    # Check SpaceGroup
    sg_ref = data_ref['spacegroup']
    sg_run = data_run['spacegroup']
    assert sg_ref['name'] == sg_run['name']
    assert sg_ref['number'] == sg_run['number']
    assert sg_ref['spinor'] == sg_run['spinor']
    assert sg_ref['num symmetries'] == sg_run['num symmetries']
    assert sg_ref['cells match'] == sg_run['cells match']
    spinor = sg_ref['spinor']  # used later

    # Todo: implement safe check of symmetries


    # Check properties of separation by eigenvalues of symmetries
    assert data_ref['separated by symmetry'] == data_run['separated by symmetry']
    assert data_ref['separating symmetries'] == data_run['separating symmetries']
    assert len(data_ref['characters and irreps']) == len(data_ref['characters and irreps'])
    num_cases = len(data_ref['characters and irreps'])

    for i in range(num_cases):

        # Check that identified eigenvalues match
        evals_ref = data_ref['characters and irreps'][i]['symmetry eigenvalues']
        evals_run = data_run['characters and irreps'][i]['symmetry eigenvalues']
        assert len(evals_ref) == len(evals_run)
        for val1, val2 in zip(evals_ref, evals_run):
            assert abs(val1 - val2) < 1e-3

        # Check properties of bandstructure
        bs_ref = data_ref['characters and irreps'][i]['subspace']
        bs_run = data_run['characters and irreps'][i]['subspace']
        assert bs_ref['indirect gap (eV)'] == bs_run['indirect gap (eV)']
        assert bs_ref['Minimal direct gap (eV)'] == bs_run['Minimal direct gap (eV)']
        if spinor:
            assert bs_ref['Z4'] == bs_run['Z4']
            assert bs_ref['number of inversion-odd Kramers pairs'] == bs_run['number of inversion-odd Kramers pairs']
        else:
            assert bs_ref['number of inversion-odd states'] == bs_run['number of inversion-odd states']

        # Check properties at each k-point
        kp_ref = bs_ref['k points'][0]
        kp_run = bs_run['k points'][0]
        assert np.allclose(kp_ref['symmetries'], kp_run['symmetries'])
        assert np.allclose(kp_ref['energies_mean'], kp_run['energies_mean'], rtol=0., atol=1e-4)
        assert np.allclose(kp_ref['characters'], kp_run['characters'], rtol=0., atol=1e-4)
        assert kp_ref['characters refUC is the same'] == kp_run['characters refUC is the same']
        assert np.allclose(kp_ref['dimensions'], kp_run['dimensions'], rtol=0., atol=1e-4)
        if check_irreps:
            for irrep_ref, irrep_run in zip(kp_ref['irreps'], kp_run['irreps']):
                assert len(irrep_ref) == len(irrep_run)
                for irrepname_ref, irrepname_run in zip(irrep_ref.keys(), irrep_run.keys()):
                    assert irrepname_ref == irrepname_run  # compare strings of irreps
                    assert np.allclose(irrep_ref[irrepname_ref], irrep_run[irrepname_run], rtol=0, atol=1e-4)  # compare multiplicities

    # Remove output files created during run
    for test_output_file in (
            "irreps.dat",
            "irreptable-template",
            "trace.txt",
            "irrep-output.json"
    ):
        if os.path.exists(test_output_file):
            os.remove(test_output_file)


def test_symm_matrix_full():
    check_symm_matrix(example_dir="vasp_spinor",
                      output_file="symm_matrix_full.npz" ,
                      Ecut=100,
                      degen_thresh=None,
                      acc=1e-6
                      )

def test_symm_matrix_block_vs_full():
    check_symm_matrix(example_dir="vasp_spinor",
                      output_file="symm_matrix_block.npz" ,
                      ref_file="symm_matrix_full.npz",
                      Ecut=30,
                      degen_thresh=1e-2,
                      acc=1e-4
                      )

def test_symm_matrix_block():
    check_symm_matrix(example_dir="vasp_spinor",
                      output_file="symm_matrix_block.npz" ,
                    #   ref_file="symm_matrix_full.npz",
                      Ecut=30,
                      degen_thresh=1e-2,
                      acc=1e-5
                      )
    

def test_symm_matrix_block_2():
    check_symm_matrix(example_dir="vasp_spinor",
                      output_file="symm_matrix_block.npz" ,
                    #   ref_file="symm_matrix_full.npz",
                      Ecut=60,
                      degen_thresh=1e-2,
                      acc=5e-7
                      )


def check_symm_matrix(example_dir, output_file="symm_matrix", ref_file=None, degen_thresh=None, Ecut=30,
                      acc=1e-6):
    if ref_file is None:
        ref_file = output_file
    path = os.path.join(TEST_FILES_PATH , example_dir)
    bandstructure = BandStructure(code='vasp', 
                                  fPOS=os.path.join(path, 'POSCAR'),
                                  fWAV=os.path.join(path, 'WAVECAR'),
                                  Ecut=Ecut, 
                                  spinor=True, normalize=False,
                                  IBend=20)
    points = []
    matrices=[]
    matrices2=[] # calculate blocks separately, but collect to one big matrix
    for k1,K1 in enumerate(bandstructure.kpoints):
        if degen_thresh is not None:
            block_indices = get_block_indices(K1.Energy_raw, thresh=degen_thresh, cyclic=False)
        else:
            block_indices = None
        for k2,K2 in enumerate(bandstructure.kpoints):
            for isym, symop in enumerate(bandstructure.spacegroup.symmetries):
                if is_round(symop.transform_k(K1.k)-K2.k, prec=1e-5):
                    points.append((k1,k2,isym))
                    kwargs = dict(K=K1.k, K_other=K2.k, 
                                  WF=K1.WF, WF_other=K2.WF, 
                                  igall=K1.ig, igall_other=K2.ig, 
                                  A=symop.rotation, S=symop.spinor_rotation, 
                                  T=symop.translation, spinor = K1.spinor)

                    matrices.append(symm_matrix( **kwargs))
                    matrices2.append(symm_matrix( block_ind=block_indices, **kwargs))   
    matrices2 = np.array(matrices2)
    matrices = np.array(matrices)                                     
    tmp_file = TMP_DATA_PATH / output_file
    np.savez_compressed(tmp_file, points=points, matrices=matrices)
    reference = np.load(REF_DATA_PATH / ref_file)
    assert np.allclose(reference['points'], points)
    diff = abs(reference['matrices']-matrices)
    if np.max(diff)>acc:
        string = f"Matrices differ by {np.max(diff)}, {diff.shape}\n"
        for i,j,k in zip(*np.where(diff>1e-5)):
            string+=f"{i}, {j} , {k}, {diff[i,j,k]}, {reference['matrices'][i,j,k]}, {matrices[i,j,k]}\n"
        raise ValueError(string)
    
    # os.remove(tmp_file)
            

    