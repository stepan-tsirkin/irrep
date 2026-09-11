from irrep.spacegroup import SpaceGroup
import numpy as np
import pytest

from irrep.tests.test_dmn import REF_FILES_PATH, TMP_FILES_PATH

from .conftest import REF_DATA_PATH, TMP_DATA_PATH
from irrep.utility import group_numbers
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

a = 4.456
c = 5.926
x = 0.24
struct_param_Te = dict(real_lattice=[[a, 0, 0],
                                     [-a / 2, a * (3)**0.5 / 2, 0],
                                     [0, 0, c]],
                       positions=[
    [x, 0, 0],
    [0, x, 1 / 3],
    [-x, -x, 2 / 3]],
    typat=[1, 1, 1],)




def test_spacegroup_Te_noTR():
    spacegroup_Te = SpaceGroup.from_cell(**struct_param_Te,
                                         include_TR=False,)

    np.savez(TMP_DATA_PATH / "spacegroup_Te_noTR.npz", **spacegroup_Te.as_dict())
    assert spacegroup_Te.size == 6
    assert spacegroup_Te.number == 152
    assert spacegroup_Te.name == "P3_121"


def test_spacegroup_Te_TR():
    spacegroup_Te = SpaceGroup.from_cell(**struct_param_Te,
                                         include_TR=True)

    np.savez(TMP_DATA_PATH / "spacegroup_Te_TR.npz", **spacegroup_Te.as_dict())
    assert spacegroup_Te.size == 12
    assert spacegroup_Te.number_str == "152.34"
    assert spacegroup_Te.name == "P3_1211'"



def test_spacegroup_Te_product_noTR():
    spacegroup = SpaceGroup( **np.load(REF_DATA_PATH / "spacegroup_Te_noTR.npz"))
    product, transl_diff, spinor_factors = spacegroup.get_product_table(get_diff=True)
    np.savez(TMP_DATA_PATH / "spacegroup_Te_noTR_product.npz", product=product, transl_diff=transl_diff)
    ref = np.load(REF_DATA_PATH / "spacegroup_Te_noTR_product.npz")
    assert np.all(product == ref["product"])
    assert np.all(transl_diff == ref["transl_diff"])
    assert np.all(spinor_factors == ref["spinor_factors"])


def test_spacegroup_Te_product_TR():
    spacegroup = SpaceGroup( **np.load(REF_DATA_PATH / "spacegroup_Te_TR.npz"))
    product, transl_diff, spinor_factors = spacegroup.get_product_table(get_diff=True)
    np.savez(TMP_FILES_PATH / "spacegroup_Te_TR_product.npz", product=product, transl_diff=transl_diff, spinor_factors=spinor_factors)
    ref = np.load(REF_FILES_PATH / "spacegroup_Te_TR_product.npz")
    assert np.all(product == ref["product"])
    assert np.all(transl_diff == ref["transl_diff"])
    assert np.all(spinor_factors == ref["spinor_factors"])


def test_group_numbers():
    lst = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
           1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    grouped = group_numbers(lst, precision=0.2)
    assert np.allclose(grouped, np.mean(lst))
    lst = [0.09, 0.1, 0.11, 0.2, 0.21, 0.22]
    grouped = group_numbers(lst, precision=0.05)
    assert np.allclose(grouped, [0.1] * 3 + [0.21] * 3)

    lst = [0.21, 0.09, 0.1, 0.2, 0.11, 0.22]
    grouped = group_numbers(lst, precision=0.05)
    assert np.allclose(grouped, [0.21, 0.1, 0.1, 0.21, 0.1, 0.21])


def test_conventional_wyckoff_positions():
    """ Test for spacegroups 221 and 225, manually"""

    output_221 = ['x,y,z', 'x,x,z', '1/2,y,z', '0,y,z', '1/2,y,y', '0,y,y', 'x,1/2,0', 'x,x,x', 'x,1/2,1/2', 'x,0,0', '1/2,0,0', '0,1/2,1/2', '1/2,1/2,1/2', '0,0,0']
    output_225 = ['x,y,z', 'x,x,z', '0,y,z', '1/2,y,y', '0,y,y', 'x,1/4,1/4', 'x,x,x', 'x,0,0', '0,1/4,1/4', '1/4,1/4,1/4', '1/2,1/2,1/2', '0,0,0']


    assert SpaceGroup.conventional_wyckoff_positions(221) == output_221
    assert SpaceGroup.conventional_wyckoff_positions(225) == output_225

    return


def test_wyckoff_positions():
    """ Test for NaCl """
    a = 5.64  # lattice distance in Å

    # Array with the 3 lattice base vectors {a_1, a_2, a_3}
    lattice = [
        [a, 0, 0],
        [0, a, 0],
        [0, 0, a]
    ]

    # Position of the atoms in the lattice
    positions = np.array([
        [0.0, 0.0, 0.0],        # Cl
        [0.5, 0.5, 0.5]         # Na
    ])
    positions = positions + [1 / 2, 1 / 2, 0]  # Shift to test the origin shift parameter

    numbers = [17, 11]  # Cl and Na atomi numbers

    cell = (lattice, positions, numbers)
    NaCl_wyckoffs = SpaceGroup.wyckoff_positions(cell)

    first_NaCl_wyckoffs = [[0.5, 0.0, -0.5],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, -0.5]]

    # Testing the first 4 wp because they are numerical arrays not symbols
    # Then we can test without the numerical errors of the symbolic strings
    for i, wp in enumerate(NaCl_wyckoffs[-4:]):
        numerical_wp = wp.split(',')
        numerical_wp = [ float(x) for x in numerical_wp]
        assert np.allclose(numerical_wp, first_NaCl_wyckoffs[i])

    return


def test_wyckoff_positions_2():
    """ Test for MnTe, which has a spacegroup containing WP of the form (2x,...,...), which can break the sympify parser """
    a = 4.17349018

    cell = (
        np.array([
            [a, 0, 0],
            [-a / 2, np.sqrt(3) * a / 2, 0],
            [0, 0, 6.75345133],
        ]),

        np.array([
            [0, 0, 0],
            [0, 0, 0.5],
            [1 / 3, 2 / 3, 1 / 4],
            [2 / 3, 1 / 3, 3 / 4],
        ]),

        np.array([25, 25, 52, 52]),
    )

    MnTe_wyckoffs = SpaceGroup.wyckoff_positions(cell)

    first_MnTe_wyckoffs = [
        [1 / 3, 2 / 3, 0.75],
        [1 / 3, 2 / 3, 0.25],
        [0, 0, 0.25],
        [0, 0, 0],
    ]
    print(MnTe_wyckoffs)
    # Testing the first 4 wp because they are numerical arrays not symbols
    for i, wp in enumerate(MnTe_wyckoffs[-4:]):
        numerical_wp = wp.split(',')
        numerical_wp = [ float(x) for x in numerical_wp]
        assert np.allclose(numerical_wp, first_MnTe_wyckoffs[i])
    # checking correct parsing of the symbolic strings
    assert MnTe_wyckoffs[4].split(',')[1] == '2.0*x'
    return


def test_implicit_multiplication_parsing():
    """test lines 1060-1061 of spacegroup.py, which use parse_expr with implicit multiplication"""
    x, y = sympy.symbols('x y')
    _TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
    assert parse_expr('2x', transformations=_TRANSFORMS) == 2 * x
    assert parse_expr('-2x+1', transformations=_TRANSFORMS) == -2 * x + 1
    assert parse_expr('2x-y', transformations=_TRANSFORMS) == 2 * x - y


@pytest.mark.parametrize("include_TR", [True, False])
def test_spacegroup_save_load(include_TR):
    spacegroup_Te = SpaceGroup.from_cell(**struct_param_Te,
                                         include_TR=include_TR)
    tmp_file_path = TMP_DATA_PATH / "spacegroup_Te_TR={}.npz".format(include_TR)
    print(f"spinor : {spacegroup_Te.spinor}, include_TR: {include_TR}")
    np.savez(tmp_file_path, **spacegroup_Te.as_dict())
    spacegroup_Te_loaded = SpaceGroup.from_npz(tmp_file_path)
    for key in spacegroup_Te.__dict__:
        type_key = type(spacegroup_Te.__dict__[key])
        assert type_key == type(spacegroup_Te_loaded.__dict__[key]), f"Type mismatch in attribute {key}, saved {type(getattr(spacegroup_Te, key))}, loaded {type(getattr(spacegroup_Te_loaded, key))}"
        if key in ['real_lattice', 'positions', 'typat', 'magmom', 'number', 'number_str']:
            if type_key == np.ndarray:
                assert np.allclose(getattr(spacegroup_Te, key), getattr(spacegroup_Te_loaded, key)), f"Value mismatch in attribute {key}, saved {getattr(spacegroup_Te, key)}, loaded {getattr(spacegroup_Te_loaded, key)}"
            else:
                assert getattr(spacegroup_Te, key) == getattr(spacegroup_Te_loaded, key), f"Value mismatch in attribute {key}, saved {getattr(spacegroup_Te, key)}, loaded {getattr(spacegroup_Te_loaded, key)}"
    assert spacegroup_Te.equals(spacegroup_Te_loaded), "Loaded spacegroup is not equal to the original"
