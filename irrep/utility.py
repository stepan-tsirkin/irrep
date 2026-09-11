
# ###   ###   #####  ###
# #  #  #  #  #      #  #
# ###   ###   ###    ###
# #  #  #  #  #      #
# #   # #   # #####  #


##################################################################
## This file is distributed as part of                           #
## "IrRep" code and under terms of GNU General Public license v3 #
## see LICENSE file in the                                       #
##                                                               #
##  Written by Stepan Tsirkin                                    #
##  e-mail: stepan.tsirkin@ehu.eus                               #
##################################################################


from fractions import Fraction
import warnings
import numpy as np
from scipy import constants
import fortio
import sys
from typing import Any
BOHR = constants.physical_constants['Bohr radius'][0] / constants.angstrom


# Global cache to store einsum paths
EINSUM_PATH_CACHE = {}


def cached_einsum(subscripts, *operands,
                  optimize='greedy',
                  **kwargs):
    """
    A wrapper for np.einsum that caches the contraction path.
    The cache key is a combination of the subscripts string and the
    shapes of the operand arrays.
    """
    shapes = tuple(op.shape for op in operands)
    cache_key = (subscripts, shapes)

    if cache_key in EINSUM_PATH_CACHE:
        path = EINSUM_PATH_CACHE[cache_key]
    else:
        path = np.einsum_path(subscripts, *operands, optimize=optimize)[0]
        EINSUM_PATH_CACHE[cache_key] = path
    return np.einsum(subscripts, *operands, optimize=path, **kwargs)


class FortranFileR(fortio.FortranFile):
    '''
    Class that implements `syrte/fortio` package to parse long records
    in Abinit WFK file.

    Parameters
    ----------
    filename : str
        Path to the WFK file.
    '''

    def __init__(self, filename):

        print("Using fortio to read")

        try:  # assuming there are not subrecords
            super().__init__(filename,
                             mode='r',
                             header_dtype='uint32',
                             auto_endian=True,
                             check_file=True
                             )
            print("Long records not found in ", filename)
        except ValueError:  # there are subrecords, allow negative markers
            print(f"File '{filename}' contains subrecords - using header_dtype='int32'")
            super().__init__(filename,
                             mode='r',
                             header_dtype='int32',
                             auto_endian=True,
                             check_file=True
                             )


def str2list(string):
    """
    Generate `list` from `str`, where elements are separated by '-'.
    Used when parsing parameters set in CLI.

    Parameters
    ----------
    string : str
        `str` to be parsed.

    Returns
    -------
    array

    Notes
    -----
    Ranges can be generated as part of the output `array`. For example, 
    `str2list('1,3-5,7')` will give as ouput `array([1,3,4,5,7])`.
    """
    return np.hstack([np.arange(*(np.array(s.split("-"), dtype=int) + np.array([0, 1])))
                      if "-" in s else np.array([int(s)]) for s in string.split(",")])


def compstr(string):
    """
    Convers `str` to `float` or `complex`.

    Parameters
    ----------
    string : str
        String to convert.
    Returns
    -------
    float or complex
        `float` if `string` does not have imaginary part, `complex` otherwise.
    """
    if "i" in string:
        if "+" in string:
            return float(string.split("+")[0]) + 1j * \
                float(string.split("+")[1].strip("i"))
        elif "-" in string:
            return float(string.split("-")[0]) + 1j * \
                float(string.split("-")[1].strip("i"))
    else:
        return float(string)


def str2list_space(string):
    """
    Generate `list` from `str`, where elements are separated by a space ''. 
    Used when parsing parameters set in CLI.

    Parameters
    ----------
    string : str
        `str` to be parsed.

    Returns
    -------
    array

    Notes
    -----
    Ranges can be generated as part of the output `array`. For example, 
    `str2list('1,3-5,7')` will give as ouput `array([1,3,4,5,7])`.
    """
    res = np.hstack([np.arange(*(np.array(s.split("-"), dtype=int) + np.array([0, 1])))
                     if "-" in s else np.array([int(s)]) for s in string.split()])
    return res


def str2bool(v1):
    """
    Convert `str` to `bool`.

    Parameter
    ---------
    v1 : str
        String to convert.

    Returns
    -------
    bool

    Raises
    ------
    RuntimeError
        `v1` does not start with 'F', 'f', 'T' nor 't'.
    """
    v = v1.lower().strip('. ')
    if v[0] == "f":
        return False
    elif v[0] == "t":
        return True
    raise RuntimeError(f" unrecognized value of bool parameter :{v1}")


def str_(x):
    """
    Round `x` to 5 floating points and return as `str`.

    Parameters
    ----------
    x : str

    Returns
    -------
    str
    """
    return str(round(x, 5))


def is_round(A, prec=1e-14):
    """
    Returns `True` if all values in A are integers.

    Parameters
    ----------
    A : array
        `array` for which the check should be done.
    prec : float, default=1e-14 (machine precision).
        Threshold to apply.

    Returns
    -------
    bool
        `True` if all elements are integers, `False` otherwise.
    """
    return (np.linalg.norm(A - np.round(A)) < prec)


def short(x, nd=3):
    """
    Format `float` or `complex` number.

    Parameter
    ---------
    x : int, float or complex
        Number to format.
    nd : int, default=3
        Number of decimals.

    Returns
    -------
    str
        Formatted number, with `nd` decimals saved.
    """
    fmt = f"{{0:+.{nd}f}}"
    if abs(x.imag) < 10 ** (-nd):
        return fmt.format(x.real)
    if abs(x.real) < 10 ** (-nd):
        return fmt.format(x.imag) + "j"
    return short(x.real, nd) + short(1j * x.imag)


def split(l):
    """
    Determine symbol used for assignment and split accordingly.

    Parameters
    ---------
    l : str
        Part of a line read from .win file.
    """
    if "=" in l:
        return l.split("=")
    elif ":" in l:
        return l.split(":")
    else:
        return l.split()


def format_matrix(A):
    """
    Format array to print it.

    Parameters
    ----------
    A : array
        Matrix that should be printed.

    Returns
    -------
    str
        Description of the matrix. Ready to be printed.
    """
    return "".join("   ".join(f"{x.real:+5.2f} {x.imag:+5.2f}j" for x in a) + "\n"
                   for a in A)


def log_message(msg, verbosity, level):
    '''
    Logger to decide if a message is printed or not

    Parameters
    ----------
    msg : str
        Message to print
    verbosity : int
        Verbosity set for the current run of the code
    level : int
        If `verbosity >= level`, the message will be printed
    '''

    if verbosity >= level:
        print(msg)


def orthogonalize(A, warning_threshold=np.inf, error_threshold=np.inf, verbosity=1,
                  debug_msg=""):
    """
    Orthogonalize a square matrix, using SVD

    Parameters
    ----------
    A : array( (M,M), dtype=complex)
        Matrix to orthogonalize.
    warning_threshold : float, default=np.inf
        Threshold for warning message. Is some singular values are far from 1
    error_threshold : float, default=np.inf
        Threshold for error message. Is some singular values are far from 1

    Returns
    -------
    array( (M,M), dtype=complex)
        Orthogonalized matrix
    """
    u, s, vh = np.linalg.svd(A)
    msg = f"Matrix is not orthogonal. Singular values {s}!=1 \n {A} \n {debug_msg}"
    if np.any(np.abs(s - 1) > error_threshold):
        raise ValueError(msg)
    elif np.any(np.abs(s - 1) > warning_threshold):
        log_message("Warning: " + msg, verbosity, 1)
    return u @ vh


def sort_vectors(list_of_vectors):
    list_of_vectors = list(list_of_vectors)
    print(list_of_vectors)
    srt = arg_sort_vectors(list_of_vectors)
    print(list_of_vectors, srt)
    return [list_of_vectors[i] for i in srt]


def arg_sort_vectors(list_of_vectors):
    """
    Compare two vectors, 
    First, the longer vector is "larger"
    second, we go element-by-element to compare
    first compare the angle of the complex number (clockwise from the x-axis), then the absolute value

    Returns
    -------
    bool
        True if v1>v2, False otherwise
    """
    if all(np.all(abs(np.array(key).imag) < 1e-4) for key in list_of_vectors):
        def key(x):
            return np.real(x)
    else:
        def key(x):
            return (np.angle(x) / (2 * np.pi) + 0.01) % 1

    def serialize(x, lenmax):
        return [len(x)] + [key(y) for y in x] + [0] * (lenmax - len(x))
    lenmax = max([len(x) for x in list_of_vectors])
    sort_key = [serialize(x, lenmax) for x in list_of_vectors]
    srt = np.lexsort(np.array(sort_key).T, axis=0)
    return srt


def get_borders(E, thresh=1e-5, cyclic=False):
    """
    Get the borders of the blocks of degenerate eigenvalues.

    Parameters
    ----------
    E : array
        Eigenvalues.
    thresh : float, default=1e-5
        Threshold for the difference between eigenvalues.
    cyclic : bool, default=False
        If `True`, the first and last eigenvalues are considered to be close. 
        (e.g. complex eigenvalues on the unit circle).

    Returns
    -------
    array(int)
        Borders of the blocks of degenerate eigenvalues.
    """
    if cyclic:
        return np.where(abs(E - np.roll(E, 1)) > thresh)[0]
    else:
        return np.hstack([
            [0],
            np.where(abs(E[1:] - E[:-1]) > thresh)[0] + 1,
            [len(E)],
        ])


def get_block_indices(E, thresh=1e-5, cyclic=False):
    """
    Get the indices of the blocks of degenerate eigenvalues.

    Parameters
    ----------
    E : array
        Eigenvalues.
    thresh : float, default=1e-5
        Threshold for the difference between eigenvalues.
    cyclic : bool, default=False
        If `True`, the first and last eigenvalues are considered to be close. 
        (e.g. complex eigenvalues on the unit circle).

    Returns
    -------
    array((N,2), dtype=int)
        Indices of the blocks of degenerate eigenvalues.
    """
    borders = get_borders(E, thresh=thresh, cyclic=cyclic)
    if cyclic:
        return np.array([borders, np.roll(borders, -1)]).T
    else:
        return np.array([borders[:-1], borders[1:]]).T


def grid_from_kpoints(kpoints, grid=None, allow_missing=False):
    """
    Given a list of kpoints in fractional coordinates, return a the size of the grid in each direction
    if some k-points are repeated, they are counted only once
    if some k-points are missing, an error is raised

    Parameters
    ----------
    kpoints : np.array((nk, ndim), dtype=float)
        list of kpoints in fractional coordinates

    Returns
    -------
    grid : tuple(int)
        size of the grid in each
    selected_kpoints : list of int
        indices of the selected kpoints

    Raises
    ------
    ValueError
        if some k-points are missing
    """
    if grid is None:
        grid = tuple(np.lcm.reduce([Fraction(k).limit_denominator(100).denominator for k in kp]) for kp in kpoints.T)
    npgrid = np.array(grid)
    print(f"mpgrid = {npgrid}, {len(kpoints)}")
    kpoints_unique = UniqueListMod1()
    selected_kpoints = []
    for i, k in enumerate(kpoints):
        if is_round(k * npgrid, prec=1e-5):
            if k not in kpoints_unique:
                kpoints_unique.append(k)
                selected_kpoints.append(i)
            else:
                warnings.warn(f"k-point {k} is repeated")
    if len(kpoints_unique) < np.prod(grid):
        if not allow_missing:
            raise ValueError(f"Some k-points are missing {len(kpoints_unique)}< {np.prod(grid)}")
    if len(kpoints_unique) > np.prod(grid):
        raise RuntimeError("Some k-points are taken twice - this must be a bug")
    if len(kpoints_unique) < len(kpoints):
        warnings.warn("Some k-points are not on the grid or are repeated")
    return grid, np.array(selected_kpoints, dtype=int)


class UniqueList(list):
    """	
    A list that only allows unique elements.
    uniqueness is determined by the == operator.
    Thus, non-hashable elements are also allowed.
    unlike set, the order of elements is preserved.
    """

    def __init__(self, iterator=[], count=False):
        super().__init__()
        self.do_count = count
        if self.do_count:
            self.counts = []
        for x in iterator:
            self.append(x)

    def append(self, item, count=1):
        for j, i in enumerate(self):
            if i == item:
                if self.do_count:
                    self.counts[self.index(i)] += count
                break
        else:
            super().append(item)
            if self.do_count:
                self.counts.append(1)

    def index(self, value: Any, start=0, stop=sys.maxsize) -> int:
        for i in range(start, stop):
            if self[i] == value:
                return i
        raise ValueError(f"{value} not in list")

    def __contains__(self, item):
        for i in self:
            if i == item:
                return True
        return False

    def remove(self, value: Any, all=False) -> None:
        for i in range(len(self)):
            if self[i] == value:
                if all or not self.do_count:
                    del self[i]
                    del self.counts[i]
                else:
                    self.counts[i] -= 1
                    if self.counts[i] == 0:
                        del self[i]
                        del self.counts[i]
                return


class UniqueListMod1(UniqueList):

    def __init__(self, iterator=[], tol=1e-5):
        self.tol = tol
        self.appended_indices = []
        self.last_try_append = -1
        super().__init__(iterator)

    def append(self, item):
        self.last_try_append += 1
        for i in self:
            if all_close_mod1(i, item, tol=self.tol):
                break
        else:
            list.append(self, item)
            self.appended_indices.append(self.last_try_append)

    def __contains__(self, item):
        for i in self:
            if all_close_mod1(i, item, tol=self.tol):
                return True
        return False

    def index(self, value: Any, start=0, stop=sys.maxsize) -> int:
        stop = min(stop, len(self))
        for i in range(start, stop):
            if all_close_mod1(self[i], value):
                return i
        raise ValueError(f"{value} not in list")


def all_close_mod1(a, b, tol=1e-5):
    """check if two vectors are equal modulo 1"""
    if not np.shape(a) == () and not np.shape(b) == () and (np.shape(a) != np.shape(b)):
        return False
    diff = a - b
    return np.allclose(np.round(diff), diff, atol=tol)


def vector_pprint(vector, fmt=None):
    """
    Format an homogeneous list or array as a vector for printing

    Parameters
    ---------
    vector : array or list
        Vector to format
    fmt : str, default None
        Format of the elements. Numeric types are always sign-padded

    Returns
    -------
    str
        formatted vector string
    """
    if fmt is None:
        fmt = " .5f"
    elif "s" not in fmt:
        fmt = " " + fmt if fmt[0] != " " else fmt
    return ("[" + ("{:{fmt}} " * len(vector)) + "]").format(*vector, fmt=fmt)


def select_irreducible(kpoints, spacegroup):
    """
    Select irreducible k-points from a list of k-points and a space group.

    Parameters
    ----------
    kpoints : array((nk, ndim), dtype=float)
        List of k-points in fractional coordinates.
    spacegroup : SpaceGroup
        Space/point group object containing the symmetries. 

    Returns
    -------
    np.array(int)
        indices of the irreducible k-points in the original list.
    """
    irreducible_list = UniqueListMod1()
    irreducible_list_ik = []
    for ik, kp in enumerate(kpoints):
        for symop in spacegroup.symmetries:
            if symop.transform_k(kp) in irreducible_list:
                break
        else:
            irreducible_list.append(kp)
            irreducible_list_ik.append(ik)
    return np.array(irreducible_list_ik, dtype=int)


def get_mapping_irr(kpoints, kptirr, spacegroup):
    """
    Get a mapping from irreducible k-points to the full list of k-points and vice versa.

    kptirr2kpt[i,isym]=j means that the i-th irreducible kpoint
    is transformed to the j-th kpoint of full grid by the isym-th symmetry operation.
    This is consistent with w90 documentations, but seems to be opposite to what pw2wannier90 does

    Parameters
    ----------
    kpoints : array((nk, ndim), dtype=float)
        List of k-points in fractional coordinates.
    kptirr : array((nirr, ndim), dtype=float)
        List of irreducible k-points in fractional coordinates.
    spacegroup : SpaceGroup
        Space group object containing the symmetries.

    Returns
    -------
    kptirr2kpt: np.array((nirr, nsym), dtype=int)
        Mapping from irreducible k-points to the full list of k-points.
        kptirr2kpt[i,j] is the index of the j-th symmetry operation applied to the i-th irreducible k-point.
    kpt2kptirr: np.array((nk,), dtype=int)
        Mapping from the full list of k-points to the irreducible k-points.
        kpt2kptirr[i] is the index of the irreducible k-point corresponding to the i-th full k-point.
    """
    symmetries = spacegroup.symmetries
    Nsym = len(symmetries)
    kpoints_mod1 = UniqueListMod1(kpoints)
    assert len(kpoints_mod1) == len(kpoints)
    NK = len(kpoints_mod1)
    kptirr2kpt = []
    kpt2kptirr = -np.ones(NK, dtype=int)
    for ikirr, i in enumerate(kptirr):
        k1 = kpoints[i]
        kptirr2kpt.append(np.zeros(Nsym, dtype=int))
        for isym, symop in enumerate(symmetries):
            k1p = symop.transform_k(k1)
            if k1p not in kpoints_mod1:
                raise RuntimeError(f"Symmetry operation {isym} maps k-point {k1} to {k1p} which is outside the grid."
                                   "Maybe the grid is incompatible with the symmetry operations")
            j = kpoints_mod1.index(k1p)
            kptirr2kpt[ikirr][isym] = j
            if kpt2kptirr[j] == -1:
                kpt2kptirr[j] = ikirr
            else:
                assert kpt2kptirr[j] == ikirr, (f"two different irreducible kpoints {ikirr} and {kpt2kptirr[j]} are mapped to t"
                                                f"he same kpoint {j}\n"
                                                f"kptirr= {kptirr}, \nkpt2kptirr= {kpt2kptirr}\n kptirr2kpt= {kptirr2kpt}")
    kptirr2kpt = np.array(kptirr2kpt)
    del kpoints_mod1
    kpt_from_kptirr_isym = get_kpt_from_kptirr_isym(kptirr2kpt=kptirr2kpt, kpt2kptirr=kpt2kptirr)
    assert np.all(kptirr2kpt >= 0)
    assert np.all(kpt2kptirr >= 0)
    return kptirr2kpt, kpt2kptirr, kpt_from_kptirr_isym



def restore_full_grid(kpoints_irr, grid, spacegroup):
    """
    Restore the full grid of k-points from the irreducible k-points and the space group.

    Parameters
    ----------
    kpoints_irr : array((nk, ndim), dtype=float)
        List of irreducible k-points in fractional coordinates.
    grid : tuple(int)
        Size of the grid in each direction.
    spacegroup : SpaceGroup
        Space group object containing the symmetries.

    Returns
    -------
    np.array((nk_grid, ndim), dtype=float)
        Full grid of k-points in fractional coordinates. 
        nk_grid = np.prod(grid), first nk points are irreducible k-points.
    kptirr2kpt: np.array((nirr, nsym), dtype=int)
        Mapping from irreducible k-points to the full list of k-points.
        kptirr2kpt[i,j] is the index of the j-th symmetry operation applied to the i-th irreducible k-point.
    kpt2kptirr: np.array((nk,), dtype=int)
        Mapping from the full list of k-points to the irreducible k-points.
        kpt2kptirr[i] is the index of the irreducible k-point corresponding to the i-th full k-point.   

    Raises
    ------
    ValueError
        If some points on the grid cannot be generated from the irreducible k-points.
    """
    n1, n2, n3 = grid
    all_k_grid = [np.array([i1 / n1, i2 / n2, i3 / n3])
                  for i1 in range(n1)
                  for i2 in range(n2)
                  for i3 in range(n3)]
    all_k_grid_mod1 = UniqueListMod1(all_k_grid, tol=1e-5)
    all_k_mod1 = UniqueListMod1(kpoints_irr, tol=1e-5)
    all_k = [k for k in kpoints_irr]  # first come the irreducible k-points
    assert len(all_k_mod1) == len(kpoints_irr), "kpoints should be unique"
    kpoints_irr_mod1 = UniqueListMod1(kpoints_irr, tol=1e-5)

    kpt2kptirr = -np.ones(len(all_k_grid_mod1), dtype=int)
    kptirr2kpt = -np.ones((len(kpoints_irr_mod1), len(spacegroup.symmetries)), dtype=int)

    for ikirr, kpirr in enumerate(kpoints_irr):
        for isym, symop in enumerate(spacegroup.symmetries):
            transformed_k = symop.transform_k(kpirr)
            if all_close_mod1(transformed_k, kpirr):
                kptirr2kpt[ikirr, isym] = ikirr
                kpt2kptirr[ikirr] = ikirr
            elif transformed_k in kpoints_irr_mod1:
                raise ValueError(f"Symmetry operation {symop} transforms k-point {kpirr} to {transformed_k}, which is already in the list of irreducible k-points.")
            elif transformed_k not in all_k_grid_mod1:
                raise ValueError(f"Symmetry operation {symop} transforms k-point {kpirr} to {transformed_k}, which is not in the grid of k-points.")
            elif transformed_k in all_k_mod1:
                ik = all_k_mod1.index(transformed_k)
                assert kpt2kptirr[ik] == ikirr
                kptirr2kpt[ikirr, isym] = ik
            else:
                ik = len(all_k_mod1)
                assert kpt2kptirr[ik] == -1, f"Two different irreducible k-points {ikirr} and {kpt2kptirr[ik]} are mapped to the same k-point {ik}"
                kpt2kptirr[ik] = ikirr
                kptirr2kpt[ikirr, isym] = ik
                all_k_mod1.append(transformed_k)
                all_k.append(transformed_k)


    kpt_from_kptirr_isym = get_kpt_from_kptirr_isym(kptirr2kpt=kptirr2kpt, kpt2kptirr=kpt2kptirr)

    return np.array(all_k), kptirr2kpt, kpt2kptirr, kpt_from_kptirr_isym


def get_kpt_from_kptirr_isym(kptirr2kpt, kpt2kptirr):
    nsym = kptirr2kpt.shape[1]
    nk = kpt2kptirr.shape[0]
    kpt_from_kptirr_isym = -np.ones(nk, dtype=int)
    for ik, ikirr in enumerate(kpt2kptirr):
        for isym in range(nsym):
            if kptirr2kpt[ikirr, isym] == ik:
                kpt_from_kptirr_isym[ik] = isym
                break
        else:
            raise RuntimeError(f"No Symmetry operation maps irreducible "
                               f"k-point {ikirr} to point {ik}, but kpt2kptirr[{ik}] = {ikirr}.")
    return kpt_from_kptirr_isym


def group_numbers(lst, precision):
    """
    Group numbers that are equal up to a given precision.

    Parameters
    ----------
    lst : list of float
        List of numbers to group.
    precision : float
        Precision for grouping.

    Returns
    -------
    list of float
        numbers that are set to an average value within each group.
    """
    lst = np.array(lst)
    srt = np.argsort(lst)
    lst_sorted = lst[srt]
    groups = []
    groups_ind = []
    current_group = [lst_sorted[0]]
    current_group_ind = [srt[0]]
    for ind, num in zip(srt[1:], lst_sorted[1:]):
        if abs(num - current_group[-1]) < precision:
            current_group.append(num)
            current_group_ind.append(ind)
        else:
            groups.append(current_group)
            groups_ind.append(current_group_ind)
            current_group = [num]
            current_group_ind = [ind]
    groups.append(current_group)
    groups_ind.append(current_group_ind)

    lst_out = np.zeros(len(lst), dtype=float)
    for group_ind, group_num in zip(groups_ind, groups):
        avg_num = np.mean(group_num)
        lst_out[group_ind] = avg_num
    return lst_out


class DummyMethod:
    def __init__(self, message):
        self.message = message

    def __call__(self, *args, **kwargs):
        raise RuntimeError(self.message)


class DummyMethodTables(DummyMethod):

    def __init__(self, method_name):
        message = f"{method_name} requires irreptables package to be installed. " \
            f"Please install irreptables>=3.0.0 via pip: pip install irreptables"
        super().__init__(message)


def clean_np_load(dict):
    """converts zero-dimensional np.ndarray to python scalars"""
    dict_new = {}
    for key, val in dict.items():
        if isinstance(val, np.ndarray) and val.ndim == 0:
            dict_new[key] = val.item()
        else:
            dict_new[key] = val
    return dict_new
