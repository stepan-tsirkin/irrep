
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
##  e-mail: stepan.tsirkin@epfl.ch                               #
##################################################################


from functools import cached_property
import warnings
import numpy as np
import spglib

from irrep.parsers import ParserAbinit, ParserEspresso, ParserVasp, ParserW90

from .symmetry_operation import SymmetryOperation
from .utility import BOHR, clean_np_load, group_numbers, log_message, select_irreducible
from packaging import version
import os

from pyxtal import Group
from sympy import Matrix, sympify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application



class SpaceGroup:

    """
    Class to represent a space-group. Contains methods to describe and print
    info about the space-group.

    Parameters
    ----------
    Lattice : array, shape=(3,3)
        Each row contains cartesian coordinates of a basis vector forming the 
        unit-cell in real space. in Angstroms.
    spinor : bool, default=True
        `True` if wave-functions are spinors (SOC), `False` if they are scalars.
    rotations : list
        Each element is a 3x3 array describing the rotation matrix of a 
        symmetry operation.
    translations : list
        Each element is a 3-element array describing the translational part 
        of a symmetry operation.
    time_reversals : list
        Each element is a boolean indicating whether the symmetry operation 
        is time-reversal or not.
    number : int, default=0
        Number of the space-group. If not specified, it is set to 0.
    name : str, default=""
        Symbol of the space-group in Hermann-Mauguin notation. If not 
        specified, it is set to an empty string.
    spinor_rotations : list, default=None
        Each element is a 2x2 array describing the transformation of the 
        spinor under the symmetry operation. If not specified, it is set to 
        `None`, which means that no spinor rotations are considered.
    symemtry_operations : list, default=None
        Each element is an instance of class `SymmetryOperation` corresponding 
        to a unitary symmetry in the point group of the space-group. If provided,
        it overrides the `rotations`, `translations`, `time_reversals` and
        `spinor_rotations` parameters.
    copy_symops : bool, default=False
        If `True`, the symmetry operations are copied to avoid modifying the 
        original ones. If `False`, the original symmetry operations are used.
        (is symmetry_operations is None, this parameter is ignored)
    translation_mod1 : bool, default=True
        If `True`, the translational part of the symmetry operations is taken modulo 1,
        otherwise it is taken as is. Thi is useful to match with specific consventions (e.g. when reading from the .sym file).

    Attributes
    ----------
    symmetries : list
        Each element is an instance of class `SymmetryOperation` corresponding 
        to a unitary symmetry in the point group of the space-group.
    name : str
        Symbol of the space-group in Hermann-Mauguin notation.
    number_str : str
        String representation of the number of the space-group. 
    spinor : bool
        `True` if wave-functions are spinors (SOC), `False` if they are scalars.
    au_symmetries : list
        Each element is an instance of class `SymmetryOperation` corresponding 
        to an antiunitary symmetry in the point group of the space-group.
    name : str 
        Symbol of the space-group in Hermann-Mauguin notation. 
    number : int 
        Number of the space-group.
    Lattice : array, shape=(3,3) 
        Each row contains cartesian coordinates of a basis vector forming the 
        unit-cell in real space.
    positions : array
        Direct coordinate of sites in the DFT cell setting.
    typat : list
        Indices to identify the element in each atom. Atoms of the same element 
        share the same index.
    refUC : array, default=None
        3x3 array describing the transformation of vectors defining the 
        unit cell to the standard setting.
    shiftUC : array, default=None
        Translation taking the origin of the unit cell used in the DFT 
        calculation to that of the standard setting.
    alat : float (optional)
        Lattice parameter in angstroms (quantum espresso convention).
    magnetic : bool

    Notes
    -----
    The symmetry operations to which elements in the attribute `symmetries` 
    correspond are the operations belonging to the point group of the 
    space-group :math:`G`, i.e. the coset representations :math:`g_i` 
    taking part in the coset decomposition of :math:`G` w.r.t the translation 
    subgroup :math:`T`:
    ..math:: G=T+ g_1 T + g_2 T +...+ g_N T 
    """

    def __init__(self, Lattice, spinor,
                 rotations=None, translations=None,
                 time_reversals=None, translation_mod1=False,
                 symmetry_operations=None,
                 number=None,
                 number_str=None,
                 name="",
                 spinor_rotations=None,
                 copy_symops=False,
                 typat=None,
                 positions=None,
                 magmom=None,
                 magnetic=None,
                 refUC=None,
                 shiftUC=None,
                 alat=None,
                 verbosity=0,
                 ):

        log_message(f"Creating space group {name} with number {number_str}", verbosity, 2)
        if rotations is not None:
            nsym = len(rotations)
        else:
            nsym = len(symmetry_operations)
        log_message(f"Number of symmetries: {nsym}", verbosity, 2)

        self.real_lattice = Lattice
        self.spinor = spinor
        self.name = name
        self.typat = typat
        self.positions = positions
        self.magnetic = magnetic
        self.magmom = magmom
        self.refUC = refUC
        self.shiftUC = shiftUC
        self.alat = alat

        if number_str is not None:
            self.number_str = number_str
            assert number is None, "number and number_str cannot be set at the same time"
        elif number is not None:
            try:
                number = int(number)
            except ValueError:
                raise ValueError(f"number must be an integer (or convertable to int), got <{number}> ({type(number)}) ")

            if number < 0:
                warnings.warn("Negative space group number is not supported. Setting it to -1")
                number = -1
            self.number_str = str(number)
        else:
            self.number_str = "0"

        if symmetry_operations is None:
            copy_symops = False
            symmetry_operations = []
            if spinor_rotations is None:
                spinor_rotations = [None] * len(rotations)

            for i, (rot, trans, tr, srot) in enumerate(zip(rotations,
                                                        translations,
                                                        time_reversals,
                                                        spinor_rotations)):
                symmetry_operations.append(SymmetryOperation(rot=rot,
                                                        trans=trans,
                                                        ind=i + 1,
                                                        Lattice=self.real_lattice,
                                                        time_reversal=tr,
                                                        spinor=self.spinor,
                                                        translation_mod1=translation_mod1,
                                                        spinor_rotation=srot))
        if copy_symops:
            # Copy symmetries to avoid modifying the original ones
            self.symmetries = [s.copy() for s in symmetry_operations]
        else:
            self.symmetries = symmetry_operations



    def as_dict(self):
        """
        return dictionary with info essential about the spacegroup
        """
        keys_dict_optional = ['refUC', 'shiftUC', 'positions', 'typat', 'alat',
                              'magnetic', 'magmom']
        dic = dict(
            Lattice=self.real_lattice,
            spinor=self.spinor,
            rotations=[s.rotation for s in self.symmetries],
            translations=[s.translation for s in self.symmetries],
            spinor_rotations=[s.spinor_rotation for s in self.symmetries],
            time_reversals=[s.time_reversal for s in self.symmetries],
            name=self.name if self.name is not None else "unknown",
            number_str=self.number_str if self.number_str is not None else "0",
        )
        for key in keys_dict_optional:
            if hasattr(self, key):
                val = getattr(self, key)
                if val is not None:
                    dic[key] = val
        return dic

    @classmethod
    def from_dict(cls, dic):
        """
        Create a SpaceGroup object from a dictionary.
        """
        dic_clean = clean_np_load(dic)
        return cls(**dic_clean)

    @classmethod
    def from_npz(cls, file_path):
        """
        Create a SpaceGroup object from a .npz file.
        """
        return cls.from_dict(np.load(file_path, allow_pickle=True))

    def to_npz(self, file_path):
        """
        Save the SpaceGroup object to a .npz file.
        """
        np.savez(file_path, **self.as_dict())

    @property
    def size(self):
        """
        Number of symmetry operations in the space-group.
        """
        return len(self.symmetries)

    def show(self, symmetries=None):
        """
        Print description of space-group and symmetry operations.

        Parameters
        ----------
        symmetries : int, default=None
            Index of symmetry operations whose description will be printed. 
            Run `IrRep` with flag `onlysym` to check the index corresponding 
            to each symmetry operation.
        """

        print('')
        print("\n ---------- CRYSTAL STRUCTURE ---------- \n")
        print('')

        # Print cell vectors in DFT cell only
        print('Cell vectors in angstroms:\n')
        print(f"{'Vectors of DFT cell':^32}")
        for i in range(3):
            vec1 = self.real_lattice[i]
            print(f'a{i:d} = {vec1[0]:7.4f}  {vec1[1]:7.4f}  {vec1[2]:7.4f}  ')
        print()

        print()
        print('\n ---------- SPACE GROUP ----------- \n')
        print()
        print(f'Space group: {self.name} (# {self.number_str})')
        print(f'Number of symmetries: {self.size} (mod. lattice translations)')
        for symop in self.symmetries:
            if symmetries is None or symop.ind in symmetries:
                symop.show(refUC=None, shiftUC=None)

    @property
    def lattice(self):
        return self.real_lattice

    @property
    def Lattice(self):
        return self.real_lattice


    @cached_property
    def lattice_inv(self):
        return np.linalg.inv(self.lattice)

    @cached_property
    def reciprocal_lattice(self):
        return self.lattice_inv.T * (2 * np.pi)


    @classmethod
    def from_cell(
            CLS,
            cell=None,
            real_lattice=None,
            positions=None,
            typat=None,
            spinor=True,
            alat=None,
            from_sym_file=None,
            magmom=None,
            include_TR=True,
            symprec=1e-5,
            mag_symprec=-1,
            angle_tolerance=-1,
            verbosity=0,
            ############
            **kwargs_tables
    ):
        """
        Determine the space-group and save info in attributes. Contains methods to 
        describe and print info about the space-group.

        Parameters
        ----------
        cell : tuple
            Override the `real_lattice`, `positions`, `typat` and `magmom`
            `cell[0]` is a 3x3 array where cartesian coordinates of basis 
            vectors **a**, **b** and **c** are given in rows. 
            `cell[1]` is an array
            where each row contains the direct coordinates of an ion's position. 
            `cell[2]` is an array where each element is a number identifying the 
            atomic species of an ion.
            `cell[3]` (optional) is an array where each element is the magnetic
            moment of an ion.
            See `get_symmetry` in 
            `Spglib <https://spglib.github.io/spglib/python-spglib.html#get-symmetry>`_.
        real_lattice : array( (3, 3), dtype=float)
            3x3 array with cartesian coordinates of basis row-vectors forming the
                unit-cell in real space. 
        positions : array, shape=(Nions, 3), dtype=float
            Each row contains the fractional coordinates of an atom in the unit cell.
        typat : array, shape=(Nions,), dtype=int
            Each element is an integer identifying the atomic species of an ion.
            Atoms of the same element share the same index.
        spinor : bool, default=True
            `True` if wave-functions are spinors (SOC), `False` if they are scalars.
        refUC : array, default=None
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting.
        shiftUC : array, default=None
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting.
        search_cell : bool, default=False
            Whether the transformation to the conventional cell should be computed.
            It is `True` if kpnames was specified in CLI.
        trans_thresh : float, default=1e-5
            Threshold used to compare translational parts of symmetries.
        alat : float, default=None
            Lattice parameter in angstroms (quantum espresso convention).
        from_sym_file : str, default=None
            If provided, the symmetry operations are read from this file.
            (format of pw2wannier90 prefix.sym  file)
        verbosity : int, default=0
            Verbosity level. Default set to minimalistic printing
        magmom : array
            Each element is the magnetic moment of an ion. if None - non-magnetic calculation
            if True - magnetic moments are set to zero, i.e. time-reversal symmetry is included in the spacegroup
        symprec, angle_tolerance, mag_symprec: float
            see `get_symmetry` and 'get_magnetic_symmetry` in 
            `Spglib documentation <https://spglib.readthedocs.io/en/stable/api/python-api.html#spglib.spglib.get_magnetic_symmetry>'__

        """


        if cell is not None:
            assert len(cell) in (3, 4), "cell must be a tuple of length 3"
            assert real_lattice is None, "real_lattice must NOT be provided if cell is given"
            real_lattice = np.array(cell[0])
            assert positions is None, "positions must NOT be provided if cell is given"
            positions = np.array(cell[1])
            assert typat is None, "typat must NOT be provided if cell is given"
            typat = cell[2]
            if len(cell) == 4:
                assert magmom is None, "magmom must NOT be provided if cell is given with 4 elements"
                magmom = cell[3]
        assert real_lattice is not None, "real_lattice must be provided"
        assert positions is not None, "positions must be provided"
        assert typat is not None, "typat must be provided"
        real_lattice = np.array(real_lattice, dtype=float)
        positions = np.array(positions, dtype=float)
        typat = np.array(typat, dtype=int)
        magmom = magmom
        include_TR = include_TR

        if not np.all(isinstance(x, int) for x in typat):
            log_message("typat are not integers -trying to enumerate them", verbosity, 2)
            typeatset = set(typat)
            typeatdict = {t: i + 1 for i, t in enumerate(typeatset)}
            typat = np.array([typeatdict[t] for t in typat], dtype=int)

        if magmom is not None or include_TR:
            magnetic = True
        else:
            magnetic = False

        cell = (real_lattice, positions, typat)
        if not magnetic:  # No magnetic moments magmom = None

            dataset = spglib.get_symmetry_dataset(cell,
                                                  symprec=symprec,
                                                  angle_tolerance=angle_tolerance)
            if version.parse(spglib.__version__) < version.parse('2.5.0'):
                name = dataset['international']
                number_str = str(dataset['number'])
                refUC = dataset['transformation_matrix']
                shiftUC = dataset['origin_shift']
                rotations = dataset['rotations']
                translations = dataset['translations']
            else:
                name = dataset.international
                number_str = str(dataset.number)
                refUC = dataset.transformation_matrix
                shiftUC = dataset.origin_shift
                rotations = dataset.rotations
                translations = dataset.translations
            time_reversal_list = [False] * len(rotations)  # to do: change it to implement grey groups

        else:  # Magnetic group
            if magmom is None or magmom is True:
                magmom = np.zeros((len(positions), 3), dtype=float)
            dataset = spglib.get_magnetic_symmetry_dataset((*cell, magmom),
                                                           symprec=symprec,
                                                           angle_tolerance=angle_tolerance,
                                                           mag_symprec=mag_symprec)
            if dataset is None:
                raise ValueError("No magnetic space group could be detected!")
            rotations = dataset.rotations
            translations = dataset.translations
            time_reversal_list = dataset.time_reversals
            refUC = dataset.transformation_matrix
            shiftUC = dataset.origin_shift

            uni_number = dataset.uni_number
            root = os.path.dirname(__file__)
            with open(root + "/data/msg_numbers.data", 'r') as f:
                number_str, name = f.readlines()[uni_number].strip().split(" ")

        # Read syms from .sym file (useful for Wannier interface)
        if from_sym_file is not None:
            assert alat is not None, "Lattice parameter must be provided to read symmetries from file"
            rot_cart, trans_cart = read_sym_file(from_sym_file)
            rotations, translations = cart_to_crystal(rot_cart,
                                                      trans_cart,
                                                      real_lattice,
                                                      alat)
            translation_mod1 = False
        else:
            translation_mod1 = True

        if not include_TR:
            log_message("Removing time-reversal symmetries", verbosity, 2)
            log_message(f"Number of time-reversal symmetries: {sum(time_reversal_list)}", verbosity, 2)
            selected_symmetries = np.where(np.logical_not(time_reversal_list))[0]
            log_message(f"Selected symmetries: {selected_symmetries}", verbosity, 2)
            # Remove time-reversal symmetries
            rotations = [rotations[i] for i in selected_symmetries]
            translations = [translations[i] for i in selected_symmetries]
            time_reversal_list = [time_reversal_list[i] for i in selected_symmetries]



        #         symmetries.append(SymmetryOperation(
        #             rotations[isym],
        #             translations[isym],
        #             real_lattice,
        #             ind=isym + 1,
        #             spinor=spinor,
        #             translation_mod1=translation_mod_1,
        #             time_reversal=time_reversal_list[isym]))

        sg = CLS(
            Lattice=real_lattice,
            spinor=spinor,
            rotations=rotations,
            translations=translations,
            time_reversals=time_reversal_list,
            translation_mod1=translation_mod1,
            number_str=number_str,
            name=name,
            spinor_rotations=None,  # No SOC in this class
            refUC=refUC,
            shiftUC=shiftUC,
            magnetic=magnetic,
            positions=positions,
            typat=typat,
            alat=alat,
            magmom=magmom,
            verbosity=verbosity,
        )
        if CLS.__name__ == "SpaceGroupIrreps":
            sg.set_irreptables(verbosity=verbosity,
                               **kwargs_tables,)
        return sg

    @property
    def number(self):
        '''
        To get the number of the space group as an int. Used in WannierBerri. 
        Returns -1 for magnetic spacegroups
        '''
        numbers = self.number_str.split('.')
        if len(numbers) > 1:  # magnetic SG
            return -1
        else:
            return int(numbers[0])

    @property
    def u_symmetries(self):
        '''
        List of unitary symmetries
        '''
        return [x for x in self.symmetries if not x.time_reversal]

    @property
    def au_symmetries(self):
        '''
        List of antiunitary symmetries
        '''
        if self.magnetic:
            return [x for x in self.symmetries if x.time_reversal]
        else:
            return []

    def is_grid_symmetrical(self, nk):
        nk = np.array(nk, dtype=int)
        basis = np.diag(1. / nk)
        for symop in self.symmetries:
            basis_rot = symop.transform_k(basis)
            A = basis_rot @ np.diag(nk)
            Aint = np.round(A).astype(int)
            if not np.allclose(A, Aint):
                return False
            if not np.allclose(abs(np.linalg.det(A)), 1):
                return False
        return True

    def get_irreducible_kpoints_grid(self, grid, allow_asymmetric=False):
        """Get irreducible k-points from a Monkhorst-Pack grid and space group

        Parameters
        ----------
        grid : array-like
            Monkhorst-Pack grid dimensions, e.g. [6, 6, 4]
        spacegroup : irrep.SpaceGroup
            the space group of the crystal

        Returns
        -------
        np.array
            Array of irreducible k-points in fractional coordinates
        """

        is_symmetric = self.is_grid_symmetrical(grid)
        if not is_symmetric:
            msg = f"The grid {grid} is not symmetric with respect to the space group {self.number_str}. "
            if allow_asymmetric:
                warnings.warn(msg + "But allow_asymmetric=True, so continuing anyway.")
            else:
                raise ValueError(msg)
        kp1 = np.linspace(0, 1, grid[0], endpoint=False)
        kp2 = np.linspace(0, 1, grid[1], endpoint=False)
        kp3 = np.linspace(0, 1, grid[2], endpoint=False)
        kpoints = np.array(np.meshgrid(kp1, kp2, kp3, indexing='ij')).reshape(3, -1).T
        irr_ik = select_irreducible(kpoints, self)
        return kpoints[irr_ik]


    def write_sym_file(self, filename, alat=None):
        """
        Write symmetry operations to a file.

        Parameters
        ----------
        filename : str
            Name of the file.
        alat : float, default=None
            Lattice parameter in angstroms. If not specified, the lattice 
            parameter is not written to the file.
        """

        if alat is None:
            if hasattr(self, 'alat'):
                alat = self.alat
        if alat is None:
            warnings.warn("Lattice parameter not specified. Symmetry operations will be written assuming A=1")
            alat = 1
        with open(filename, "w") as f:
            f.write(f" {len(self.symmetries)} \n")
            for symop in self.symmetries:
                f.write(symop.str_sym(alat))


    def get_product_table(self, get_diff=False):
        """
        Compute the product table of the space-group.

        Parameters
        ----------
        get_diff : bool, default=False
            If `True`, the difference of the translational parts of the 
            product and the resulting symmetry operation is also returned.
            If `False`, only the product table is returned.
            Also returns the spinor 

        Returns
        -------
        np.array(Nsym, Nsym), dtype=int
            Product table of the space-group. The element (i,j) is the index 
            of the symmetry operation corresponding to the product of symmetry 
            operations i and j.

        np.array(Nsym, Nsym, 3), dtype=int
            (only if `get_diff` is `True`) Difference of the translational parts 
            of the product and the resulting symmetry operation. The element 
            (i,j) is the difference for the product of symmetry operations i 
            and j.
        np.array(Nsym, Nsym), dtype=int
            (only if `get_diff` is `True` and `self.spinor` is `True`) Spinor 
            factor for the product of symmetry operations i and j. The element 
            (i,j) is 1 if the spinor rotation of the product matches that of 
            the resulting symmetry operation, -1 if it is opposite
        """

        nsym = len(self.symmetries)
        prod_table = np.zeros((nsym, nsym), dtype=int)
        if get_diff:
            transl_diff = np.zeros((nsym, nsym, 3), dtype=float)
            if self.spinor:
                spinor_factors = np.zeros((nsym, nsym), dtype=int)
            else:
                spinor_factors = np.ones((nsym, nsym), dtype=int)
        for i, sym_i in enumerate(self.symmetries):
            for j, sym_j in enumerate(self.symmetries):
                sym_ij = sym_i.multiply_keeptransl(sym_j, mod1=not get_diff)
                for k, sym_k in enumerate(self.symmetries):
                    if sym_ij.equals(sym_k, mod1=True):
                        prod_table[i, j] = k
                        if get_diff:
                            transl_diff[i, j] = np.round(sym_ij.translation - sym_k.translation).astype(int)
                            if self.spinor:
                                if np.allclose(sym_ij.spinor_rotation, sym_k.spinor_rotation):
                                    spinor_factors[i, j] = 1
                                elif np.allclose(sym_ij.spinor_rotation, -sym_k.spinor_rotation):
                                    spinor_factors[i, j] = -1
                                else:
                                    raise ValueError(f"Spinor rotations do not match for product {i}x{j}={k}: \n"
                                                     f"the product of spinor rotations: \n {sym_ij.spinor_rotation}\n"
                                                     f"does not match the spinor rotation of the resulting symmetry: \n {sym_k.spinor_rotation}")

                        break
                else:
                    raise ValueError(f"Product of symmetries {i} and {j} not found in the list of symmetries" +
                                     f" sym_i: {sym_i.str()},\n sym_j: {sym_j.str()}, \nsym_ij: {sym_ij.str()}")
        if get_diff:
            transl_diff_round = np.round(transl_diff).astype(int)
            assert np.allclose(transl_diff, transl_diff_round), f"Translations differences are not integers: {transl_diff}"
            return prod_table, transl_diff_round, spinor_factors
        else:
            return prod_table

    def get_identity_operation(self):
        for sym in self.symmetries:
            if sym.is_identity:
                return sym
        raise ValueError("Identity symmetry not found in the list of symmetries")

    def get_identity_index(self):
        for i, sym in enumerate(self.symmetries):
            if sym.is_identity:
                return i
        raise ValueError("Identity symmetry not found in the list of symmetries")

    def get_inverse_table(self):
        """
        Compute the inverse table of the space-group.

        Returns
        -------
        np.array(Nsym,), dtype=int
            Inverse table of the space-group. The element i is the index 
            of the symmetry operation corresponding to the inverse of symmetry 
            operation i.
        """

        nsym = len(self.symmetries)
        inv_table = np.zeros((nsym,), dtype=int)
        for i, sym_i in enumerate(self.symmetries):
            sym_i_inv = sym_i.inverse()
            for k, sym_k in enumerate(self.symmetries):
                if sym_i_inv.equals(sym_k, mod1=True):
                    inv_table[i] = k
                    break
            else:
                raise ValueError(f"Inverse of symmetry {sym_i.ind} not found in the list of symmetries")
        return inv_table

    @cached_property
    def translations_cart(self):
        return np.array([symop.translation @ self.real_lattice for symop in self.symmetries])

    @classmethod
    def from_espresso(CLS,
                      prefix,
                      spinor=None,
                      alat=None,
                      from_sym_file=None,
                      verbosity=0,
                      **kwargs_cell
                      ):
        """
        Parse files to create a space-group object from Quantum ESPRESSO output.

        Parameters
        ----------
        prefix : str
            Prefix of the Quantum ESPRESSO output files (e.g. 'prefix' for 'prefix.save').
        spinor : bool
            `True` if wave-functions are spinors (SOC), `False` if they are scalars.
            If not specified, it is determined from the code.
        alat : float, default=None
            Lattice parameter in angstroms (quantum espresso convention).
        from_sym_file : str
            If provided, the symmetry operations are read from this file.
            (format of pw2wannier90 prefix.sym  file)
        verbosity : int
            Verbosity level. Default set to minimalistic printing.
        **kwargs_cell : dict
            Additional keyword arguments to pass to the 'SpaceGroup.from_cell()` and `SpaceGroupIrrep.set_irreptables` method.

        Returns
        -------
        SpaceGroup
            An instance of the `SpaceGroup` class with the parsed symmetry operations.
        """
        parser = ParserEspresso(prefix)
        _spinor = parser.spinor
        # alat is saved to be used to write the prefix.sym file
        Lattice, positions, typat, _alat = parser.parse_lattice()
        if alat is None:
            alat = _alat
        return CLS.from_cell(
            real_lattice=Lattice,
            positions=positions,
            typat=typat,
            spinor=spinor if spinor is not None else _spinor,
            alat=alat,
            from_sym_file=from_sym_file,
            verbosity=verbosity,
            **kwargs_cell
        )

    @classmethod
    def from_vasp(
        CLS,
        fWAV=None,
        fPOS=None,
        verbosity=0,
        spinor=None,
        **kwargs_cell
    ):
        """
        Parse files to create a space-group object.
        Parameters
        ----------
        fWAV : str, default=None
            Name of the file with wavefunctions (e.g. WAVECAR for VASP).
        fWFK : str, default=None
            Name of the file with wavefunctions (e.g. WFK for ABINIT).
        fPOS : str, default=None
            Name of the file with positions (e.g. POSCAR for VASP).
        spinor : bool, default=None
            `True` if wave-functions are spinors (SOC), `False` if they are scalars.
            If not specified, it is determined from the code.
        verbosity : int, default=0
            Verbosity level. Default set to minimalistic printing.
        **kwargs_cell : dict
            Additional keyword arguments to pass to the 'SpaceGroup.from_cell()` and `SpaceGroupIrrep.set_irreptables` method.

        Returns
        -------
        SpaceGroup
            An instance of the `SpaceGroup` class with the parsed symmetry operations.
        """
        if spinor is None:
            log_message("Spinor is not specified (for VASP), assuming non-spinor calculation", verbosity, 2)
            spinor = False
        parser = ParserVasp(fPOS, fWAV, onlysym=True, verbosity=verbosity)
        Lattice, positions, typat = parser.parse_poscar()

        return CLS.from_cell(
            real_lattice=Lattice,
            positions=positions,
            typat=typat,
            spinor=spinor,
            verbosity=verbosity,
            **kwargs_cell
        )


    @classmethod
    def from_abinit(CLS,
                    fWFK=None,
                    verbosity=0,
                    spinor=None,
                    **kwargs_cell):
        """
        Parse files to create a space-group object from ABINIT output.
        Parameters
        ----------
        fWFK : str, default=None
            Name of the file with wavefunctions (e.g. WFK for ABINIT).
        verbosity : int, default=0
            Verbosity level. Default set to minimalistic printing.
        spinor : bool, default=None
            `True` if wave-functions are spinors (SOC), `False` if they are scalars.
            If not specified, it is determined from the code.
        **kwargs_cell : dict
            Additional keyword arguments to pass to the 'SpaceGroup.from_cell()` and `SpaceGroupIrrep.set_irreptables` method.

        Returns
        -------
        SpaceGroup
            An instance of the `SpaceGroup` class with the parsed symmetry operations.
        """
        parser = ParserAbinit(fWFK)
        (nband, NK, Lattice, Ecut0, _spinor, typat, positions, EF_in) = \
            parser.parse_header(verbosity=verbosity)
        return CLS.from_cell(
            real_lattice=Lattice,
            positions=positions,
            typat=typat,
            spinor=spinor if spinor is not None else _spinor,
            verbosity=verbosity,
            **kwargs_cell
        )


    @classmethod
    def from_wannier90(CLS,
                       prefix,
                       spinor=None,
                       verbosity=0,
                       **kwargs_cell):
        """
        Parse files to create a space-group object from Wannier90 output.
        Parameters
        ----------
        prefix : str
            Prefix of the Wannier90 output files (e.g. 'prefix' for 'prefix.win').
        spinor : bool, default=None
            `True` if wave-functions are spinors (SOC), `False` if they are scalars.
            If not specified, it is determined from the code.
        verbosity : int, default=0
            Verbosity level. Default set to minimalistic printing.
        **kwargs_cell : dict        
            Additional keyword arguments to pass to the 'SpaceGroup.from_cell()` and `SpaceGroupIrrep.set_irreptables` method.
        Returns
        -------
        SpaceGroup
            An instance of the `SpaceGroup` class with the parsed symmetry operations.
        """
        parser = ParserW90(prefix, unk_formatted=None)
        NK, NBin, _spinor, EF_in = parser.parse_header()
        Lattice, positions, typat, kpred = parser.parse_lattice()
        return CLS.from_cell(
            real_lattice=Lattice,
            positions=positions,
            typat=typat,
            spinor=spinor if spinor is not None else _spinor,
            verbosity=verbosity,
            **kwargs_cell
        )


    def set_gpaw(self, calculator):
        for sym in self.symmetries:
            sym.set_gpaw(calculator)

    @classmethod
    def from_gpaw(cls,
                  calculator,
                  mag_symprec=0.05,
                  typat=None,
                  magmom=None,
                  magnetic=True,
                  spinor=None,
                  verbosity=0,
                  **kwargs_cell):
        """Get the spacegroup of a GPAW calculator (non-spinor only).
        Parameters
        ----------
        calculator : GPAW
            The GPAW calculator.
        mag_symprec : float
            The precision for distinguishing different magnetic moments.
        symprec : float
            The symmetry precision for spacegroup detection.
        include_TR : bool
            Whether to include time-reversal symmetry.
        typat : list of int, optional
            The typat to use for spacegroup detection. If None, the atomic numbers are used,
            and if magmom is also None, the magnetic moments are used to distinguish different types of atoms.
            The magnetic moments are rounded to the nearest integer and mapped to consecutive integers starting from 1.
            For example, if the magnetic moments are [2.1, -2.1, 0.0, 2.9], they are rounded to [2, -2, 0, 3],
            and then mapped to [2, 1, 0, 3] (since -2 is the smallest unique value, it is mapped to 1).
            The final typat will be atomic_number*1000 + mapped_magnetic_moment.
        magmom : list of float, optional
            The magnetic moments to use for spacegroup detection. If None, the magnetic moments from the calculator are used.
        Returns
        -------
        spacegroup : irrep.spacegroup.SpaceGroup
            The detected spacegroup.
        """
        lattice = calculator.atoms.cell
        if typat is None:
            typat = calculator.atoms.get_atomic_numbers()
            if magnetic:
                if magmom is None:
                    magmom = calculator.get_magnetic_moments()
                    assert magmom.shape == (len(calculator.atoms),)
            else:
                magmom = np.zeros(len(calculator.atoms), dtype=float)
            magmom = group_numbers(magmom, precision=mag_symprec)
            magmom_set = set(magmom)
            if len(magmom_set) > 1:
                magmom_map = {m: i + 1 for i, m in enumerate(sorted(magmom_set))}
                typat = [int(typat[i] * 1000 + magmom_map[mag]) for i, mag in enumerate(magmom)]
        else:
            assert len(typat) == len(calculator.atoms), "typat should have the same length as the number of atoms"
        log_message(f"typat used for spacegroup detection (accounting magmom): {typat}", verbosity, 1)
        positions = calculator.atoms.get_scaled_positions()
        return cls.from_cell(real_lattice=lattice,
                             positions=positions,
                             typat=typat,
                             spinor=spinor if spinor is not None else False,
                             **kwargs_cell
                                )

    # @classmethod
    # def from_gpaw_magnetic(cls, calculator, theta=0, phi=0, include_TR=True,
    #                        magmom=None, magmoms=None,
    #                        mag_symprec=0.05, symprec=1e-5,
    #                        ):
    #     """
    #     Get the spacegroup of a GPAW calculator (spinor magnetic version).

    #     Parameters:
    #     calculator : GPAW
    #         The GPAW calculator.
    #     theta, phi : float
    #         The angles defining the global spin quantization axis. (in radians)
    #         magnetic moments will be aligned along that direction (in positive or negative direction, as defined from dft calculation)
    #     include_TR : bool
    #         Whether to include time-reversal symmetry.
    #     magmom : array( Nat, 3), optional
    #         if provided, overrides the magnetic moments of the calculator
    #     magmoms : array(Nat,), optional
    #         The older argument for magmom, kept for backward compatibility. Used only if magmom is None.
    #     mag_symprec : float
    #         The precision for distinguishing different magnetic moments. Only used if magmom is None.
    #     symprec : float
    #         The symmetry precision for spacegroup detection.
    #     """
    #     lattice = calculator.atoms.cell
    #     typat = calculator.atoms.get_atomic_numbers()
    #     positions = calculator.atoms.get_scaled_positions()
    #     if magmom is None:
    #         magmom = magmoms
    #     if magmom is None:
    #         magmom_axis = calculator.get_magnetic_moments()
    #         magmom_axis = group_numbers(magmom_axis, precision=mag_symprec)
    #         axis = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    #         magmom = magmom_axis[:, None] * axis[None, :]
    #     print(f"using magmom \n {magmom}")
    #     return SpaceGroup.from_cell(real_lattice=lattice,
    #                             positions=positions,
    #                             typat=typat,
    #                             spinor=True,
    #                             include_TR=include_TR,
    #                             symprec=symprec,
    #                             mag_symprec=mag_symprec,
    #                             magmom=magmom)

    @classmethod
    def get_trivial(cls, lattice, spinor=False, time_reversal=False):
        if not time_reversal:
            return SpaceGroup(Lattice=lattice, spinor=spinor, rotations=[np.eye(3)], translations=[np.zeros(3)], time_reversals=[False], number=1,
                              name="trivial", spinor_rotations=[np.eye(2)])
        else:
            return SpaceGroup(Lattice=lattice, spinor=spinor, rotations=[np.eye(3)] * 2, translations=[np.zeros(3)], time_reversals=[False, True], number=1,
                              name="trivial+TR", spinor_rotations=[np.eye(2)] * 2)


    def equals(self, other, tol=1e-5, mod1=True):
        if self.size != other.size:
            return False
        if not np.allclose(self.real_lattice, other.real_lattice, atol=tol):
            return False
        if not self.spinor == other.spinor:
            return False
        if self.magnetic != other.magnetic:
            return False
        # Check magnetic moments
        if self.magnetic:
            if not hasattr(self, 'positions') or not hasattr(other, 'positions'):
                raise ValueError("Cannot compare magnetic spacegroups without positions")
            if not np.allclose(self.positions, other.positions, atol=tol):
                return False
            if not np.allclose(self.typat, other.typat, atol=tol):
                return False
        # Check symmetries
        for isym in range(len(self.symmetries)):
            if not self.symmetries[isym].equals(other.symmetries[isym], tol=tol, mod1=mod1):
                return False
        return True

    def conventional_wyckoff_positions(spacegroup_num):
        """
            Returns an array of all Wyckoff Positions in string symbolic format
            of a given space group of symmetries
            Args:
                spacegroup_num: int of the spacegroup number
        """
        group = Group(spacegroup_num, style="spglib")
        conventional_wyckoff_poss = []

        for position in group.Wyckoff_positions:
            op = position.ops[0]
            coordinate = op.as_xyz_str()
            coordinate = coordinate.replace(' ', '')
            conventional_wyckoff_poss.append(coordinate)


        return conventional_wyckoff_poss


    def wyckoff_positions(cell_structure):
        """
            Given a cell structure, returns all the wyckoff positions associated to its
            symmetry group, in the lattice vector's basis. As symbolic strings

            Args:
                cell_structure: 3d tuple with
                    1- numerical array of lattice vectors [[a1,a2,a3]]
                    2- numerical array of atoms positions [[r1,r2]]
                    3- numerical array with atomical numbers in the same order of their
                        position [11,17]
        """


        symmetry_dataset = spglib.get_symmetry_dataset(cell_structure)
        group_number = symmetry_dataset.number

        conventional_wyckoffs = SpaceGroup.conventional_wyckoff_positions(group_number)

        # Makes SymPy to execute a simbolic operation
        inverse_matrix = np.linalg.inv(symmetry_dataset.transformation_matrix)

        inv_transformation_matrix = Matrix(inverse_matrix).applyfunc(sympify)
        origin_shift = Matrix(symmetry_dataset.origin_shift).applyfunc(sympify)

        wyckoffs_in_cell_base = []

        for sim_vector in conventional_wyckoffs:

            # sim_vector is a string on array shape change to numeric or simbolic
            _TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
            vector = Matrix([parse_expr(v, transformations=_TRANSFORMS) for v in sim_vector.split(',')])
            wyckoff_new_base = inv_transformation_matrix * (vector - origin_shift)

            # Change to string again
            parts = [str(e).replace(" ", "") for e in wyckoff_new_base]
            wyckoff_new_base_str = ",".join(parts)

            wyckoffs_in_cell_base.append(wyckoff_new_base_str)

        return wyckoffs_in_cell_base


    def get_transformed_group_index(self, symmetry_operation):
        """
        calculates the operations S^-1*g*S for a given symmetry operation S and each symmetry operation g in the group. Returns the index of the transformed operation in the group.
        Also returns the translation difference between the transformed operation and the corresponding operation in the group.
        """
        transformed_group_index = []
        translation_difference = []
        symmetry_operation_inv = symmetry_operation.inverse(mod1=False)
        for sym in self.symmetries:
            transformed_sym = symmetry_operation_inv.multiply_keeptransl(sym).multiply_keeptransl(symmetry_operation)
            sym.show()
            transformed_sym.show()
            for i, sym2 in enumerate(self.symmetries):
                if transformed_sym.equals(sym2, mod1=True) and transformed_sym.time_reversal == sym2.time_reversal:
                    transl_diff = transformed_sym.translation - sym2.translation
                    transl_diff_int = np.round(transl_diff).astype(int)
                    if np.allclose(transl_diff, transl_diff_int):
                        transformed_group_index.append(i)
                        translation_difference.append(transl_diff_int)
                        break
            else:
                raise ValueError(f"Transformed symmetry operation {transformed_sym.str()} not found in the group")
        return np.array(transformed_group_index), np.array(translation_difference).astype(int)



def read_sym_file(fname):
    """
    Read symmetry operations from a file.

    Parameters
    ----------
    fname : str
        Name of the file.

    Returns
    -------
    np.array(Nsym, 3, 3)
        Each element is a 3x3 array describing a rotation matrix in cartesian coordinates
    np.array(Nsym, 3)
        Each element is a 3D vector describing a translation in units of alat.
    """

    with open(fname, "r") as f:
        lines = f.readlines()
    lines = [line.split() for line in lines]
    lines = [line for line in lines if len(line) > 0]
    nsym = int(lines[0][0])
    assert len(lines) == 1 + 4 * nsym
    RT = np.array(lines[1:], dtype=float).reshape(nsym, 4, 3)
    rotations = RT[:, 0:3]  # .swapaxes(1, 2)
    translations = RT[:, 3]
    return rotations, translations


def cart_to_crystal(rot_cart, trans_cart, lattice, alat):
    """
    Convert rotation and translation matrices from cartesian to crystal 
    coordinates.

    Parameters
    ----------
    rot_cart : array, shape=(Nsym, 3, 3)
        Each element is a 3x3 array describing a rotation matrix in cartesian
        coordinates
    trans_cart : array, shape=(Nsym, 3)
        Each element is a 3D vector describing a translation in cartesian
        coordinates
    lattice : array, shape=(3, 3)
        Each row contains cartesian coordinates of a basis vector forming the 
        unit-cell in real space.
    alat : float, default=1
        Lattice parameter in angstroms (quantum espresso convention).

    Returns
    -------
    array, shape=(Nsym, 3, 3)
        Each element is a 3x3 array describing a rotation matrix in crystal 
        coordinates (should be integers)
    array, shape=(Nsym, 3)
        Each element is a 3D vector describing a translation in crystal 
        coordinates
    """

    lat_inv = np.linalg.inv(lattice)
    rot_crystal = np.array([(lat_inv.T @ rot @  lattice.T) for rot in rot_cart])
    assert np.allclose(rot_crystal, np.round(rot_crystal)), f"rotations are not integers in crystal coordinates : {rot_crystal}"
    rot_crystal = np.round(rot_crystal).astype(int)
    trans_crystal = - trans_cart @ lat_inv * alat * BOHR
    return rot_crystal, trans_crystal


# For compatibility with old code
SpaceGroupBare = SpaceGroup
