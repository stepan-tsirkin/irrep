
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
##  Written by Stepan Tsirkin, University of Zurich.             #
##  e-mail: stepan.tsirkin@physik.uzh.ch                         #
##################################################################


from functools import cached_property
import warnings
import numpy as np
from math import pi
from scipy.linalg import expm
import spglib
from irreptables import IrrepTable
from scipy.optimize import minimize
from .utility import str_, log_message, BOHR
from packaging import version

pauli_sigma = np.array(
    [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])


class SymmetryOperation():
    """
    Contains information to describe a symmetry operation and methods to get 
    info about the symmetry, transform it to a reference choice of unit cell 
    and print a description of it.

    Parameters
    ----------
    rotation : array, shape=(3,3)
        Matrix describing the tranformation of basis vectors of the unit cell 
        under the symmetry operation.
    translation : array, shape=(3,)
        Translational part of the symmetry operation, in terms of the basis 
        vectors of the unit cell.
    Lattice : array, shape=(3,3) 
        Each row contains cartesian coordinates of a basis vector forming the 
        unit-cell in real space.
    ind : int, default=-1
        Index of the symmetry operation.
    spinor : bool, default=true
        `True` if wave-functions are spinors, `False` if they are scalars.
    translation_mod1 : bool, default=True
        If `True`, the translation part of the symmetry operation is taken
        modulo 1. Otherwise, it is taken as it is

    Attributes
    ---------
    ind : int
        Index of the symmetry operation.
    rotation : array, shape=(3,3)
        Matrix describing the tranformation of basis vectors of the unit cell 
        under the symmetry operation.
    translation : array, shape=(3,)
        Translational part of the symmetry operation, in terms of the basis 
        vectors of the unit cell.
    Lattice : array, shape=(3,3) 
        Each row contains cartesian coordinates of a basis vector forming the 
        unit-cell in real space.
    axis : array, shape=(3,)
        Rotation axis of the symmetry.
    angle : float
        Rotation angle of the symmmetry, in radians.
    inversion : bool
        `False` if the symmetry preserves handedness (identity, rotation, 
        translation or screw rotation), `True` otherwise (inversion, reflection 
        roto-inversion or glide reflection).
    angle_str : str
        String describing the rotation angle in radians.
    spinor : bool
        `True` if wave-functions are spinors, `False` if they are scalars.
    spinor_rotation : array, shape=(2,2)
        Matrix describing how spinors transform under the symmetry.
    sign : float
        Factor needed to match the matrix for the rotation of spinors 
        to that in tables.
    """

    def __init__(self, rot, trans, Lattice, ind=-1, spinor=True, 
                 translation_mod1=True):
        self.ind = ind
        self.rotation = rot
        self.Lattice = Lattice
        self.translation_mod1 = translation_mod1
        self.translation = self.get_transl_mod1(trans)
        self.axis, self.angle, self.inversion = self._get_operation_type()
        iangle = (round(self.angle / pi * 6) + 6) % 12 - 6
        if iangle == -6:
            iangle = 6
        self.angle = iangle * pi / 6
        self.angle_str = self.get_angle_str()
        self.spinor = spinor
        self.spinor_rotation = expm(-0.5j * self.angle *
                                    np.einsum('i,ijk->jk', self.axis, pauli_sigma))
        self.sign = 1  # May be changed later externally

    def get_transl_mod1(self, t):
        """
        Take translation modulo 1 if needed.
        governed by the translation_mod1 attribute.
        """    
        if self.translation_mod1:
            t = t%1 
            t[1 - t < 1e-5] = 0
            return t
        else:
            return t

    def get_angle_str(self):
        """
        Give str of rotation angle.

        Returns
        -------
        str
            Rotation angle in radians.
        
        Raises
        ------
        RuntimeError
            Angle does not belong to 1, 2, 3, 4 or 6-fold rotation.
        """
        accur = 1e-4
        def is_close_int(x): return abs((x + 0.5) % 1 - 0.5) < accur
        api = self.angle / np.pi
        if abs(api) < 0.01:
            return " 0 "
        for n in 1, 2, 3, 4, 6:
            if is_close_int(api * n):
                return "{0:.0f}{1} pi".format(
                    round(api * n), "" if n == 1 else "/" + str(n))
        raise RuntimeError(
            "{0} pi rotation cannot be in the space group".format(api))

    def _get_operation_type(self):
        """
        Calculates the rotation axis and angle of the symmetry and if it 
        preserves handedness or not.

        Returns
        -------
        tuple
            The first element is an array describing the rotation axis. The 
            second element describes the rotation angle. The third element is a 
            boolean, `True` if the symmetry preserves handedness 
            (determinant -1).
        """
        rotxyz = self.Lattice.T.dot(
            self.rotation).dot(
            np.linalg.inv(
                self.Lattice).T)
#        print ("rotation in real space:\n",rotxyz)
        E, V = np.linalg.eig(rotxyz)
        if not np.isclose(abs(E), 1).all():
            raise RuntimeError(
                "some eigenvalues of the rotation are not unitary")
        if E.prod() < 0:
            inversion = True
            E *= -1
        else:
            inversion = False
        idx = np.argsort(E.real)
        E = E[idx]
        V = V[:, idx]
        axis = V[:, 2].real
        if np.isclose(E[:2], 1).all():
            angle = 0
        elif np.isclose(E[:2], -1).all():
            angle = np.pi
        else:
            angle = np.angle(E[0])
            v = V[:, 0]
            s = np.real(np.linalg.det([v, v.conj(), axis]) / 1.j)
            if np.isclose(s, -1):
                angle = 2 * np.pi - angle
            elif not np.isclose(s, 1):
                raise RuntimeError("the sign of rotation should be +-1")
        return (axis, angle, inversion)

    def rotation_refUC(self, refUC):
        """
        Calculate the matrix of the symmetry in the reference cell choice.
        
        Parameters
        ----------
        refUC : array
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting.

        Returns
        -------
        R1 : array, shape=(3,3)
            Matrix for the transformation of basis vectors forming the 
            reference unit cell.

        Raises
        ------
        RuntimeError
            If the matrix contains non-integer elements after the transformation.
        """
        R = np.linalg.inv(refUC).dot(self.rotation).dot(refUC)
        R1 = np.array(R.round(), dtype=int)
        if (abs(R - R1).max() > 1e-6):
            raise RuntimeError(
                "the rotation in the reference UC is not integer. Is that OK? \n{0}".format(R))
        return R1

    def translation_refUC(self, refUC, shiftUC):
        """
        Calculate translation in reference choice of unit cell.

        Parameters
        ----------
        refUC : array
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting.
        shiftUC : array
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting.

        Returns
        -------
        array
            Translation in reference choice of unit cell.
        """
        t_ref =  - shiftUC + self.translation + self.rotation.dot(shiftUC)
        t_ref = np.linalg.inv(refUC).dot(t_ref)
        return t_ref


    def show(self, refUC=np.eye(3), shiftUC=np.zeros(3)):
        """
        Print description of symmetry operation.
        
        Parameters
        ----------
        refUC : array, default=np.eye(3)
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting.
        shiftUC : array, default=np.zeros(3)
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting.
        """

        def parse_row_transform(mrow):
            s = ""
            coord = ["kx","ky","kz"]
            is_first = True
            for i in range(len(mrow)):
                b = int(mrow[i]) if np.isclose(mrow[i],int(mrow[i])) else mrow[i]
                if b == 0:
                    continue
                if b == 1:
                    if is_first:
                        s += coord[i]
                    else:
                        s += "+" + coord[i]
                elif b == -1:
                    s += "-" + coord[i]
                else:
                    if b > 0:
                        s += "+" + str(b) + coord[i]
                    else:
                        s += str(b) + coord[i]
                is_first = False
            return s

        if (not np.allclose(refUC, np.eye(3)) or
            not np.allclose(shiftUC, np.zeros(3))):
            write_ref = True  # To avoid writing this huge block again
        else:
            write_ref = False

        # Print header
        print("\n ### {} \n".format(self.ind))

        # Print rotation part
        rotstr = [s +
                  " ".join("{0:3d}".format(x) for x in row) +
                  t for s, row, t in zip(["rotation : |", " " *
                                          11 +
                                          "|", " " *
                                          11 +
                                          "|"], self.rotation, [" |", " |", " |"])]
        if write_ref:
            fstr = ("{0:3d}")
            R = self.rotation_refUC(refUC)
            rotstr1 = [" " *
                       5 +
                       s +
                       " ".join(fstr.format(x) for x in row) +
                       t for s, row, t in zip(["rotation : |",
                                               " (refUC)   |",
                                               " " * 11 + "|"
                                               ],
                                              R,
                                              [" |", " |", " |"])]
            rotstr = [r + r1 for r, r1 in zip(rotstr, rotstr1)]

        kstring = "gk = [" + ", ".join(
                    [parse_row_transform(r) for r in np.transpose(np.linalg.inv(self.rotation))]
                    ) + "]"

        if write_ref:
            kstring += "  |   refUC:  gk = ["+", ".join(
                    [parse_row_transform(r) for r in np.transpose(np.linalg.inv(R))]
                    )+ "]"
                
        print("\n".join(rotstr))
        print("\n\n",kstring)

        # Print spinor transformation matrix
        if self.spinor:
            spinstr = [s +
                       " ".join("{0:6.3f}{1:+6.3f}j".format(x.real, x.imag) for x in row) +
                       t 
                       for s, row, t in zip(["\nspinor rot.         : |",
                                             " " * 22 + "|",
                                             ], 
                                             self.spinor_rotation, 
                                             [" |", " |"]
                                           )
                       ]
            print("\n".join(spinstr))
            if write_ref:
                spinstr = [s +
                           " ".join("{0:6.3f}{1:+6.3f}j".format(x.real, x.imag) for x in row) +
                           t 
                           for s, row, t in zip(["spinor rot. (refUC) : |",
                                                 " " * 22 + "|",
                                                 ], 
                                                 self.spinor_rotation*self.sign, 
                                                 [" |", " |"]
                                               )
                           ]
                print("\n".join(spinstr))

        # Print translation part
        trastr = ("\ntranslation         :  [ " 
                  + " ".join("{0:8.4f}"
                             .format(x) for x in self.get_transl_mod1(self.translation.round(6))
                             ) 
                  + " ] "
                  )
        print(trastr)

        if write_ref:
            _t=self.translation_refUC(refUC,shiftUC)
            trastr = ("translation (refUC) :  [ " 
                      + " ".join("{0:8.4f}"
                                 .format(x) for x in self.get_transl_mod1(_t.round(6))
                                 )
                  + " ] "
                  )
            print(trastr)

        print("\naxis: {0} ; angle = {1}, inversion : {2}\n".format(
            self.axis.round(6), self.angle_str, self.inversion))

    def str(self, refUC=np.eye(3), shiftUC=np.zeros(3)):
        """
        Construct description of symmetry operation.

        Parameters
        ----------
        refUC : array, default=np.eye(3)
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting.
        shiftUC : array, default=np.zeros(3)
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting.

        Returns
        -------
        str
            Description to print.
        """
#        print ( "symmetry # ",self.ind )
        R = self.rotation_refUC(refUC)
        t = self.translation_refUC(refUC, shiftUC)
#        np.savetxt(stdout,np.hstack( (R,t[:,None])),fmt="%8.5f" )
        S = self.spinor_rotation
        return ("   ".join(" ".join(str(x) for x in r) for r in R) + "     " + " ".join(str_(x) for x in t) + ("      " + \
                "    ".join("  ".join(str_(x) for x in X) for X in (np.abs(S.reshape(-1)), np.angle(S.reshape(-1)) / np.pi))))

    def str2(self, refUC=np.eye(3), shiftUC=np.zeros(3)):
        """
        Print matrix of a symmetry operation in the format: 
        {{R|t}}-> R11,R12,...,R23,R33,t1,t2,t3 and, when SOC was included, the 
        elements of the matrix describing the transformation of the spinor in 
        the format:
        Re(S11),Im(S11),Re(S12),...,Re(S22),Im(S22).
        
        Parameters
        ----------
        refUC : array, default=np.eye(3)
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting.
        shiftUC : array, default=np.zeros(3)
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting.

        Returns
        -------
        str
            Description to print.
        """
        if refUC is None:
            refUC = np.eye(3, dtype=int)
        if shiftUC is None:
            shiftUC = np.zeros(3, dtype=float)
# this method for Bilbao server
#       refUC - row-vectors, expressing the reference unit cell vectors in terms of the lattice used in calculation
#        print ( "symmetry # ",self.ind )
        R = self.rotation
        t = self.translation
#        np.savetxt(stdout,np.hstack( (R,t[:,None])),fmt="%8.5f" )
        S = self.spinor_rotation
        return ("   ".join(" ".join("{0:2d}".format(x) for x in r) for r in R) + "     " + " ".join("{0:10.6f}".format(x) for x in t) + (
            ("      " + "    ".join("  ".join("{0:10.6f}".format(x) for x in (X.real, X.imag)) for X in S.reshape(-1))) if S is not None else "") + "\n")

    def str_sym(self, alat):
        """
        Write 4 strings (+1 empty) for the prefix.sym file 
        for sitesym in wannier90: 
        The symmetry operations act on a point r as rR − t.

        Parameters
        ----------
        alat : float
            Lattice parameter in angstroms.

        Returns
        -------
        str
            Description to print.
            1 blank line
            3 lines: cartesian rotation matrix
            1 line : cartesian translation in units of alat
        """

        Rcart  = self.Lattice.T.dot(self.rotation).dot(np.linalg.inv(self.Lattice).T)
        t =  - self.translation @ self.Lattice/alat/BOHR   

        arr = np.vstack((Rcart, [t]))
        return "\n"+"".join("   ".join(f"{x:20.15f}" for x in r) + "\n" for r in arr  )

    def json_dict(self, refUC=np.eye(3), shiftUC=np.zeros(3)):
        '''
        Prepare dictionary with info of symmetry to save in JSON

        Returns
        -------
        d : dict
            Dictionary with info about symmetry
        '''

        d = {}
        d["axis"]  = self.axis
        d["angle str"] = self.angle_str
        d["angle pi"] = self.angle/np.pi
        d["inversion"] = self.inversion
        d["sign"] = self.sign

        d["rotation matrix"] = self.rotation
        d["translation"] = self.translation

        R = self.rotation_refUC(refUC)
        t = self.translation_refUC(refUC, shiftUC)
        d["rotation matrix refUC"] = R
        d["translation refUC"]= t

        return d
    
    def transform_r(self, vector, inverse =False):
        """
        Transform a real-space vector (in lattice coordinates) under the symmetry operation.

        Parameters
        ----------
        vector : array((...,3), dtype=float) 
            Vector to transform. (or array of vectors)
        
        Returns
        -------
        array
            Transformed vector.
        """
        if inverse:
            return (vector-self.translation[...,:]).dot(self.rotation_inv.T)
        else:
            return vector.dot(self.rotation.T) + self.translation[...,:]
        
    @cached_property
    def rotation_cart(self):
        return self.Lattice.T.dot(self.rotation).dot(np.linalg.inv(self.Lattice).T)
    
    @cached_property
    def reciprocal_lattice(self):
        return np.linalg.inv(self.Lattice).T

    @cached_property
    def rotation_inv(self):
        return np.linalg.inv(self.rotation)
        
    def transform_k(self, vector, inverse=False):
        """
        Transform a k-space vector under the symmetry operation.

        Parameters
        ----------
        vector : array((...,3), dtype=float) 
            Vector to transform. (or array of vectors)
        
        Returns
        -------
        array
            Transformed vector.
        """
        # print (f"rotation = {self.rotation}")
        # print (f"transforming {vector} ({vector.dot(self.reciprocal_lattice)})")
        # print (f"by {self.rotation_inv}")
        if inverse:
            res = vector.dot(self.rotation)
        else:
            res = vector.dot(self.rotation_inv)
        # print (f"got {res} ({res.dot(self.reciprocal_lattice)})")
        return res


class SpaceGroup():
    """
    Determine the space-group and save info in attributes. Contains methods to 
    describe and print info about the space-group.

    Parameters
    ----------
    cell : tuple, default=None
        `cell[0]` is a 3x3 array where cartesian coordinates of basis 
        vectors **a**, **b** and **c** are given in rows. `cell[1]` is an array
        where each row contains the direct coordinates of an ion's position. 
        `cell[2]` is an array where each element is a number identifying the 
        atomic species of an ion. See `cell` parameter of function 
        `get_symmetry` in 
        `Spglib <https://spglib.github.io/spglib/python-spglib.html#get-symmetry>`_.
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

    Attributes
    ----------
    spinor : bool
        `True` if wave-functions are spinors (SOC), `False` if they are scalars.
    symmetries : list
        Each element is an instance of class `SymmetryOperation` corresponding 
        to a symmetry in the point group of the space-group.
    symmetries_tables : list
        Attribute `symmetries` of class `IrrepTable`. Each component is an 
        instance of class `SymopTable` corresponding to a symmetry operation
        in the "point-group" of the space-group.
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
    order : int
        Number of symmetries in the space group (in the coset decomposition 
        w.r.t. the translation subgroup).
    refUC : array, default=None
        3x3 array describing the transformation of vectors defining the 
        unit cell to the standard setting.
    shiftUC : array, default=None
        Translation taking the origin of the unit cell used in the DFT 
        calculation to that of the standard setting.
    alat : float
        Lattice parameter in angstroms (quantum espresso convention).
    from_sym_file : str, default=None
        if provided, the symmetry operations are read from this file.
        (format of pw2wannier90 prefix.sym  file)

    Notes
    -----
    The symmetry operations to which elements in the attribute `symmetries` 
    correspond are the operations belonging to the point group of the 
    space-group :math:`G`, i.e. the coset representations :math:`g_i` 
    taking part in the coset decomposition of :math:`G` w.r.t the translation 
    subgroup :math:`T`:
    ..math:: G=T+ g_1 T + g_2 T +...+ g_N T 
    """

    def __init__(
            self,
            cell,
            spinor=True,
            refUC=None,
            shiftUC=None,
            search_cell=False,
            trans_thresh=1e-5,
            alat=None,
            from_sym_file=None,
            no_match_symmetries=False,
            verbosity=0
            ):
        self.spinor = spinor
        self.Lattice = cell[0]
        self.positions = cell[1]
        self.typat = cell[2]
        (self.symmetries, 
         self.name, 
         self.number, 
         refUC_tmp, 
         shiftUC_tmp) = self._findsym(cell, from_sym_file, alat)
        self.order = len(self.symmetries)
        self.alat=alat
        if from_sym_file is not None:
            no_match_symmetries = True

        # Determine refUC and shiftUC according to entries in CLI
        self.symmetries_tables = IrrepTable(self.number, self.spinor, v=verbosity).symmetries
        self.refUC, self.shiftUC = self.determine_basis_transf(
                                            refUC_cli=refUC, 
                                            shiftUC_cli=shiftUC,
                                            refUC_lib=refUC_tmp, 
                                            shiftUC_lib=shiftUC_tmp,
                                            search_cell=search_cell,
                                            trans_thresh=trans_thresh,
                                            verbosity=verbosity
                                            )


        # Check matching of symmetries in refUC. If user set transf.
        # in the CLI and symmetries don't match, raise a warning.
        # Otherwise, transf. was calculated automatically and 
        # matching of symmetries was checked in determine_basis_transf
        if not no_match_symmetries:
            try:
                ind, dt, signs = self.match_symmetries(signs=self.spinor,
                                                    trans_thresh=trans_thresh
                                                    )
                # Sort symmetries like in tables
                args = np.argsort(ind)
                for i,i_ind in enumerate(args):
                    self.symmetries[i_ind].ind = i+1
                    self.symmetries[i_ind].sign = signs[i_ind]
                    self.symmetries.append(self.symmetries[i_ind])
                self.symmetries = self.symmetries[i+1:]
            except RuntimeError:
                if search_cell:  # symmetries must match to identify irreps
                    raise RuntimeError((
                        "refUC and shiftUC don't transform the cellto one where "
                        "symmetries are identical to those read from tables. "
                        "Try without specifying refUC and shiftUC."
                        ))
                elif refUC is not None or shiftUC is not None:
                    # User specified refUC or shiftUC in CLI. He/She may
                    # want the traces in a cell that is not neither the
                    # one in tables nor the DFT one
                    msg = ("WARNING: refUC and shiftUC don't transform the cell to "
                        "one where symmetries are identical to those read from "
                        "tables. If you want to achieve the same cell as in "
                        "tables, try not specifying refUC and shiftUC.")
                    log_message(msg, verbosity, 1)
                    pass

    def _findsym(self, cell, from_sym_file, alat):
        """
        Finds the space-group and constructs a list of symmetry operations
        
        Parameters
        ----------
        cell : list
            `cell[0]` is a 3x3 array where cartesian coordinates of basis 
            vectors **a**, **b** and **c** are given in rows. 
            `cell[1]` is an array
            where each row contains the direct coordinates of an ion's position. 
            `cell[2]` is an array where each element is a number identifying the 
            atomic species of an ion. See `cell` parameter of function 
            `get_symmetry` in 
            `Spglib <https://spglib.github.io/spglib/python-spglib.html#get-symmetry>`_.
        from_sym_file : str, default=None
            if provided, the symmetry operations are read from this file.
            (format of pw2wannier90 prefix.sym  file)
        alat : float
            Lattice parameter in angstroms. (quantum espresso convention)
        
        Returns
        -------
        list
            Each element is an instance of class `SymmetryOperation` corresponding 
            to a symmetry in the point group of the space-group.
        str
            Symbol of the space-group in Hermann-Mauguin notation. 
        int
            Number of the space-group.
        array
            3x3 array where cartesian coordinates of basis  vectors **a**, **b** 
            and **c** are given in rows. 
        array
            3x3 array describing the transformation of vectors defining the 
            unit cell to the convenctional setting.
        array
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting of spglib. It may not be
            the shift taking to the convenctional cell of tables; indeed, in 
            centrosymmetric groups they adopt origin choice 1 of ITA, rather 
            than choice 2 (BCS).
        """

        lattice = cell[0]
        dataset = spglib.get_symmetry_dataset(cell)
        if version.parse(spglib.__version__) < version.parse('2.5.0'):
            symbol = dataset['international']
            number = dataset['number']
            transformation_matrix = dataset['transformation_matrix']
            origin_shift = dataset['origin_shift']
            rotations = dataset['rotations']
            translations = dataset['translations']
        else:
            symbol = dataset.international
            number = dataset.number
            transformation_matrix = dataset.transformation_matrix
            origin_shift = dataset.origin_shift
            rotations = dataset.rotations
            translations = dataset.translations

        if from_sym_file is not None:
            print (f"Reading symmetries from file {from_sym_file}")
            assert alat is not None, "Lattice parameter must be provided to read symmetries from file"
            rot_cart, trans_cart = read_sym_file(from_sym_file)
            rotations, translations = cart_to_crystal(rot_cart, trans_cart, lattice, alat )
            translation_mod_1=False
        else:
            translation_mod_1=True

        symmetries = []
        for i, rot in enumerate(rotations):
            symmetries.append(SymmetryOperation(rot,
                                                translations[i],
                                                cell[0],
                                                ind=i+1,
                                                spinor=self.spinor,
                                                translation_mod1=translation_mod_1))

        return (symmetries, 
                symbol,
                number,
                transformation_matrix,
                origin_shift)

    @property
    def size(self):
        """
        Number of symmetry operations in the space-group.
        """
        return len(self.symmetries)

    def json(self, symmetries=None):
        '''
        Prepare dictionary with info of space group to save in JSON

        Returns
        -------
        d : dict
            Dictionary with info about space group
        '''

        d = {}

        if (np.allclose(self.refUC, np.eye(3)) and
            np.allclose(self.shiftUC, np.zeros(3))):
            cells_match = True
        else:
            cells_match = False

        d = {"name": self.name,
             "number": self.number,
             "spinor": self.spinor,
             "num symmetries": self.order,
             "cells match": cells_match,
             "symmetries": {}
             }

        for sym in self.symmetries:
            if symmetries is None or sym.ind in symmetries:
                d["symmetries"][sym.ind] = sym.json_dict(self.refUC, self.shiftUC)

        return d

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

        # Print cell vectors in DFT and reference cells
        vecs_refUC = np.dot(self.Lattice, self.refUC).T
        #vecs_refUC = np.dot(self.refUC, self.Lattice)
        print('Cell vectors in angstroms:\n')
        print('{:^32}|{:^32}'.format('Vectors of DFT cell', 'Vectors of REF. cell'))
        for i in range(3):
            vec1 = self.Lattice[i]
            vec2 = vecs_refUC[i]
            s = 'a{:1d} = {:7.4f}  {:7.4f}  {:7.4f}  '.format(i, vec1[0], vec1[1], vec1[2])
            s += '|  '
            s += 'a{:1d} = {:7.4f}  {:7.4f}  {:7.4f}'.format(i, vec2[0], vec2[1], vec2[2])
            print(s)
        print()

        # Print atomic positions
        print('Atomic positions in direct coordinates:\n')
        print('{:^} | {:^25} | {:^25}'.format('Atom type', 'Position in DFT cell', 'Position in REF cell'))
        positions_refUC = self.positions.dot(np.linalg.inv(self.refUC.T))
        for itype, pos1, pos2 in zip(self.typat, self.positions, positions_refUC):
            s = '{:^9d}'.format(itype)
            s += ' | '
            s += '  '.join(['{:7.4f}'.format(x) for x in pos1])
            s += ' | '
            s += '  '.join(['{:7.4f}'.format(x) for x in pos2])
            print(s)

        print()
        print('\n ---------- SPACE GROUP ----------- \n')
        print()
        print('Space group: {} (# {})'.format(self.name, self.number))
        print('Number of symmetries: {} (mod. lattice translations)'.format(self.order))
        refUC_print = self.refUC.T  # print following convention in paper
        print("\nThe transformation from the DFT cell to the reference cell of tables is given by: \n"
              + "        | {} |\n".format("".join(["{:8.4f}".format(el) for el in refUC_print[0]]))
              + "refUC = | {} |    shiftUC = {}\n".format("".join(["{:8.4f}".format(el) for el in refUC_print[1]]), np.round(self.shiftUC, 5))
              + "        | {} |\n".format("".join(["{:8.4f}".format(el) for el in refUC_print[2]]))
              )

        for symop in self.symmetries:
            if symmetries is None or symop.ind in symmetries:
                symop.show(refUC=self.refUC, shiftUC=self.shiftUC)


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
            f.write(" {0} \n".format(len(self.symmetries)))
            for symop in self.symmetries:
                f.write(symop.str_sym(alat))


    def write_trace(self):
        """
        Construct description of matrices of symmetry operations of the 
        space-group in the format: 
        {{R|t}}-> R11,R12,...,R23,R33,t1,t2,t3 and, when SOC was included, the 
        elements of the matrix describing the transformation of the spinor in 
        the format:
        Re(S11),Im(S11),Re(S12),...,Re(S22),Im(S22).

        Returns
        -------
        str
            String describing matrices of symmetry operations.
        """

        res = (" {0} \n"  # Number of Symmetry operations
               # In the following lines, one symmetry operation for each operation of the point group n"""
               ).format(len(self.symmetries))
        for symop in self.symmetries:
            res += symop.str2(refUC=self.refUC, shiftUC=self.shiftUC)
        return(res)

    def str(self):
        """
        Create a string to describe of space-group and its symmetry operations.

        Returns
        -------
        str
            Description to print.
        """
        return (
            "SG={SG}\n name={name} \n nsym= {nsym}\n spinor={spinor}\n".format(
                SG=self.number,
                name=self.name,
                nsym=len(
                    self.symmetries),
                spinor=self.spinor) +
            "symmetries=\n" +
            "\n".join(
                s.str(
                    self.refUC,
                    self.shiftUC) for s in self.symmetries) +
            "\n\n")

    def __match_spinor_rotations(self, S1, S2):
        """
        Determine the sign difference between matrices describing the 
        transformation of spinors found by `spglib` and those read from tables.

        Parameters
        ----------
        S1 : list
            Contains the matrices for the transformation of spinors 
            corresponding to symmetry operations found by `spglib`.
        S2 : list
            Contains the matrices for the transformation of spinors 
            corresponding to symmetry operations read from tables.

        Returns
        -------
        array
            The `j`-th element is the matrix to match the `j`-th matrices of 
            `S1` and `S2`.
        """
        n = 2

        def RR(x): 
            """
            Constructs a 2x2 complex matrix out of a list containing real and 
            imaginary parts.

            Parameters
            ----------
            x : list, length=8
                Length is 8. `x[:4]` contains the real parts, `x[4:]` the 
                imaginary parts.
            
            Returns
            -------
            array, shape=(2,2)
                Matrix of complex elements. 
            """

            return np.array([[x1 + 1j * x2 for x1, x2 in zip(l1, l2)] for l1, l2 in zip(x[:n * n].reshape((n, n)), x[n * n:].reshape((n, n)))])

        def residue_matrix(r): 
            """
            Calculate the residue of a matrix.

            Parameters
            ----------
            r : array
                Matrix used as ansatz for the minimization.

            Returns
            -------
            float            
            """

            return sum([min(abs(r.dot(b).dot(r.T.conj()) - s * a).sum() for s in (1, -1)) for a, b in zip(S1, S2)])

        def residue(x): 
            """
            Calculate the normalized residue.

            Parameters
            ----------
            x : list, length=8
                Length is 8. `x[:4]` contains the real parts, `x[4:]` the 
                imaginary parts.
            
            Returns
            -------
            float
            """
            return residue_matrix(RR(x)) / len(S1)

        for i in range(11):
            x0 = np.random.random(2 * n * n)
            res = minimize(residue, x0)
            r = res.fun
            if r < 1e-4:
                break
        if r > 1e-3:
            raise RuntimeError(
                "the accurcy is only {0}. Is this good?".format(r))

        R1 = RR(res.x)

        return np.array([R1.dot(b).dot(R1.T.conj()).dot(np.linalg.inv(
            a)).diagonal().mean().real.round() for a, b in zip(S1, S2)], dtype=int)

    def get_irreps_from_table(self, kpname, K, verbosity=0):
        """
        Read irreps of the little-group of a maximal k-point. 
        
        Parameters
        ----------
        kpname : str
            Label of the maximal k-point.
        K : array, shape=(3,)
            Direct coordinates of the k-point.
        verbosity : int, default=0
            Verbosity level. Default set to minimalistic printing

        Returns
        -------
        tab : dict
            Each key is the label of an irrep, each value another `dict`. Keys 
            of every secondary `dict` are indices of symmetries (starting from 
            1 and following order of operations in tables of BCS) and 
            values are traces of symmetries.

        Raises
        ------
        RuntimeError
            Translational or rotational parts read from tables and found by 
            `spglib` do not match for a symmetry operation.
        RuntimeError
            A symmetry from the tables matches with many symmetries found by 
            `spglib`.
        RuntimeError
            The label of a k-point given in the CLI matches with the label of a 
            k-point read from tables, but direct coordinates of these 
            k-vectors do not match.
        RuntimeError
            There is not any k-point in the tables whose label matches that 
            given in parameter `kpname`.
        """

        table = IrrepTable(self.number, self.spinor, v=verbosity)
        tab = {}
        for irr in table.irreps:
            if irr.kpname == kpname:
                k1 = np.round(np.linalg.inv(self.refUC.T).dot(irr.k), 5) % 1
                k2 = np.round(K, 5) % 1
                if not all(np.isclose(k1, k2)):
                    raise RuntimeError(
                        "the kpoint {0} does not correspond to the point {1} ({2} in refUC / {3} in primUC) in the table".format(
                            K,
                            kpname,
                            np.round(
                                irr.k,
                                3),
                            k1))
                tab[irr.name] = {}
                for i,(sym1,sym2) in enumerate(zip(self.symmetries,table.symmetries)):
                    try:
                        dt = sym2.t - sym1.translation_refUC(self.refUC, self.shiftUC)
                        tab[irr.name][i + 1] = irr.characters[i + 1] * \
                            sym1.sign * np.exp(2j * np.pi * dt.dot(irr.k))
                    except KeyError:
                        pass
        if len(tab) == 0:
            raise RuntimeError(
                "the k-point with name {0} is not found in the spacegroup {1}. found only :\n{2}".format(
                    kpname, table.number, "\n ".join(
                        "{0}({1}/{2})".format(
                            irr.kpname, irr.k, np.linalg.inv(self.refUC).dot(
                                irr.k) %
                            1) for irr in table.irreps)))
        return tab

    def determine_basis_transf(
            self,
            refUC_cli,
            shiftUC_cli,
            refUC_lib,
            shiftUC_lib,
            search_cell,
            trans_thresh,
            verbosity=0
            ):
        """ 
        Determine basis transformation to conventional cell. Priority
        is given to the transformation set by the user in CLI.

        Parameters
        ----------
        refUC_cli : array
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting. Set in CLI.
        shiftUC_cli : array
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting. Set in CLI.
        refUC_lib : array
            Obtained via spglib.
        shiftUC_lib : array
            Obtained via spglib. It may not be the shift taking the
            origin to the position adopted in the tables (BCS). For 
            example, origin choice 1 of ITA is adopted in spglib for 
            centrosymmetric groups, while origin choice 2 in BCS.
        search_cell : bool, default=False
            Whether the transformation to the conventional cell should be computed.
            It is `True` if kpnames was specified in CLI.
        trans_thresh : float, default=1e-5
            Threshold to compare translational parts of symmetries.
        verbosity : int, default=0
            Verbosity level. Default set to minimalistic printing

        Returns
        -------
        array 
            Transformation of vectors defining the unit cell to the 
            standard setting.
        array
            Shift taking the origin of the unit cell used in the DFT 
            calculation to that of the convenctional setting used in 
            BCS.

        Raises
        ------
        RuntimeError
            Could not find a pait (refUC,shiftUC) matching symmetries 
            obtained from spglib to those in tables (mod. lattice 
            translations of the primitive cell).

        """
        # Give preference to CLI input
        refUC_cli_bool = refUC_cli is not None
        shiftUC_cli_bool = shiftUC_cli is not None
        if refUC_cli_bool and shiftUC_cli_bool:  # Both specified in CLI.
            refUC = refUC_cli.T  # User sets refUC as if it was acting on column
            shiftUC = shiftUC_cli
            log_message('refUC and shiftUC read from CLI', verbosity, 1)
            return refUC, shiftUC
        elif refUC_cli_bool and not shiftUC_cli_bool:  # shiftUC not given in CLI.
            refUC = refUC_cli.T  # User sets refUC as if it was acting on column
            shiftUC = np.zeros(3, dtype=float)
            msg = ('refUC was specified in CLI, but shiftUC was not. Taking '
                   'shiftUC=(0,0,0)')
            log_message(msg, verbosity, 1)
            return refUC, shiftUC
        elif not refUC_cli_bool and shiftUC_cli_bool:  # refUC not given in CLI.
            refUC = np.eye(3, dtype=float)
            shiftUC = shiftUC_cli
            msg = ('shitfUC was specified in CLI, but refUC was not. Taking '
                   '3x3 identity matrix as refUC.')
            log_message(msg, verbosity, 1)
            return refUC, shiftUC
        elif not search_cell:
            refUC = np.eye(3, dtype=float)
            shiftUC = np.zeros(3, dtype=float)
            msg = ('Taking 3x3 identity matrix as refUC and shiftUC=(0,0,0). '
                   'If you want to calculate the transformation to '
                   'conventional cell, run IrRep with -searchcell')
            log_message(msg, verbosity, 1)
            return refUC, shiftUC
        else:  # Neither specifiend in CLI.
            msg = ('Determining transformation to conventional setting '
                   '(refUC and shiftUC)')
            log_message(msg, verbosity, 1)
            refUC = np.linalg.inv(refUC_lib)  # from DFT to convenctional cell

            # Check if the shift given by spglib works
            shiftUC = -refUC.dot(shiftUC_lib)
            try:
                ind, dt, signs = self.match_symmetries(
                                    refUC,
                                    shiftUC,
                                    trans_thresh=trans_thresh
                                    )
                return refUC, shiftUC
            except RuntimeError:
                pass

            # Check if the group is centrosymmetric
            inv = None
            for sym in self.symmetries:
                if np.allclose(sym.rotation, -np.eye(3)):
                    inv = sym

            if inv is None:  # Not centrosymmetric
                for r_center in self.vecs_centering():
                    shiftUC = shiftUC_lib + refUC.dot(r_center)
                    try:
                        ind, dt, signs = self.match_symmetries(
                                            refUC,
                                            shiftUC,
                                            trans_thresh=trans_thresh
                                            )
                        msg = (f'ShiftUC achieved with the centering: {r_center}')
                        log_message(msg, verbosity, 1)
                        return refUC, shiftUC
                    except RuntimeError:
                        pass
                raise RuntimeError(("Could not find any shift that leads to "
                                    "the expressions for the symmetries found "
                                    "in the tables."))

            else:  # Centrosymmetric. Origin must sit in an inv. center
                for r_center in self.vecs_inv_centers():
                    shiftUC = 0.5 * inv.translation + refUC.dot(0.5 * r_center)
                    try:
                        ind, dt, signs = self.match_symmetries(
                                            refUC,
                                            shiftUC,
                                            trans_thresh=trans_thresh
                                            )
                        msg = ('ShiftUC achieved in 2 steps:\n'
                               '  (1) Place origin of primitive cell on '
                               'inversion center: {}\n'
                               '  (2) Move origin of convenctional cell to the '
                               'inversion-center: {}'
                               .format(0.5 * inv.translation, r_center))
                        log_message(msg, verbosity, 1)
                        return refUC, shiftUC
                    except RuntimeError:
                        pass
                raise RuntimeError(("Could not find any shift that places the "
                                    "origin on an inversion center which leads "
                                    "to the expressions for the symmetries "
                                    "found in the tables. Enter refUC and "
                                    "shiftUC in command line"))


    def match_symmetries(
            self,
            refUC=None,
            shiftUC=None,
            signs=False,
            trans_thresh=1e-5
            ):
        """
        Matches symmetry operations of two lists. Translational parts 
        are matched mod. lattice translations (important for centered 
        structures).

        Parameters
        ----------
        refUC : array, default=None
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting.
        shiftUC : array, default=None
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting.
        signs : bool, default=False
            If `True`, match also rotations of spinors corresponding to 
            each symmetry.
        trans_thresh : float, default=1e-5
            Threshold used to compare translational parts of symmetries.
        
        Returns
        -------
        list
            The :math:`i^{th}` element corresponds to the :math:`i^{th}` 
            symmetry found by `spglib` and it is the position that the 
            same symmetry has in the tables.
        list
            The :math:`i^{th}` element corresponds to the :math:`i^{th}` 
            symmetry found by `spglib` and it is the phase difference 
            w.r.t. the same symmetry in the tables, which may arise if 
            their translational parts differ by a lattice vector.
        array
            The :math:`i^{th}` element corresponds to the :math:`i^{th}` 
            symmetry found by `spglib` and it is the sign needed to make
            the matrix for spinor rotation identical to that in tables.

        Note
        ----
        Arguments symmetries1 and symmetries2 always take values of 
        attributes self.symmetry and self.symmetries_tables, so they can
        be removed, but this way keeps the function more generic.
        """

        if refUC is None:
            refUC = self.refUC
        if shiftUC is None:
            shiftUC = self.shiftUC
        ind = []
        dt = []
        for j, sym in enumerate(self.symmetries):
            R = sym.rotation_refUC(refUC)
            t = sym.translation_refUC(refUC, shiftUC)
            found = False
            for i, sym2 in enumerate(self.symmetries_tables):
                t1 = refUC.dot(sym2.t - t) % 1
                #t1 = np.dot(sym2.t - t, refUC) % 1
                t1[1 - t1 < trans_thresh] = 0
                if np.allclose(R, sym2.R):
                    if np.allclose(t1, [0, 0, 0], atol=trans_thresh):
                        ind.append(i)
                        dt.append(sym2.t - t)
                        found = True
                        break
                        # Tolerance for rotational part comparison
                        # is much more restrictive than for transl.
                        # Make them consistent?
                    else:
                        raise RuntimeError((
                            "Error matching translational part for symmetry {}."
                            " A symmetry with identical rotational part has "
                            " been fond in tables, but their translational "
                            "parts do not match:\n"
                            "R (found, in conv. cell)= \n{} \n"
                            "t(found) = {} \n"
                            "t(table) = {} \n"
                            "t(found, in conv. cell) = {}\n"
                            "t(table)-t(found) "
                            "(in conv. cell, mod. lattice translation)= {}"
                            .format(
                                j+1, 
                                R, 
                                sym.translation, 
                                sym2.t, 
                                t,
                                t1
                                ))
                            )
            if not found:
                raise RuntimeError(
                    "Error matching rotational part for symmetry {0}. In the "
                    .format(j+1) +
                     "tables there is not any symmetry with identical " + 
                     "rotational part. \nR(found) = \n{} \nt(found) = {}"
                     .format(R, t))

        if (len(set(ind)) != len(self.symmetries)):
            raise RuntimeError(
                "Error in matching symmetries detected by spglib with the \
                 symmetries in the tables. Try to modify the refUC and shiftUC \
                 parameters")
        if signs:
            S1 = [sym.spinor_rotation for sym in self.symmetries]
            S2 = [self.symmetries_tables[i].S for i in ind]
            signs_array = self.__match_spinor_rotations(S1, S2)
        else:
            signs_array = np.ones(len(ind), dtype=int)
        return ind, dt, signs_array

    def vecs_centering(self):
        """ 
        Check the space group and generate vectors of centering.

        Returns
        -------
        array
            Each row is a lattice vector describing the centering.
        """
        cent = np.array([[0,0,0]])
        if self.name[0] == 'P':
            pass  # Just to make it explicit
        elif self.name[0] == 'C':
            cent = np.vstack((cent, cent + [1/2,1/2,0]))
        elif self.name[0] == 'I':
            cent = np.vstack((cent, cent + [1/2,1/2,1/2]))
        elif self.name[0] == 'F':
            cent = np.vstack((cent,
                              cent + [0,1/2,1/2],
                              cent + [1/2,0,1/2],
                              cent + [1/2,1/2,0],
                              )
                             )
        elif self.name[0] == 'A':  # test this
            cent = np.vstack((cent, cent + [0,1/2,1/2]))
        else:  # R-centered
            cent = np.vstack((cent,
                              cent + [2/3,1/3,1/3],
                              cent + [1/3,2/3,2/3],
                              )
                             )
        return cent

    def vecs_inv_centers(self):
        """
        Get the positions of all inversion centers in the unit cell.

        Returns
        -------
        array
            Each element is a vector pointing to a position center in 
            the convenctional unit cell.

        Notes
        -----
        If the space group is primitive, there are 8 inversion centers, 
        but there are more if it is a group with a centering.

        """
        vecs = np.array(
                        [
                        [0,0,0],
                        [1,0,0],
                        [0,1,0],
                        [0,0,1],
                        [1,1,0],
                        [1,0,1],
                        [0,1,1],
                        [1,1,1]
                        ]
                        )
        vecs = np.vstack([vecs + r for r in self.vecs_centering()])
        return vecs


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
    rotations = RT[:, 0:3]#.swapaxes(1, 2)
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
    return rot_crystal , trans_crystal 