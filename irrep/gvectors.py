
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


import warnings
import numpy as np
import numpy.linalg as la
from .readfiles import Hartree_eV
from .utility import log_message, orthogonolize


class NotSymmetryError(RuntimeError):
    """
    Pass if we attemp to apply to a k-vector a symmetry that does not belong 
    to its little-group.
    """
    pass


#!!   constant  below is 2m/hbar**2 in units of 1/eV Ang^2 (value is
#!!   adjusted in final decimal places to agree with VASP value; program
#!!   checks for discrepancy of any results between this and VASP values)
twomhbar2 = 0.262465831


# This function is a python translation of a part of WaveTrans Code
def calc_gvectors(
    K,
    RecLattice,
    Ecut,
    nplane=np.inf,
    Ecut1=-1,
    thresh=1e-3,
    spinor=True,
    nplanemax=10000,
    verbosity=0
):
    """ 
    Generates G-vectors taking part in the plane-wave expansion of 
    wave-functions in a particular k-point. Optionally, a cutoff `Ecut1` is 
    applied to get rid of large G-vectors.

    Parameters
    ----------
    K : array
        Direct coordinates of the k-point.
    RecLattice : array, shape=(3,3)
        Each row contains the cartesian coordinates of a basis vector forming 
        the unit-cell in reciprocal space.
    Ecut : float
        Plane-wave cutoff (in eV) used in the DFT calulation. Always read from 
        DFT files.
    nplane : int, default=np.inf
        Number of plane-waves in the expansion of wave-functions (read from DFT 
        files). Only significant for VASP. 
    Ecut1 : float, default=Ecut
        Plane-wave cutoff (in eV) to consider in the expansion of wave-functions.
    thresh : float, default=1e-3
        Threshold for defining the g-vectors with the same energy.
    spinor : bool, default=True
        `True` if wave functions are spinors, `False` if they are scalars. It 
        will be read from DFT files. Mandatory for `vasp`.
    nplanemax : int, default=10000
        Sets the maximun number of iterations when calculating vectors.
    verbosity : int, default=0
        Level of verbosity. If 0, no output is printed. If 1, only the most 
        important messages are printed. If 2, all messages are printed.

    Returns
    -------
    igall : array
        Every column corresponds to a plane-wave of energy smaller than 
        `Ecut`. The number of rows is 6: the first 3 contain direct 
        coordinates of the plane-wave, the fourth row stores indices needed
        to short plane-waves based on energy (ascending order). Fitfth 
        (sixth) row contains the index of the first (last) plane-wave with 
        the same energy as the plane-wave of the current column.
    
"""

    msg = ('Generating plane waves at k: ({} )'
           .format(' '.join([f'{x:6.3f}' for x in K])))
    log_message(msg, verbosity, 2)
    if Ecut1 <= 0:
        Ecut1 = Ecut
    B = RecLattice

    igp = np.zeros(3)
    igall = []
    Eg = []
    memory = np.full(10, True)
    for N in range(nplanemax):
        flag = True
        if N % 10 == 0:
            msg = f'Cycle {N:>3d}: number of plane waves = {len(igall):>10d}'
            log_message(msg, verbosity, 2)
        if len(igall) >= nplane / 2:    # Only enters if vasp
            if spinor:
                break
            else:      # Sure that not spinors?
                if len(igall) >= nplane: # spinor=F, all plane waves found
                    break
                elif np.all(memory): # probably spinor wrong set as spinor=F
                    raise RuntimeError(
                          "calc_gvectors is stuck "
                          "calculating plane waves of energy larger "
                          "than cutoff Ecut = {}. Make sure that the "
                          "VASP calculation does not include SOC and "
                          "set -spinor if it does.".format(Ecut)
                    )

        for ig3 in range(-N, N + 1):
            for ig2 in range(-(N - abs(ig3)), N - abs(ig3) + 1):
                for ig1 in set([-(N - abs(ig3) - abs(ig2)), N - abs(ig3) - abs(ig2)]):
                    igp = (ig1, ig2, ig3)
                    etot = la.norm((K + np.array(igp)).dot(B)) ** 2 / twomhbar2
                    if etot < Ecut:
                        igall.append(igp)
                        Eg.append(etot)
                        flag = False
        memory[:-1] = memory[1:]
        memory[-1] = flag

    ncnt = len(igall)
    if nplane < np.inf: # vasp
        if spinor:
            if 2 * ncnt != nplane:
                raise RuntimeError(
                    "*** error - computed 2*ncnt={0} != input nplane={1}".format(
                        2 * ncnt, nplane
                    )
                )
        else:
            if ncnt != nplane:
                raise RuntimeError(
                    "*** error - computed ncnt={0} != input nplane={1}".format(
                        ncnt, nplane
                    )
                )
    igall = np.array(igall, dtype=int)
    ng = igall.max(axis=0) - igall.min(axis=0)
    igall1 = igall % ng[None, :]
    igallsrt = np.argsort((igall1[:, 2] * ng[1] + igall1[:, 1]) * ng[0] + igall1[:, 0])
    igall1 = igall[igallsrt]
    Eg = np.array(Eg)[igallsrt]
    igall = np.zeros((ncnt, 6), dtype=int)
    igall[:, :3] = igall1
    igall[:, 3] = np.arange(ncnt)
    igall = igall[Eg <= Ecut1]
    Eg = Eg[Eg <= Ecut1]
    srt = np.argsort(Eg)
    Eg = Eg[srt]
    igall = igall[srt, :].T
    wall = [0] + list(np.where(Eg[1:] - Eg[:-1] > thresh)[0] + 1) + [igall.shape[1]]
    for i in range(len(wall) - 1):
        igall[4, wall[i] : wall[i + 1]] = wall[i]
        igall[5, wall[i] : wall[i + 1]] = wall[i + 1]
    #    print ("K={0}\n E={1}\nigall=\n{2}".format(K,Eg,igall.T))
    return igall

def sortIG(ik, kg, kpt, CG, RecLattice, Ecut0, Ecut, spinor, verbosity=0):
    """
    Apply plane-wave cutoff specified in CLI to the expansion of 
    wave-functions and sort the coefficients and plane-waves in ascending 
    order in energy.

    Parameters
    ----------
    ik : int
        Index of the k-point.
    kg : array
        Each row contains the integer coefficients of a reciprocal lattice 
        vector taking part in the plane-wave expansion of wave-functions at 
        the current k-point.
    kpt : array, shape=(3,)
        Direct coordinates of the k-point.
    CG : array
        `CG[i,j]` contains the complex coefficient corresponding to 
        :math:`j^{th}` plane-wave in the expansion of :math:`i^{th}` 
        wave-function.
    RecLattice : array, shape=(3,3)
        Each row contains the cartesian coordinates of a basis vector forming 
        the unit-cell in reciprocal space.
    Ecut : float
        Plane-wave cutoff (in eV) to consider in the expansion of wave-functions.
        Will be set equal to `Ecut0` if input parameter `Ecut` was not set or 
        the value of this is negative or larger than `Ecut0`.
    Ecut0 : float
        Plane-wave cutoff (in eV) used for DFT calulations. Always read from 
        DFT files. Insignificant if `code`=`wannier90`.
    spinor : bool
        `True` if wave-functions are spinors, `False` if they are scalars.

    Returns
    -------
    CG : array
        Contains the coefficients (same row-column formatting as argument 
        `CG`) of the expansion of wave-functions corresponding to 
        plane-waves of energy smaller than `Ecut`. Columns (plane-waves) 
        are shorted based on their energy, from smaller to larger.
    igall : array
        Every column corresponds to a plane-wave of energy smaller than 
        `Ecut`. The number of rows is 6: the first 3 contain direct 
        coordinates of the plane-wave, the fourth row stores indices needed
        to short plane-waves based on energy (ascending order). Fitfth 
        (sixth) row contains the index of the first (last) plane-wave with 
        the same energy as the plane-wave of the current column.
    """
    thresh = 1e-4 # default thresh to distinguish energies of plane-waves
    KG = (kg + kpt).dot(RecLattice)
    npw = kg.shape[0]
    eKG = Hartree_eV * (la.norm(KG, axis=1) ** 2) / 2
    log_message(
        f"Found cutoff: {Ecut0:12.6f} eV   Largest plane wave energy in K-point {ik:4d}: {np.max(eKG):12.6f} eV",
        verbosity=verbosity, level=2
        )
    assert Ecut0 * 1.000000001 > np.max(eKG)
    sel = np.where(eKG < Ecut)[0]

    KG = KG[sel]
    kg = kg[sel]
    eKG = eKG[sel]
    srt = np.argsort(eKG)
    eKG = eKG[srt]
    igall = np.zeros((6, len(sel)), dtype=int)
    igall[:3, :] = kg[srt].T
    igall[3, :] = srt
    wall = (
        [0] + list(np.where(eKG[1:] - eKG[:-1] > thresh)[0] + 1) + [igall.shape[1]]
    )
    for i in range(len(wall) - 1):
        igall[4, wall[i] : wall[i + 1]] = wall[i]
        igall[5, wall[i] : wall[i + 1]] = wall[i + 1]

    if spinor:
        CG = CG[:, np.hstack((sel[srt], sel[srt] + npw))]
    else:
        CG = CG[:, sel[srt]]

    return CG, igall

def transformed_g(kpt, ig, A, kpt_other=None, ig_other=None, inverse=False):
    """
    Determines how the transformation matrix `A` reorders the reciprocal
    lattice vectors taking part in the plane-wave expansion of wave-functions.

    Parameters
    ----------
    kpt : array, shape=(3,)
        Direct coordinates of the k-point.
    ig : array
        Every column corresponds to a plane-wave of energy smaller than 
        `Ecut`. The number of rows is 6: the first 3 contain direct 
        coordinates of the plane-wave, the fourth row stores indices needed
        to short plane-waves based on energy (ascending order). Fitfth 
        (sixth) row contains the index of the first (last) groups of 
        plane-waves of identical energy.
    ig_other : array, default=None
        If `ig` is not the same as `ig_other` the order of the rotates g-vectors 
        is determined by `ig_other`. (for transformations between different k-points)
    A : array, shape=(3,3)
        Matrix describing the tranformation of basis vectors of the unit cell 
        under the symmetry operation.
    
    Returns
    -------
    rotind : array
        `rotind[i] = j` if `B @ ig[:,i] == ig_other[:,j]`. if inverse is `False`,
        'rotind[j] = i' if `B @ ig[:,i] == ig_other[:,j]`. if inverse is `True`.

        where `B = np.linalg.inv(A).T`
""" 
    assert (ig_other is None) == (kpt_other is None), "ig_other and kpt_other must be provided (or not) together"
    if ig_other is None:
        ig_other = ig
        kpt_other = kpt
    B = np.linalg.inv(A).T
    kpt_ = B.dot(kpt)
    dkpt = np.array(np.round(kpt_ - kpt_other), dtype=int)

    if not np.isclose(dkpt, kpt_ - kpt_other).all():
        raise NotSymmetryError(
            f"The k-point {kpt} is transformed point {kpt_}  that is non-equivalent to the final point {kpt_other} "
            f"under transformation\n {A}"
        )

    igTr = B.dot(ig[:3, :]) + dkpt[:, None]  # the transformed
    igTr = np.array(np.round(igTr), dtype=int)
    ng = ig.shape[1]
    rotind = -np.ones(ng, dtype=int)
    for i in range(ng):
        for j in range(ig[4, i], ig[5, i]):
            if (igTr[:, i] == ig_other[:3, j]).all():
                if inverse:
                    rotind[j] = i
                else:
                    rotind[i] = j
                break

    for i in range(ng):
        if rotind[i] == -1:
            raise RuntimeError(
                    "Error in the transformation of plane-waves in k-point={}: "
                    .format(kpt) +
                    "Not pair found for the g-vector igTr[{i}]={igtr}"
                    .format(i=i, igtr=igTr[:,i]) +
                    "obtained when transforming the g-vector ig[{i}]={ig} "
                    .format(i=i, ig=ig_other[:3,i] +
                    "with the matrix {B}, where B=inv(A).T with A={A}"
                    .format(B=B, A=A)
                )
            )
    return rotind


def symm_eigenvalues(
    K, WF, igall, A, S, T, spinor
):
    """
    Calculate the traces of a symmetry operation for the wave-functions in a 
    particular k-point.

    Parameters
    ----------
    K : array, shape=(3,)
        Direct coordinates of the k-point.
    WF : array
        `WF[i,j]` contains the coefficient corresponding to :math:`j^{th}`
        plane-wave in the expansion of the wave-function in :math:`i^{th}`
        band. It contains only plane-waves of energy smaller than `Ecut`.
    igall : array
        Returned by `__sortIG`.
        Every column corresponds to a plane-wave of energy smaller than 
        `Ecut`. The number of rows is 6: the first 3 contain direct 
        coordinates of the plane-wave, the fourth row stores indices needed
        to short plane-waves based on energy (ascending order). Fitfth 
        (sixth) row contains the index of the first (last) plane-wave with 
        the same energy as the plane-wave of the current column.
    A : array, shape=(3,3)
        Matrix describing the tranformation of basis vectors of the unit cell 
        under the symmetry operation.
    S : array, shape=(2,2)
        Matrix describing how spinors transform under the symmetry.
    T : array, shape=(3,)
        Translational part of the symmetry operation, in terms of the basis 
        vectors of the unit cell.
    spinor : bool
        `True` if wave-functions are spinors, `False` if they are scalars.

    Returns
    -------
    array
        Each element is the trace of the symmetry operation in a wave-function.
    """
    npw1 = igall.shape[1]
    multZ = np.exp(
        -1.0j * (2 * np.pi * np.linalg.inv(A).dot(T).dot(igall[:3, :] + K[:, None]))
    )
    igrot = transformed_g(kpt=K, ig=igall, A=A)
    if spinor:
        part1 = WF[:, igrot].conj() * WF[:, :npw1] * S[0, 0]
        part2 = (
            WF[:, igrot + npw1].conj() * WF[:, npw1:] * S[1, 1]
            + WF[:, igrot].conj() * WF[:, npw1:] * S[0, 1]
            + WF[:, igrot + npw1].conj() * WF[:, :npw1] * S[1, 0]
        )
        return np.dot(part1 + part2, multZ)
    else:
        return np.dot(WF[:, igrot].conj() * WF[:, :], multZ)


def symm_matrix(
    K, WF, igall, A, S, T, spinor,
    WF_other=None, igall_other=None, K_other=None,
    block_ind=None,
    return_blocks=False,
    ortogonalize=True
):
    """
    Computes the matrix S_mn such that
    {A|T} |Psi_mk> = sum_n S_mn * |Psi_nk'>

    Parameters
    ----------
    K : array, shape=(3,)
        Direct coordinates of the k-point.
    WF : array
        `WF[i,j]` contains the coefficient corresponding to :math:`j^{th}`
        plane-wave in the expansion of the wave-function in :math:`i^{th}`
        band. It contains only plane-waves if energy smaller than `Ecut`.
    igall : array
        Returned by `__sortIG`.
        Every column corresponds to a plane-wave of energy smaller than 
        `Ecut`. The number of rows is 6: the first 3 contain direct 
        coordinates of the plane-wave, the fourth row stores indices needed
        to short plane-waves based on energy (ascending order). Fitfth 
        (sixth) row contains the index of the first (last) plane-wave with 
        the same energy as the plane-wave of the current column.
    WF_other, igall_other : array, default=None
        if provided, transformation to a different point is calculated.
    A : array, shape=(3,3)
        Matrix describing the tranformation of basis vectors of the unit cell 
        under the symmetry operation.
    S : array, shape=(2,2)
        Matrix describing how spinors transform under the symmetry.
    T : array, shape=(3,)
        Translational part of the symmetry operation, in terms of the basis 
        vectors of the unit cell.
    spinor : bool
        `True` if wave functions are spinors, `False` if they are scalars.
    block_ind : list( tuple(int,int) ), default=None
        If provided, only the diagonal blocks specified in the list are computed 
        The list contains tuples of the form (m,n) where m and n are the indices
        of the blocks to be computed.. i.e. S[m:n,m:n] is computed.
    return_blocks : bool, default=False
        If `True`, returns the diagonal blocks as a list. Otherwise, returns the
        matrix composed of those blocks.
    ortogonalize : bool, default=True
        If `True`, the matrix is orthogonalized. Set to `False` for speedup. 
        (in general it is not needed, but just in case)
    Returns
    -------
    array
        Matrix of the symmetry operation in the basis of eigenstates of the 
        Bloch Hamiltonian :math:`H(k)`.
    """
    assert (WF_other is None) == (igall_other is None) == (K_other is None), "WF_other and igall_other must be provided (or not) together"
    if WF_other is None:
        WF_other = WF
        igall_other = igall
        K_other = K
    if block_ind is None:
        block_ind = np.array([(0, WF.shape[0])])

    npw1 = igall.shape[1]
    multZ = np.exp(-2j * np.pi * T.dot(igall_other[:3, :] + K_other[:, None])) [None,:]
    igrot = transformed_g(kpt=K, ig=igall, A=A, ig_other=igall_other, kpt_other=K_other, inverse=True)
    if spinor:
        WFrot_up   = WF[:, igrot]*multZ
        WFrot_down = WF[:, igrot + npw1]*multZ 
        WFrot = np.stack([WFrot_up, WFrot_down], axis=2)
        WFrot = np.einsum("ts,mgs->mgt", S,WFrot)
        WFrot = WFrot.reshape((WFrot.shape[0], -1),order='F')
    else:
        WFrot = WF[:, igrot]*multZ
    block_list = []
    for b1,b2 in block_ind:
        WFinv = right_inverse(WF_other[b1:b2])
        block = np.dot(WFrot[b1:b2,:], WFinv)
        if ortogonalize:
            block = orthogonolize(block)
        block_list.append(block)
    if return_blocks:
        return block_list
    else:
        nwfout = sum(b2-b1 for b1,b2 in block_ind)
        M = np.zeros( (nwfout, nwfout), dtype=complex)
        i=0
        for (b1,b2),block in zip(block_ind, block_list):
            b = b2-b1
            M[i:i+b,i:i+b] = block
            i+=b
        return M
    
    

def right_inverse(A):
    """
    Compute the right inverse of a rectangular matrix A (m x n) where m < n.
    
    Parameters:
    A (numpy.ndarray): The input matrix of shape (m, n).
    
    Returns:
    numpy.ndarray: The right inverse of A of shape (n, m).
    """
    assert A.shape[0] <= A.shape[1], "Matrix must be rectangular, and m <= n."
    A_T = A.T.conj()
    return A_T @ np.linalg.inv(A @ A_T)