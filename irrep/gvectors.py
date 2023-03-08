
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


import numpy as np
import numpy.linalg as la
from .readfiles import Hartree_eV


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
    nplane=np.Inf,
    Ecut1=-1,
    thresh=1e-3,
    spinor=True,
    nplanemax=10000,
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
    nplane : int, default=np.Inf
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
            print(N, len(igall))
        # if len(igall) >= nplane / (2 if spinor else 1):
        #     break
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
    #    print ("\n".join("{0:+4d}  {1:4d} {2:4d}  |  {3:6d}".format(ig[0],ig[1],ig[2],np.abs(ig).sum()) for ig in igall) )
    #    print (len(igall),len(set(igall)))
    if nplane < np.Inf: # vasp
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
    #    print ("ng=",ng)
    #    print ("igall1=",igall1)
    igallsrt = np.argsort((igall1[:, 2] * ng[1] + igall1[:, 1]) * ng[0] + igall1[:, 0])
    #    print (igallsrt)
    igall1 = igall[igallsrt]
    Eg = np.array(Eg)[igallsrt]
    #    print (igall1)
    igall = np.zeros((ncnt, 6), dtype=int)
    igall[:, :3] = igall1
    #    print (igall)
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

def sortIG(ik, kg, kpt, CG, RecLattice, Ecut0, Ecut, spinor):
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
        if `CG is None` - nothing is done for CG, but the indices of Gvectors are returned as CG
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
    print(
        "Found cutoff: {0:12.6f} eV   Largest plane wave energy in K-point {1:4d}: {2:12.6f} eV".format(
            Ecut0, ik, np.max(eKG)
        )
    )
    assert Ecut0 * 1.000000001 > np.max(eKG)
    sel = np.where(eKG < Ecut)[0]
    npw1 = sel.shape[0]

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

    if CG is not None:
        if spinor:
            CG = CG[:, np.hstack((sel[srt], sel[srt] + npw))]
        else:
            CG = CG[:, sel[srt]]
    else:
        if spinor:
            CG = np.hstack((sel[srt], sel[srt] + npw))
        else:
            CG = sel[srt]

    return CG, igall

def transformed_g(kpt, ig, RecLattice, A):
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
    RecLattice : array, shape=(3,3)
        Each row contains the cartesian coordinates of a basis vector forming 
        the unit-cell in reciprocal space.
    A : array, shape=(3,3)
        Matrix describing the tranformation of basis vectors of the unit cell 
        under the symmetry operation.
    
    Returns
    -------
    rotind : array
        `rotind[i]`=`j` if `A`*`ig[:,i]`==`ig[:,j]`.
"""
    #    Btrr=RecLattice.dot(A).dot(np.linalg.inv(RecLattice))
    #    Btr=np.array(np.round(Btrr),dtype=int) # The transformed rec. lattice expressed in the basis of the original rec. lattice
    #    if np.sum(np.abs(Btr-Btrr))>1e-6:
    #        raise NotSymmetryError("The lattice is not invariant under transformation \n {0}".format(A))
    B = np.linalg.inv(A).T
    kpt_ = B.dot(kpt)
    dkpt = np.array(np.round(kpt_ - kpt), dtype=int)
    #    print ("Transformation\n",A)
    #    print ("kpt ={0} -> {1}".format(kpt,kpt_))
    if not np.isclose(dkpt, kpt_ - kpt).all():
        raise NotSymmetryError(
            "The k-point {0} is transformed to non-equivalent point {1}  under transformation\n {2}".format(
                kpt, kpt_, A
            )
        )

    igTr = B.dot(ig[:3, :]) + dkpt[:, None]  # the transformed
    igTr = np.array(np.round(igTr), dtype=int)
    #    print ("the original g-vectors :\n",ig)
    #    print ("the transformed g-vectors :\n",igTr)
    ng = ig.shape[1]
    rotind = -np.ones(ng, dtype=int)
    for i in range(ng):
        for j in range(ig[4, i], ig[5, i]):
            if (igTr[:, i] == ig[:3, j]).all():
                rotind[i] = j
                break
        if rotind[i] == -1:
            raise RuntimeError(
                    "Error in the transformation of plane-waves in k-point={}: "
                    .format(kpt) +
                    "Not pair found for the g-vector igTr[{i}]={igtr}"
                    .format(i=i, igtr=igTr[:,i]) +
                    "obtained when transforming the g-vector ig[{i}]={ig} "
                    .format(i=i, ig=ig[:3,i] +
                    "with the matrix {B}, where B=inv(A).T with A={A}"
                    .format(B=B, A=A) +
                    "other g-vectors with the same energy:\n{other}"
                    .format(other)
                )
            )
    return rotind


def symm_eigenvalues(
    K, RecLattice, WF, igall, A, S, T, spinor
):
    """
    Calculate the traces of a symmetry operation for the wave-functions in a 
    particular k-point.

    Parameters
    ----------
    K : array, shape=(3,)
        Direct coordinates of the k-point.
    RecLattice : array, shape=(3,3)
        Each row contains the cartesian coordinates of a basis vector forming 
        the unit-cell in reciprocal space.
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
    igrot = transformed_g(K, igall, RecLattice, A)
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
    K, RecLattice, WF, igall, A, S, T, spinor
):
    """
    Computes the matrix S_mn = <Psi_m|{A|T}|Psi_n>

    Parameters
    ----------
    K : array, shape=(3,)
        Direct coordinates of the k-point.
    RecLattice : array, shape=(3,3)
        Each row contains the cartesian coordinates of a basis vector forming 
        the unit-cell in reciprocal space.
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

    Returns
    -------
    array
        Matrix of the symmetry operation in the basis of eigenstates of the 
        Bloch Hamiltonian :math:`H(k)`.
    """
    npw1 = igall.shape[1]
    multZ = np.exp(-1.0j * (2 * np.pi * A.dot(T).dot(igall[:3, :] + K[:, None])))
    igrot = transformed_g(K, igall, RecLattice, A)
    if spinor:
        WF1 = np.stack([WF[:, igrot], WF[:, igrot + npw1]], axis=2).conj()
        WF2 = np.stack([WF[:, :npw1], WF[:, npw1:]], axis=2)
        #        print (WF1.shape,WF2.shape,multZ.shape,S.shape)
        return np.einsum("mgs,ngt,g,st->mn", WF1, WF2, multZ, S)
    else:
        return np.einsum("mg,ng,g->mn", WF[:, igrot].conj(), WF, multZ)
