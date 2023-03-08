
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
import scipy
from scipy.io import FortranFile as FF
from sys import stdout
from .utility import FortranFileR as FFR


class WAVECARFILE:
    """
    Routines to read info from file WAVECAR of VASP.

    Parameters
    ----------
    fname : str, default=None
        Name of the WAVECAR file.
    RL : int, default=3
        Length parameter used to locate info in the file.

    Attributes
    ----------
    f : file object
        Corresponds to `fname`.
    rl : int
        Equal to parameter `RL`.
    """

    def __init__(self, fname=None, RL=3):
        self.f = open(fname, "rb")
        self.rl = 3
        self.rl = int(self.record(0)[0])

    def record(self, irec, cnt=np.Inf, dtype=float):
        """An auxilary function to get records from WAVECAR"""
        self.f.seek(irec * self.rl)
        return np.fromfile(self.f, dtype=dtype, count=min(self.rl, cnt))


def record_abinit(fWFK, st):
    """
    An auxilary function to get records from WAVECAR

    Parameters
    ----------
    fWFK : file object
        Corresponds to the WFK file of Abinit.
    st : str
        Format to be read.

    Returns
    -------
    r : str
        String read.
    """
    r = fWFK.read_record(st)
    if scipy.__version__.split(".")[0] == "0":
        r = r[0]
    return r


Rydberg_eV = 13.605693  # eV
Hartree_eV = 2 * Rydberg_eV


class AbinitHeader():
    """
    Parse header of the WFK file of Abinit.

    Paramters
    ---------
    fname : str
        Name of the WFK file of Abinit.

    Attributes
    ----------
    fWFK : file object
        Corresponds to the WFK file.
    rprim : array, shape=(3,3)
        Each row contains cartesian coordinates of a basis vector forming the 
        unit-cell in real space.
    ecut : float
        Plane-wave cutoff (in eV) to consider in the expansion of 
        wave-functions.
    usepaw : int
        1 if pseudopotentials are PAW, 0 otherwise.
    typat : array
        Each element is a number identifying the atomic species of an ion. 
        atomic species of an ion. See `cell` parameter of function 
        `get_symmetry` in 
        `Spglib <https://spglib.github.io/spglib/python-spglib.html#get-symmetry>`_.
    kpt : array
        Each row contains the direct coordinates of a k-point.
    nband : array
        Each element contains the number of bands in a k-point.
    xred : array
        Each row contains the direct coordinates of an ion's position. 
    efermi : float
        Fermi-level.

    Notes
    -----
    Abinit's routines that write the header can be found
    `here <https://docs.abinit.org/guide/abinit/#5>`_ or in the file
    `src/56_io_mpi/m_hdr.f90`.
    """

    def __init__(self, fname):

        #fWFK = FF(fname, "r")
        fWFK = FFR(fname)
        self.fWFK = fWFK

        # 1st record
            # write(unit=header) codvsn,headform,fform
        try:  # version < 9.0.0
            record = record_abinit(fWFK, 'a6,2i4')
        except:  # version > 9.0.0
            fWFK.goto_record(0)
            record = record_abinit(fWFK, 'a8,2i4')
        stdout.flush()
        codsvn = record[0][0].decode('ascii').strip()
        headform, fform = record[0][1]
        defversion = ['8.6.3', '9.6.2', '8.4.4', '8.10.3']
        if codsvn not in defversion:
            print(("WARNING, the version {0} of abinit is not in {1} "
                   "and may not be fully tested"
                   .format(codsvn, defversion))
                  )
        if headform < 80:
            raise ValueError(
                "Head form {0}<80 is not supported".format(headform)
                )

        # 2nd record
            # write(unit=header) bantot,date,intxc,ixc,natom,ngfft(1:3), nkpt,&
            # & nspden,nspinor,nsppol,nsym,npsp,ntypat,occopt,pertcase,usepaw,&
            # & ecut,ecutdg,ecutsm,ecut_eff,qptn(1:3),rprimd(1:3,1:3),stmbias,&
            # & tphysel,tsmear,usewvl, hdr%nshiftk_orig, hdr%nshiftk, hdr%mband
        record = record_abinit(fWFK, '18i4,19f8,4i4')[0]
        stdout.flush()
        (bandtot,
         self.natom,
         self.nkpt,
         nsym,
         npsp,
         nsppol,
         ntypat,
         self.usepaw,
         nspinor,
         occopt) = np.array(record[0])[[0, 4, 8, 12, 13, 11, 14, 17, 10, 15]]
        self.rprimd = record[1][7:16].reshape((3, 3))
        self.ecut = record[1][0] * Hartree_eV
        nshiftk_orig = record[2][1]
        nshiftk = record[2][2]

        # Check spin-polarization and if wave functions are spinors
        if nsppol != 1:
            raise RuntimeError(
                "Only nsppol=1 is supported. found {0}".format(nsppol)
                )
        if occopt == 9:  # extra records since Abinit v9
            raise RuntimeError("occopt=9 is not supported.")
        if nspinor == 2:
            self.spinor = True
        elif nspinor == 1:
            self.spinor = False
        else:
            raise RuntimeError(
                "Unexpected value nspinor = {0}".format(nspinor)
                )

        # 3rd record
            # write(unit=header) istwfk(1:nkpt),nband(1:nkpt*nsppol),&
            # & npwarr(1:nkpt),so_psp(1:npsp),symafm(1:nsym), &
            # & symrel(1:3,1:3,1:nsym),typat(1:natom), kpt(1:3,1:nkpt), &
            # & occ(1:bantot),tnons(1:3,1:nsym),znucltypat(1:ntypat), &
            # & wtk(1:nkpt)
        fmt = ('{nkpt}i4,{n2}i4,{nkpt}i4,{npsp}i4,{nsym}i4,({nsym},3,3)i4,'
               '{natom}i4,({nkpt},3)f8,{bandtot}f8,({nsym},3)f8,{ntypat}f8,'
               '{nkpt}f8'
               .format(
                    nkpt=self.nkpt,
                    n2=self.nkpt *
                    nsppol,
                    npsp=npsp,
                    nsym=nsym,
                    natom=self.natom,
                    bandtot=bandtot,
                    ntypat=ntypat)
               )
        record = record_abinit(fWFK, fmt)[0]
        self.typat = record[6]
        self.kpt = record[7]
        self.nband = record[1]
        self.npwarr, self.istwfk = record[2], record[0]
        self.__fix_arrays() # fix istwfk, npwarr and nband when nkpt==1

        # Check that istwfk was 1 and consistency of number of bands
        if self.istwfk != {1}:
            raise ValueError(("istwfk should be 1 for all kpoints. Found {0}"
                              .format(self.istwfk))
                             )
        assert np.sum(self.nband) == bandtot, "Probably a bug in Abinit"

        # 4th record
            # write(unit,err=10, iomsg=errmsg) hdr%residm, hdr%xred(:,:), &
            # & hdr%etot, hdr%fermie, hdr%amu(:)
        record = record_abinit(
                    fWFK,
                    'f8,({natom},3)f8,f8,f8,{ntypat}f8'.format(natom=self.natom,
                                                               ntypat=ntypat
                                                               )
                    )[0]
        self.xred = record[1]
        self.efermi = record[3]

        # 5th record: skip it
            # write(unit,err=10, iomsg=errmsg) &
            # & hdr%kptopt, hdr%pawcpxocc, hdr%nelect, hdr%charge, &
            # & hdr%icoulomb, hdr%kptrlatt,hdr%kptrlatt_orig, &
            # & hdr%shiftk_orig(:,1:hdr%nshiftk_orig), &
            # & hdr%shiftk(:,1:hdr%nshiftk)
        fmt = ('i4,i4,f8,f8,i4,(3,3)i4,(3,3)i4,({nshiftkorig},3)f8,'
               '({nshiftk},3)f8'
               .format(nshiftkorig=nshiftk_orig, nshiftk=nshiftk)
               )
        record = record_abinit(fWFK,fmt)[0]

        # 6th record: skip it
            # read(unit, err=10, iomsg=errmsg) hdr%title(ipsp), &
            # & hdr%znuclpsp(ipsp), hdr%zionpsp(ipsp), hdr%pspso(ipsp), &
            # & hdr%pspdat(ipsp), hdr%pspcod(ipsp), hdr%pspxc(ipsp), &
            # & hdr%lmn_size(ipsp), hdr%md5_pseudos(ipsp)
        for ipsp in range(npsp):
            record = record_abinit(fWFK, "a132,f8,f8,5i4,a32")[0]

        # 7th record: additional records if usepaw=1
        if self.usepaw == 1:
            record_abinit(fWFK,"i4")
            record_abinit(fWFK,"i4")

    def __fix_arrays(self):
        '''when nkpt=1, some integers must be converted to iterables, to avoid problems with indices'''
        if self.nkpt == 1:
            # istwfk and npwarr are int, should be set, array and array
            self.istwfk = set([self.istwfk])
            self.npwarr = np.array([self.npwarr]) 
            self.nband  = np.array([self.nband])
        else:
            self.istwfk = set(self.istwfk)
