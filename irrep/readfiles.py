
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
from .utility import str2bool, BOHR, split, log_message
import xml.etree.ElementTree as ET

Rydberg_eV = 13.605693  # eV
Hartree_eV = 2 * Rydberg_eV


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

    def __init__(self, filename, RL=3):
        self.f = open(filename, "rb")
        self.rl = 3
        # RECLENGTH=3 # the length of a record in WAVECAR. It is defined in the
        # first record, so let it be 3 fo far"
        self.rl, ispin, iprec = [int(x) for x in self.record(0)]
        if iprec != 45200:
            raise RuntimeError("double precision WAVECAR is not supported")
        if ispin != 1:
            raise RuntimeError(
                "WAVECAR contains spin-polarized non-spinor wavefunctions. "
                + "ISPIN={0}  this is not supported yet".format(ispin)
            )

    def record(self, irec, cnt=np.inf, dtype=float):
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



class ParserAbinit():
    """
    Parse header of the WFK file of Abinit.

    Parameters
    ---------
    filename : str
        Name of the WFK file of Abinit.

    Attributes
    ----------
    fWFK : file object
        Corresponds to the WFK file.
    kpt : array
        Each row contains the direct coordinates of a k-point.
    nband : array
        Each element contains the number of bands in a k-point.
    spinor : bool
        Whether the DFT calculation involved spinors (SOC) or not
    npwarr : array
        Each element is the number of plane waves used at a k-point
    kpt : array
        Each row contains the coordinates of a k-point in the DFT BZ

    Notes
    -----
    Abinit's routines that write the header can be found
    `here <https://docs.abinit.org/guide/abinit/#5>`_ or in the file
    `src/56_io_mpi/m_hdr.f90`.
    """

    def __init__(self, filename):
        #fWFK = FF(fname, "r")
        self.fWFK = FFR(filename)  # temporary
        self.kpt_count = 0  # index of the next k-point to be read

    def parse_header(self, verbosity=0):
        '''
        Parse header of WFK file and save as attributes quantities that 
        will be used in the rest of methods

        Parameters
        ----------
        verbosity : int, default=0
            Verbosity level. Default set to minimal printing

        Returns
        -------
        nband : array
            Each element contains the number of bands in a k-point.
        nkpt : int
            Number of k-points in WFK file
        rprimd : array
            Each row contains the Cartesian coords of a DFT cell vector
        ecut : float
            Plane-wave cutoff used for the DFT calculation
        spinor : bool
            Whether the DFT calculation involved spinors (SOC) or not
        typat : array
            Each element is an integer expressing the type ion
        xred : array
            Each row contains the direct coords of an ion
        efermi : float
            Fermi energy
        '''

        # 1st record
            # write(unit=header) codvsn,headform,fform
        try:  # version < 9.0.0
            record = record_abinit(self.fWFK, 'a6,2i4')
        except:  # version > 9.0.0
            self.fWFK.goto_record(0)
            record = record_abinit(self.fWFK, 'a8,2i4')
        stdout.flush()

        # Check version number of Abinit
        codsvn = record[0][0].decode('ascii').strip()
        headform, fform = record[0][1]
        defversion = ['8.6.3', '9.6.2', '8.4.4', '8.10.3']
        if codsvn not in defversion:
            msg = ("WARNING, the version {0} of abinit is not in {1} "
                   "and may not be fully tested"
                   .format(codsvn, defversion))
            log_message(msg, verbosity, 1)
        if headform < 80:
            raise ValueError(
                "Head form {0}<80 is not supported".format(headform)
                )

        # 2nd record
            # write(unit=header) bantot,date,intxc,ixc,natom,ngfft(1:3), nkpt,&
            # & nspden,nspinor,nsppol,nsym,npsp,ntypat,occopt,pertcase,usepaw,&
            # & ecut,ecutdg,ecutsm,ecut_eff,qptn(1:3),rprimd(1:3,1:3),stmbias,&
            # & tphysel,tsmear,usewvl, hdr%nshiftk_orig, hdr%nshiftk, hdr%mband
        record = record_abinit(self.fWFK, '18i4,19f8,4i4')[0]
        stdout.flush()
        (bandtot,
         natom,
         nkpt,
         nsym,
         npsp,
         nsppol,
         ntypat,
         usepaw,
         nspinor,
         occopt) = np.array(record[0])[[0, 4, 8, 12, 13, 11, 14, 17, 10, 15]]
        rprimd = record[1][7:16].reshape((3, 3))
        ecut = record[1][0] * Hartree_eV
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
            spinor = True
        elif nspinor == 1:
            spinor = False
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
                    nkpt=nkpt,
                    n2=nkpt * nsppol,
                    npsp=npsp,
                    nsym=nsym,
                    natom=natom,
                    bandtot=bandtot,
                    ntypat=ntypat)
               )
        record = record_abinit(self.fWFK, fmt)[0]
        typat = record[6]
        kpt = record[7]
        nband = record[1]
        istwfk = record[0]
        npwarr = record[2]

        # istwfk and npwarr are int, should be set, array and array
        if nkpt == 1:
            istwfk = set([int(istwfk)])
            npwarr = np.array([npwarr])
            nband  = np.array([nband])
        else:
            istwfk = set(istwfk)

        # Check that istwfk was 1 and consistency of number of bands
        if istwfk != {1}:
            raise ValueError(("istwfk should be 1 for all kpoints. Found {0}"
                              .format(istwfk))
                             )
        assert np.sum(nband) == bandtot, "Probably a bug in Abinit"

        # 4th record
            # write(unit,err=10, iomsg=errmsg) hdr%residm, hdr%xred(:,:), &
            # & hdr%etot, hdr%fermie, hdr%amu(:)
        record = record_abinit(
                    self.fWFK,
                    'f8,({natom},3)f8,f8,f8,{ntypat}f8'.format(natom=natom,
                                                               ntypat=ntypat
                                                               )
                    )[0]
        xred = record[1]
        efermi = record[3] * Hartree_eV

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
        record = record_abinit(self.fWFK,fmt)[0]

        # 6th record: skip it
            # read(unit, err=10, iomsg=errmsg) hdr%title(ipsp), &
            # & hdr%znuclpsp(ipsp), hdr%zionpsp(ipsp), hdr%pspso(ipsp), &
            # & hdr%pspdat(ipsp), hdr%pspcod(ipsp), hdr%pspxc(ipsp), &
            # & hdr%lmn_size(ipsp), hdr%md5_pseudos(ipsp)
        for ipsp in range(npsp):
            record = record_abinit(self.fWFK, "a132,f8,f8,5i4,a32")[0]

        # 7th record: additional records if usepaw=1
        if usepaw == 1:
            record_abinit(self.fWFK,"i4")
            record_abinit(self.fWFK,"i4")

        # Set as attributes quantities that need to be retrieved by the rest of methods
        self.nband = nband
        self.spinor = spinor
        self.npwarr = npwarr
        self.kpt = kpt
        return (nband, nkpt, rprimd, ecut, spinor, typat, xred, efermi)

    def parse_kpoint(self, ik):
        '''
        Parse block of a k-point from WFK file

        Returns
        -------
        CG : array
            Each row contains the coefficients of the plane-wave 
            expansion of a wave function
        eigen : array
            Energies of the wave functions
        kg : array
            Each row contains the direct coords of a reciprocal vector 
            used in the expansion of wave functions
        '''

        nspinor = 2 if self.spinor else 1

        # We need to skip lines in fWFK until we reach the lines of ik
        for i in range(self.kpt_count, ik+1):

            if self.kpt_count < ik:
                skip = True
            else:
                skip = False

            # 1st record: npw, nspinor, nband
            record = record_abinit(self.fWFK, "i4")  # [0]
            npw, nspinor_loc, nband = record
            assert npw == self.npwarr[ik], ("Different number of plane waves "
                                            "in header and k-point's block. "
                                            "Probably a bug in Abinit...")
            assert nspinor_loc == nspinor, ("Different values of nspinor in "
                                            "header and k-point's block. "
                                            "Probably a bug in Abinit...")
            assert nband == self.nband[ik], ("Different number of bands in "
                                             "header and k-point's block. "
                                             "Probably a bug in Abinit...")

            # 2nd record: reciprocal lattice vectors in the expansion
            kg = record_abinit(self.fWFK, "i4").reshape(npw, 3)

            # 3rd record: energies and occupations
            record = record_abinit(self.fWFK, "f8")
            if not skip:
                eigen = record[:nband]
                eigen *= Hartree_eV

            # 4th record: coefficients of expansions in plane waves
            if skip:
                record = record_abinit(self.fWFK, "f8")
            else:
                CG = np.zeros((nband, npw * nspinor), dtype=complex)
                for iband in range(nband):
                    record = record_abinit(self.fWFK, "f8")
                    CG[iband] = record[0::2] + 1.0j * record[1::2]

            self.kpt_count += 1

        return CG, eigen, kg

        # Check orthonormality for norm-conserving pseudos
        #if self.usepaw == 0:
        #    largest_value = np.max(np.abs(CG.conj().dot(CG.T)
        #                                  - np.eye(IBend - IBstart)))
        #    assert largest_value < 1e-10, "Wave functions are not orthonormal"


class ParserVasp:
    """
    Parser for Vasp interface

    Parameters
    ---------
    fPOS : str
        Name of the POSCAR file.
    fWAV : str
        Name of the WAVECAR file.
    onlysym : bool
        To stop right after parsing the POSCAR, before parsing the 
        header of the WAVECAR.

    Attributes
    ----------
    fPOS : str
        Name of the POSCAR file.
    fWAV : class
        Instance of `WAVECARFILE`
    """

    def __init__(self, fPOS, fWAV, onlysym=False):
        self.fPOS = fPOS
        if not onlysym:
            self.fWAV = WAVECARFILE(fWAV)

    def parse_poscar(self, verbosity=0):
        """
        Parses POSCAR.

        Parameters
        ----------
        verbosity : int, default=0
            Verbosity level. Default set to minimal printing

        Returns
        ------
        lattice : array
            3x3 array where cartesian coordinates of basis  vectors **a**, **b** 
            and **c** are given in rows. 
        positions : array
            Each row contains the direct coordinates of an ion's position. 
        typat : list
            Each element is a number identifying the atomic species of an ion.
        """

        msg = f'Reading POSCAR: {self.fPOS}'
        log_message(msg, verbosity, 1)
        fpos = (l.strip() for l in open(self.fPOS))
        title = next(fpos)  # title
        del title
        lattice = float(
            next(fpos)) * np.array([next(fpos).split() for i in range(3)], dtype=float)
        try:
            nat = np.array(next(fpos).split(), dtype=int)
        except BaseException:
            nat = np.array(next(fpos).split(), dtype=int)

        typat = [i + 1 for i in range(len(nat)) for j in range(nat[i])]

        l = next(fpos)
        if l[0] in ['s', 'S']:
            l = next(fpos)
        cartesian=False
        if l[0].lower()=='c':
            cartesian=True
        elif l[0].lower()!='d':
            raise RuntimeError(
                'only "direct" or "cartesian"atomic coordinates are supproted')
        positions = np.zeros((np.sum(nat), 3))
        i = 0
        for l in fpos:
            if i >= sum(nat):
                break
            try:
                positions[i] = np.array(l.split()[:3])
                i += 1
            except Exception as err:
                log_message(err, msg, 1)
                pass
        if sum(nat) != i:
            raise RuntimeError(
                "not all atomic positions were read : {0} of {1}".format(
                    i, sum(nat)))
        if cartesian:
            positions = positions.dot(np.linalg.inv(lattice))
        return lattice, positions, typat

    def parse_header(self):
        '''
        Parse header of WAVECAR

        Returns
        -------
        NK : int
            Number of k-points
        NBin : int
            Number of bands
        Ecut0 : float
            Plane-wave cutoff (in eV) used in DFT
        lattice : array
            Each row contains the cartesian coords of a unit cell vector
        '''

        tmp = self.fWAV.record(1)
        NK = int(tmp[0])
        NBin = int(tmp[1])
        Ecut0 = tmp[2]
        lattice = np.array(tmp[3:12]).reshape(3, 3)
        return NK, NBin, Ecut0, lattice

    def parse_kpoint(self, ik, NBin, spinor):
        '''
        Parse block of a particular k-point from WAVECAR

        Parameters
        ----------
        ik : int
            Index of the k-point
        NBin : int
            Number of bands
        spinor : bool
            Whether wave functions are spinors (SOC)

        Returns
        -------
        Energy : array
            Energy levels. Degenerate levels are repeated
        kpt : array
            Direct coords of the k-points with respect to the basis vectors 
            of the DFT reciprocal space cell
        npw : int
            Number of plane waves in the expansion of wave functions
        '''

        r = self.fWAV.record(2 + ik * (NBin + 1))

        # Check if number of plane waves is even for spinors
        npw = int(r[0])
        if spinor and npw % 2 != 0:
            raise RuntimeError("odd number of coefs {0} for spinor "
                               "wavefunctions".format(npw))
        kpt = r[1:4]
        Energy = np.array(r[4 : 4 + NBin * 3]).reshape(NBin, 3)[:, 0]
        WF = np.zeros((NBin, npw), dtype=np.complex64)
        for ib in range(NBin):
            WF[ib] = self.fWAV.record(3 + ik * (NBin + 1) + ib, npw, np.complex64)
        return WF, Energy, kpt, npw

class ParserEspresso:
    '''
    Parser of the interface for Quantum Espresso

    Parameters
    ----------
    prefix : str
        Prefix that serves as path to the `.save` directory. For example: 
        if the path is `./foo/bar.save` then `prefix` is `foo/bar`.

    Attributes
    ----------
    prefix : str
        Prefix that serves as path to the `.save` directory.
    input : class
        Instance of `Element` in the ElementTree XML API corresponding 
        to tag `input` in `data-file-schema.xml` file
    bandstructure : class
        Instance of `Element` in the ElementTree XML API corresponding 
        to tag `band_structure` in `data-file-schema.xml` file
    spinor : bool
        Whether wave functions are spinors (SOC)
    '''

    def __init__(self, prefix):
        self.prefix = prefix
        mytree = ET.parse(prefix + ".save/data-file-schema.xml")
        myroot = mytree.getroot()

        self.input = myroot.find("input")
        outp = myroot.find("output")
        self.bandstr = outp.find("band_structure")

        # todo: define spinor as property with getter
        self.spinor = str2bool(self.bandstr.find("noncolin").text)


    def parse_header(self):
        '''
        Parse universal info of the bandstructure from `data-file-schema.xml` 
        file

        Returns
        -------
        spinpol : bool
            Whether the calculation is spin-polarized
        Ecut0 : float
            Plane-wave cutoff (in eV) used in DFT
        EF : float
            Fermi energy in eV
        NK : int
            Number of k-points
        NBin_list : list
            If the calculation was spin polarized, `NBin_list[0]` and 
            `NBin_list[1]` are the number of bands for spin up and down 
            channels, respectively. Otherwise, `NBin_list[0]` is the 
            number of bands.
        '''

        Ecut0 = float(self.input.find("basis").find("ecutwfc").text)
        Ecut0 *= Hartree_eV
        NK = len(self.bandstr.findall("ks_energies"))

        # Parse number of bands
        try:
            NBin_dw=int(self.bandstr.find('nbnd_dw').text)
            NBin_up=int(self.bandstr.find('nbnd_up').text)
            spinpol=True
            print("spin-polarised bandstructure composed of {} up and {} dw "
                  "states".format(NBin_up,NBin_dw))
            NBin_dw+NBin_up
        except AttributeError:
            spinpol=False
            NBin=int(self.bandstr.find('nbnd').text)

        try:
            EF = float(self.bandstr.find("fermi_energy").text) * Hartree_eV
        except:
            EF = None

        if spinpol:
            NBin_list = [NBin_up, NBin_dw]
        else:
            NBin_list = [NBin]
        return spinpol, Ecut0, EF, NK, NBin_list

    def parse_lattice(self):
        '''
        Parse info about the crystal structure from `data-file-schema.xml` 
        file

        Returns
        -------
        lattice : array
            Each row contains the cartesian coords of a DFT unit cell vector
        positions : array
            Each row contains the direct coords of an ion in the DFT cell
        typat : list
            Indices that describe the type of element at each position. 
            All ions of the same type share the same index.
        alat : float
            Lattice parameter in Quantum Espresso's convention
        '''

        ntyp = int(self.input.find("atomic_species").attrib["ntyp"])
        struct = self.input.find("atomic_structure")
        nat = int(struct.attrib["nat"])
        alat = float(struct.attrib["alat"])
        del nat

        # Parse lattice vectors
        lattice = []
        for i in range(3):
            lattice.append(struct.find("cell").find("a{}".format(i + 1)).text.strip().split())
        lattice = BOHR * np.array(lattice, dtype=float)

        # Parse atomic positions in cartesian coordinates
        positions = []
        for at in struct.find("atomic_positions").findall("atom"):
            positions.append(at.text.split())
        positions = np.dot(np.array(positions, dtype=float)*BOHR,
                      np.linalg.inv(lattice))

        # Parse indices denoting type of atom
        atnames = []
        for sp in self.input.find("atomic_species").findall("species"):
            atnames.append(sp.attrib["name"])
        if len(atnames) != ntyp:
            raise RuntimeError("Error in the assigment of atom species. "
                               "Probably a bug in Quantum Espresso, but "
                               "your DFT input files")
        atnumbers = {atn: i + 1 for i, atn in enumerate(atnames)}
        typat = []
        for at in struct.find("atomic_positions").findall("atom"):
            typat.append(atnumbers[at.attrib["name"]])

        return lattice, positions, typat, alat


    def parse_kpoint(self, ik, NBin, spin_channel, verbosity=0):
        '''
        Parse block of a particular k-point from `data-file-schema.xml` file

        Parameters
        ----------
        ik : int
            Index of the k-point
        NBin : int
            Number of bands
        spin_channel : str
            `up` for spin up, `dw` for spin down, `None` if not spin polarized
        verbosity : int, default=0
            Verbosity level. Default set to minimalistic printing
       

        Returns
        -------
        WF : array
            Each row contains the coefficients of the plane-wave expansion of 
            a wave function
        Energy : array
            Energy levels in eV. Degenerate levels are repeated
        kg : array
            Each row contains the direct coords (ints) of a plane wave's 
            reciprocal lattice vector
        kpt : array
            Direct coords of the k-point w.r.t. DFT cell vectors
         '''

        kptxml = self.bandstr.findall("ks_energies")[ik]

        # Parse energy levels
        Energy = np.array(kptxml.find("eigenvalues").text.split(), dtype=float)
        Energy *= Hartree_eV

        # Open file with the wave functions
        wfcname="wfc{}{}".format({None:"","dw":"dw","up":"up"}[spin_channel], ik+1)
        try:
            fWFC=FF("{}.save/{}.dat".format(self.prefix,wfcname.lower()),"r")
        except FileNotFoundError:
            fWFC=FF("{}.save/{}.dat".format(self.prefix,wfcname.upper()),"r")

        rec = record_abinit(fWFC, "i4,3f8,i4,i4,f8")[0]
        kpt = rec[1]  # cartesian coords of k-point

        rec = record_abinit(fWFC, "4i4")
        igwx = rec[1]

        # Determine direct coords of k-point
        rec = record_abinit(fWFC, "(3,3)f8")
        B = np.array(rec)
        kpt = kpt.dot(np.linalg.inv(B))

        rec = record_abinit(fWFC, "({},3)i4".format(igwx))
        kg = np.array(rec)

        # Parse coefficients of wave functions
        npw = int(kptxml.find("npw").text)
        npwtot = npw * (2 if self.spinor else 1)
        msg = 'npwtot: {}, igwx: {}'.format(npwtot, igwx)
        log_message(msg, verbosity, 2)
        WF = np.zeros((NBin, npwtot), dtype=complex)
        for ib in range(NBin):
            rec = record_abinit(fWFC, "{}f8".format(npwtot * 2))
            WF[ib] = rec[0::2] + 1.0j * rec[1::2]

        return WF, Energy, kg, kpt


class ParserW90:
    '''
    Parser for Wannier90's interface

    Parameters
    ----------
    prefix : str
        Part of the path that serves as prefix for the `wannier90.win` file.
        For example: if the path is `./foo/bar.win`, then `prefix` is 
        `foo/bar`
    
    Attributes
    ----------
    prefix : str
        Part of the path that serves as prefix for the `wannier90.win` file.
    fwin : list
        Each element is a list of a non-comment line in `wannier90.win` file,
        split by blank spaces
    ind : list
        Each element is the first keyword of a line in `wannier90.win`
    iterwin : iterator object
        Iterator object for attribute `fwin`
    NBin : int
        Number of DFT bands
    spinor : bool
        Whether wave functions are spinors (SOC)
    NK : int
        Number of k-points in DFT calculation
    '''

    def __init__(self, prefix):

        self.prefix = prefix
        self.fwin = [l.strip().lower() for l in open(prefix + ".win").readlines()]
        self.fwin = [
            [s.strip() for s in split(l)]
            for l in self.fwin
            if len(l) > 0 and l[0] not in ("!", "#")
        ]
        self.ind = np.array([l[0] for l in self.fwin])
        self.iterwin = iter(self.fwin)

    def parse_header(self):
        '''
        Parse universal properties of the band structure

        Returns
        -------
        NK : int
            Number of k-points in DFT calculation
        NBin : int
            Number of DFT bands
        spinor : bool
            Whether wave functions are spinors (SOC)
        EF : float
            Fermi energy in eV
        '''

        self.NBin = self.get_param("num_bands", int)
        self.spinor = str2bool(self.get_param("spinors", str))
        try:
            EF = self.get_param("fermi_energy", float, None)
        except RuntimeError:
            EF = None
        self.NK = np.prod(np.array(self.get_param("mp_grid", str).split(), dtype=int))

        return self.NK, self.NBin, self.spinor, EF


    def parse_lattice(self):
        '''
        Parse crystal structure from `wannier90.win` file

        Returns
        -------
        lattice : array
            Each row contains the cartesian coords of a DFT unit cell vector
        positions : array
            Each row contains the direct coords of an ion in the DFT cell
        typat : list
            Indices that describe the type of element at each position. 
            All ions of the same type share the same index.
        kpred : array
            Each row contains the direct coords of a k-point in the DFT cell
        '''

        # Initialize quantities to make sure they aren't twice in .win
        lattice = None
        kpred = None
        found_atoms = False

        for l in self.iterwin:

            if l[0].startswith("begin"):

                # Parse lattice vectors
                if l[1] == "unit_cell_cart":
                    if lattice is not None:
                        raise RuntimeError(
                            "'begin unit_cell_cart' found more then once  in {}.win".format(
                                self.prefix
                            ))
                    l1 = next(self.iterwin)
                    if l1[0] in ("bohr", "ang"):
                        units = l1[0]
                        L = [next(self.iterwin) for i in range(3)]
                    else:
                        units = "ang"
                        L = [l1] + [next(self.iterwin) for i in range(2)]
                    lattice = np.array(L, dtype=float)
                    if units == "bohr":
                        lattice *= BOHR
                    self.check_end("unit_cell_cart")

                # Parse k-points
                elif l[1] == "kpoints":
                    if kpred is not None:
                        raise RuntimeError(
                            "'begin kpoints' found more then once  in {}.win".format(
                                self.prefix
                            ))
                    kpred = np.zeros((self.NK, 3), dtype=float)
                    for i in range(self.NK):
                        kpred[i] = next(self.iterwin)[:3]
                    self.check_end("kpoints")

                # Parse atomic positions
                elif l[1].startswith("atoms_"):
                    if l[1][6:10] not in ("cart", "frac"):
                        raise RuntimeError("unrecognised block :  '{}' ".format(l[0]))
                    if found_atoms:
                        raise RuntimeError(
                            "'begin atoms_***' found more then once  in {}.win".format(
                                self.prefix
                            ))
                    found_atoms = True
                    positions = []
                    nameat = []
                    while True:
                        l1 = next(self.iterwin)
                        if l1[0] == "end":
                            if l1[1] != l[1]:
                                raise RuntimeError(
                                    "'{}' ended with 'end {}'".format(
                                        " ".join(l), l1[1]
                                    )
                                )
                            else:
                                break
                        nameat.append(l1[0])
                        positions.append(l1[1:4])
                    typatdic = {n: i + 1 for i, n in enumerate(set(nameat))}
                    typat = [typatdic[n] for n in nameat]
                    positions = np.array(positions, dtype=float)
                    if l[1][6:10] == "cart":  # from cartesian to direct coords
                        positions = positions.dot(np.linalg.inv(lattice))

        return lattice, positions, typat, kpred


    def parse_energies(self):
        '''
        Parse energies from `wannier90.eig` file

        Returns
        -------
        Energy : array
            Each row contains the energy levels of a k-point. Degenerate 
            levels are repeated
        '''

        feig = self.prefix + ".eig"
        Energy = np.loadtxt(self.prefix + ".eig")
        try:
            if Energy.shape[0] != self.NBin * self.NK:
                raise RuntimeError("wrong number of entries ")
            ik = np.array(Energy[:, 1]).reshape(self.NK, self.NBin)
            if not np.all(
                ik == np.arange(1, self.NK + 1)[:, None] * np.ones(self.NBin, dtype=int)[None, :]
            ):
                raise RuntimeError("wrong k-point indices")
            ib = np.array(Energy[:, 0]).reshape(self.NK, self.NBin)
            if not np.all(
                ib == np.arange(1, self.NBin + 1)[None, :] * np.ones(self.NK, dtype=int)[:, None]
            ):
                raise RuntimeError("wrong band indices")
            Energy = Energy[:, 2].reshape(self.NK, self.NBin)
        except Exception as err:
            raise RuntimeError(" error reading {} : {}".format(feig,err))
        return Energy


    def parse_kpoint(self, ik, selectG):
        '''
        Parse wave functions' file of a k-point

        Parameters
        ----------
        ik : int
            Index of the k-point
        selectG : array
            First 3 rows of the array returned by :func:`~gvectors.calc_gvectors`

        Returns
        -------
        WF : array
            Each row contains the coefficients of the expansion of a wave 
            function in plane waves
        '''

        fname = "UNK{:05d}.{}".format(ik, "NC" if self.spinor else "1")
        fUNK = FF(fname, "r")
        ngx, ngy, ngz, ik_in, nbnd = record_abinit(fUNK, "i4,i4,i4,i4,i4")[0]
        ngtot = ngx * ngy * ngz
        nspinor = 2 if self.spinor else 1

        # Checks of consistency between UKN and .win
        if ik_in != ik:
            raise RuntimeError(
                "file {} contains point number {}, expected {}".format(
                    fname, ik_in, ik
                ))
        if nbnd != self.NBin:
            raise RuntimeError(
                "file {} contains {} bands , expected {}".format(fname, nbnd, self.NBin)
            )

        # Parse WF coefficients
        WF = []
        for ib in range(self.NBin):
            WF_tmp = []
            for i in range(nspinor):
                cg_tmp = record_abinit(fUNK, "{}f8".format(ngtot * 2))
                cg_tmp = (cg_tmp[0::2] + 1.0j * cg_tmp[1::2]).reshape(
                    (ngx, ngy, ngz), order="F")
                cg_tmp = np.fft.fftn(cg_tmp)
                WF_tmp.append(cg_tmp[selectG])
            WF.append(np.hstack(WF_tmp))

        WF = np.array(WF)
        return WF

    def parse_grid(self, ik):
        '''
        Parse grid of plane waves for a k-point from the file of 
        wave functions

        Parameters
        ----------
        ik : int
            Index of k-point

        Returns
        -------
        ngx : int
            Number of k-point along 1st direction in reciprocal space 
        ngy : int
            Number of k-point along 2nd direction in reciprocal space 
        ngz : int
            Number of k-point along 3rd direction in reciprocal space 
        '''

        fname = "UNK{:05d}.{}".format(ik, "NC" if self.spinor else "1")
        fUNK = FF(fname, "r")
        ngx, ngy, ngz, ik_in, nbnd = record_abinit(fUNK, "i4,i4,i4,i4,i4")[0]
        fUNK.close()
        return ngx, ngy, ngz

    def check_end(self, name):
        """
        Check if block in .win file is closed.

        Parameters
        ----------
        name : str
            Name of the block in .win file.
        
        Raises
        ------
        RuntimeError
            Block is not closed.
        """
        s = next(self.iterwin)
        if " ".join(s) != "end " + name:
            raise RuntimeError(
                "expected 'end {}, found {}'".format(name, " ".join(s))
            )



    def get_param(self, key, tp, default=None, join=False):
        """
        Return value of a parameter in .win file.

        Parameters
        ----------
        key : str
            Wannier90 input parameter.
        tp : function
            Function to apply to the value of the parameter, before 
            returning it.
        default
            Default value to return in case parameter `key` is not found.
        join : bool, default=False
            If the value of parameter `key` contains more than one element, 
            they will be concatenated with a blank space if `join` is set 
            to `True`. Used when the parameter is `mpgrid`.

        Returns
        -------
        Type(`tp`)
            Return the value of the parameter, after applying function 
            passed es keyword `tp`.

        Raises
        ------
        RuntimeError
            The parameter is not found in .win file, it is found more than 
            once or its value is formed by many elements but it is not
            `mpgrid`.
        """
        i = np.where(self.ind == key)[0]
        if len(i) == 0:
            if default is None:
                raise RuntimeError(
                    "parameter {} was not found in {}.win".format(key, self.prefix)
                )
            else:
                return default
        if len(i) > 1:
            raise RuntimeError(
                "parameter {} was found {} times in {}.win".format(
                    key, len(i), self.prefix
                )
            )

        x = self.fwin[i[0]][1:]  # mp_grid should work
        if len(x) > 1:
            if join:
                x = " ".join(x)
            else:
                raise RuntimeError(
                    "length {} found for parameter {}, rather than lenght 1 in {}.win".format(
                        len(x), key, self.prefix
                    )
                )
        else:
            x = self.fwin[i[0]][1]
        return tp(x)