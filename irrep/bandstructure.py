
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


import copy
import functools

import numpy as np
import numpy.linalg as la

from .utility import str2bool, BOHR, split
from .readfiles import ParserAbinit, ParserVasp, ParserEspresso, ParserW90, Hartree_eV
from .readfiles import WAVECARFILE
from .kpoint import Kpoint
from .spacegroup import SpaceGroup
from .gvectors import sortIG, calc_gvectors


class BandStructure:
    """
    Parses files and organizes info about the whole band structure in 
    attributes. Contains methods to calculate and write traces (and irreps), 
    for the separation of the band structure in terms of a symmetry operation 
    and for the calculation of the Zak phase and wannier charge centers.

    Parameters
    ----------
    fWAV : str, default=None
        Name of file containing wave-functions in VASP (WAVECAR format).
    fWFK : str, default=None
        Name of file containing wave-functions in ABINIT (WFK format).
    prefix : str, default=None
        Prefix used for Quantum Espresso calculations or seedname of Wannier90 
        files.
    fPOS : str, default=None
        Name of file containing the crystal structure in VASP (POSCAR format).
    Ecut : float, default=None
        Plane-wave cutoff in eV to consider in the expansion of wave-functions.
    IBstart : int, default=None
        First band to be considered.
    IBend : int, default=None
        Last band to be considered.
    kplist : array, default=None
        List of indices of k-points to be considered.
    spinor : bool, default=None
        `True` if wave functions are spinors, `False` if they are scalars. 
        Mandatory for VASP.
    code : str, default='vasp'
        DFT code used. Set to 'vasp', 'abinit', 'espresso' or 'wannier90'.
    EF : float, default=None
        Fermi-energy.
    onlysym : bool, default=False
        Exit after printing info about space-group.
    spin_channel : str, default=None
        Selection of the spin-channel. 'up' for spin-up, 'dw' for spin-down. 
        Only applied in the interface to Quantum Espresso.
    refUC : array, default=None
        3x3 array describing the transformation of vectors defining the 
        unit cell to the standard setting.
    shiftUC : array, default=None
        Translation taking the origin of the unit cell used in the DFT 
        calculation to that of the standard setting.
    search_cell : bool, default=False
        Whether the transformation to the conventional cell should be computed.
        It is `True` if kpnames was specified in CLI.
    _correct_Ecut0 : float
        In case of VASP, if you get an error like ' computed ncnt=*** != input nplane=*** ', 
        try to set this parameter to a small positive or negative value (usually of order  +- 1e-7)
    trans_thresh : float, default=1e-5
        Threshold to compare translational parts of symmetries.

    Attributes
    ----------
    spacegroup : class
        Instance of `SpaceGroup`.
    spinor : bool
        `True` if wave functions are spinors, `False` if they are scalars. It 
        will be read from DFT files.
    efermi : float
        Fermi-energy. If user set a number as `EF` in CLI, it will be used. If 
        `EF` was set to `auto`, it will try to parse it and set to 0.0 if it 
        could not.
    Ecut0 : float
        Plane-wave cutoff (in eV) used in DFT calulations. Always read from 
        DFT files. Insignificant if `code`='wannier90'.
    Ecut : float
        Plane-wave cutoff (in eV) to consider in the expansion of wave-functions.
        Will be set equal to `Ecut0` if input parameter `Ecut` was not set or 
        the value of this is negative or larger than `Ecut0`.
    Lattice : array, shape=(3,3) 
        Each row contains cartesian coordinates of a basis vector forming the 
        unit-cell in real space.
    RecLattice : array, shape=(3,3)
        Each row contains the cartesian coordinates of a basis vector forming 
        the unit-cell in reciprocal space.
    kpoints : list
        Each element is an instance of `class Kpoint` corresponding to a 
        k-point specified in input parameter `kpoints`. If this input was not 
        set, all k-points found in DFT files will be considered.
    _correct_Ecut0 : float
        if you get an error like ' computed ncnt=*** != input nplane=*** ', 
        try to set this parameter to a small positive or negative value (usually of order  +- 1e-7)
    """

    def __init__(
        self,
        fWAV=None,
        fWFK=None,
        prefix=None,
        fPOS=None,
        Ecut=None,
        IBstart=None,
        IBend=None,
        kplist=None,
        spinor=None,
        code="vasp",
        EF='0.0',
        onlysym=False,
        spin_channel=None,
        refUC = None,
        shiftUC = None,
        search_cell = False,
        trans_thresh=1e-5,
        degen_thresh=1e-8
    ):

        code = code.lower()
        if spin_channel is not None:
            spin_channel=spin_channel.lower()
        if spin_channel=='down':
            spin_channel='dw'
        
        if code == "vasp":

            if spinor is None:
                raise RuntimeError(
                    "spinor should be specified in the command line for VASP bandstructure"
                )
            self.spinor = spinor
            parser = ParserVasp(fPOS, fWAV, onlysym)
            self.Lattice, positions, typat = parser.parse_poscar()
            if not onlysym:
                NK, NBin, self.Ecut0, lattice = parser.parse_header()
                if not np.allclose(self.Lattice, lattice):
                    raise RuntimeError("POSCAR and WAVECAR contain different lattices")
                EF_in = None  # not written in WAVECAR

        elif code == "abinit":

            # To do: use a return instead of attributes
            parser = ParserAbinit(fWFK)
            self.spinor = parser.spinor
            self.Lattice = parser.rprimd
            positions = parser.xred
            typat = parser.typat
            self.Ecut0 = parser.ecut
            EF_in = parser.efermi
            NBin = max(parser.nband)
            NK = parser.nkpt

        elif code == "espresso":

            parser = ParserEspresso(prefix)
            self.spinor = parser.spinor
            self.Lattice, positions, typat = parser.parse_lattice()
            spinpol, self.Ecut0, EF_in, NK, NBin_list = parser.parse_header()

            # Set NBin
            IBstartE=0
            if self.spinor and spinpol:
                raise RuntimeError("bandstructure cannot be both noncollinear and spin-polarised. Smth is wrong with the 'data-file-schema.xml'")
            elif spinpol:
                if spin_channel is None:
                    raise ValueError("Need to select a spin channel for spin-polarised calculations set  'up' or 'dw'")
                assert (spin_channel in ['dw','up'])
                if spin_channel == 'dw':
                    IBstartE = NBin_list[0]
                    NBin = NBin_list[1]
                else:
                    NBin = NBin_list[0]
            else:
                NBin = NBin_list[0]
                if spin_channel is not None:
                    raise ValueError("Found a non-polarized bandstructure, but spin channel is set to {}".format(spin_channel))

        elif code == "wannier90":

            if Ecut is None:
                raise RuntimeError("Ecut mandatory for Wannier90")

            self.Ecut0 = Ecut
            parser = ParserW90(prefix)
            NK, NBin, self.spinor, EF_in = parser.parse_header()
            self.Lattice, positions, typat, kpred = parser.parse_lattice()
            Energies = parser.parse_energies()

        else:
            raise RuntimeError("Unknown/unsupported code :{}".format(code))

        self.spacegroup = SpaceGroup(
                              cell=(self.Lattice, positions, typat),
                              spinor=self.spinor,
                              refUC=refUC,
                              shiftUC=shiftUC,
                              search_cell=search_cell,
                              trans_thresh=trans_thresh)
        if onlysym:
            return

        # Set Fermi energy
        if EF.lower() == "auto":
            if EF_in is None:
                print("WARNING : fermi-energy not found. Setting it as zero")
                self.efermi = 0.0
            else:
                self.efermi = EF_in
        else:
            try:
                self.efermi = float(EF)
            except:
                raise RuntimeError("Invalid value for keyword EF. It must be "
                                   "a number or 'auto'")
        print("Efermi: {:.4f} eV".format(self.efermi))

        # Fix indices of bands to be considered
        if IBstart is None or IBstart <= 0:
            IBstart = 0
        else:
            IBstart -= 1
        if IBend is None or IBend <= 0 or IBend > NBin:
            IBend = NBin
        NBout = IBend - IBstart
        if NBout <= 0:
            raise RuntimeError("No bands to calculate")

        # Set cutoff to calculate traces
        if Ecut is None or Ecut > self.Ecut0 or Ecut <= 0:
            self.Ecut = self.Ecut0
        else:
            self.Ecut = Ecut

        # Calculate vectors of reciprocal lattice
        self.RecLattice = np.zeros((3,3), dtype=float)
        for i in range(3):
            self.RecLattice[i] = np.cross(self.Lattice[(i + 1) % 3], self.Lattice[(i + 2) % 3])
        self.RecLattice *= (2.0*np.pi/np.linalg.det(self.Lattice))

        # To do: create writer of description for this class
        print(
            "WAVECAR contains {0} k-points and {1} bands.\n Saving {2} bands starting from {3} in the output".format(
                NK, NBin, NBout, IBstart + 1
            )
        )
        print("Energy cutoff in WAVECAR : ", self.Ecut0)
        print("Energy cutoff reduced to : ", self.Ecut)

        # Create list of indices for k-points
        if kplist is None:
            kplist = range(NK)
        else:
            kplist -= 1
            kplist = np.array([k for k in kplist if k >= 0 and k < NK])

        # Parse wave functions at each k-point
        self.kpoints = []
        for ik in kplist:

            if code == 'vasp':
                WF, Energy, kpt, npw = parser.parse_kpoint(ik, NBin, self.spinor)
                kg = calc_gvectors(kpt,
                                   self.RecLattice,
                                   self.Ecut0,
                                   npw,
                                   Ecut,
                                   spinor=self.spinor
                                   )
                if not self.spinor:
                    selectG = kg[3]
                else:
                    selectG = np.hstack((kg[3], kg[3] + int(npw / 2)))
                WF = WF[:, selectG]

            elif code == 'abinit':
                NBin = parser.nband[ik]
                kpt = parser.kpt[ik]
                WF, Energy, kg = parser.parse_kpoint(ik)
                WF, kg = sortIG(ik, kg, kpt, WF, self.RecLattice, self.Ecut0, self.Ecut, self.spinor)

            elif code == 'espresso':
                WF, Energy, kg, kpt = parser.parse_kpoint(ik, NBin, spin_channel)
                WF, kg = sortIG(ik+1, kg, kpt, WF, self.RecLattice/2.0, self.Ecut0, Ecut, self.spinor)

            elif code == 'wannier90':
                kpt = kpred[ik]
                Energy = Energies[ik]
                ngx, ngy, ngz = parser.parse_grid(ik+1)
                kg = calc_gvectors(kpred[ik],
                                   self.RecLattice,
                                   Ecut,
                                   spinor=self.spinor,
                                   nplanemax=np.max([ngx, ngy, ngz]) // 2
                                   )
                selectG = tuple(kg[0:3])
                WF = parser.parse_kpoint(ik+1, selectG)

            # Pick energy of IBend+1 band to calculate gaps
            try:
                upper = Energy[IBend] - self.efermi
            except BaseException:
                upper = np.NaN

            # Preserve only bands in between IBstart and IBend
            WF = WF[IBstart:IBend]
            Energy = Energy[IBstart:IBend] - self.efermi

            kp = Kpoint(
                ik=ik,
                kpt=kpt,
                WF=WF,
                Energy=Energy,
                ig=kg,
                upper=upper,
                IBstart=IBstart,
                IBend=IBend,
                RecLattice=self.RecLattice,
                symmetries_SG=self.spacegroup.symmetries,
                spinor=self.spinor,
                degen_thresh=degen_thresh,
                refUC=self.spacegroup.refUC,
                shiftUC=self.spacegroup.shiftUC,
                symmetries_tables=self.spacegroup.symmetries_tables
                )
            self.kpoints.append(kp)
        
    def getNK(self):
        """Getter for `self.kpoints`."""
        return len(self.kpoints)

    NK = property(getNK)


    def identify_irreps(self, kpnames):

        for ik, KP in enumerate(self.kpoints):
            
            if kpnames is not None:
                irreps = self.spacegroup.get_irreps_from_table(kpnames[ik], KP.K)
            else:
                irreps = None
            KP.identify_irreps(irreptable=irreps)

    def write_characters2(self):

        for KP in self.kpoints:

            # Print block of irreps and their characters
            KP.write_characters2()

            # Print number of inversion odd Kramers pairs
            if KP.num_bandinvs is None:
                print("\nInvariant under inversion: No")
            else:
                print("\nInvariant under inversion: Yes")
                if self.spinor:
                    print("Number of inversions-odd Kramers pairs : {}"
                          .format(int(KP.num_bandinvs / 2))
                          )
                else:
                    print("Number of inversions-odd states : {}"
                          .format(KP.num_bandinvs))

            # Print gap with respect to next band
            if not np.isnan(KP.upper):
                print("Gap with upper bands: ", KP.upper - KP.Energy[-1])
        
        # Print total number of band inversions
        if self.spinor:
            print("\nTOTAL number of inversions-odd Kramers pairs : {}"
                  .format(int(self.num_bandinvs/2)))
        else:
            print("TOTAL number of inversions-odd states : {}"
                  .format(self.num_bandinvs))
        
        print('Z2 invariant: {}'.format(int(self.num_bandinvs/2 % 2)))
        print('Z4 invariant: {}'.format(int(self.num_bandinvs/2 % 4)))

        # Print indirect gap and smalles direct gap
        print('Indirect gap: {}'.format(self.gap_indirect))
        print('Smallest direct gap in the given set of k-points: {}'.format(self.gap_direct))
    

    def json(self, kpnames=None):

        kpline = self.KPOINTSline()
        json_data = {}
        json_data['kpoints_line'] = kpline
        json_data['k-points'] = []
        
        for ik, KP in enumerate(self.kpoints):
            json_kpoint = KP.json()
            json_kpoint['kp in line'] = kpline[ik]
            if kpnames is None:
                json_kpoint['kpname'] = None
            else:
                json_kpoint['kpname'] = kpnames[ik]
            json_data['k-points'].append(json_kpoint)
        
        json_data['indirect gap (eV)'] =  self.gap_indirect
        json_data['Minimal direct gap (eV)'] =  self.gap_direct

        if self.spinor:
            json_data["number of inversion-odd Kramers pairs"]  = int(self.num_bandinvs / 2)
            json_data["Z4"] = int(self.num_bandinvs / 2) % 4,
        else:
            json_data["number of inversion-odd states"]  = self.num_bandinvs

        return json_data

    @property
    def gap_direct(self):
        gap = np.Inf
        for KP in self.kpoints:
            gap = min(gap, KP.upper-KP.Energy[-1])
        return gap

    @property
    def gap_indirect(self):
        min_upper = np.Inf  # smallest energy of bands above set
        max_lower = -np.inf  # largest energy of bands in the set
        for KP in self.kpoints:
            min_upper = min(min_upper, KP.upper)
            max_lower = max(max_lower, KP.Energy[-1])
        return min_upper - max_lower

    @property
    def num_bandinvs(self):
        num_bandinvs = 0
        for KP in self.kpoints:
            if KP.num_bandinvs is not None:
                num_bandinvs += KP.num_bandinvs
        return num_bandinvs

    def write_plotfile(self, plotFile):

        try:
            pFile = open(plotFile, "w")
        except BaseException:
            return

        kpline = self.KPOINTSline()
        for KP, kpl in zip(self.kpoints, kpline):
            KP.write_plotfile(pFile, kpl, self.fermi)
        pFile.close()


    def write_irrepsfile(self):

        file = open('irreps.dat', 'w')
        for KP in self.kpoints:
            KP.write_irrepsfile(file)
        file.close()


    def write_characters(
        self,
        degen_thresh=1e-8,
        kpnames=None,
        symmetries=None,
        preline="",
        plotFile=None,
    ):
        """
        Calculate irreps, number of band-inversion (if little-group contains 
        inversion), smallest direct gap and indirect gap and print all of them.

        Parameters
        ----------
        degen_thresh : float, default=1e-8
            Threshold energy used to decide whether wave-functions are
            degenerate in energy.
        refUC : array, default=None
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting.
        shiftUC : array, default=np.zeros(3)
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting.
        kpnames : list, default=None
            Labels of maximal k-points at which irreps of bands must be computed. 
            If it is not specified, only traces will be printed, not irreps.
        symmetries : list, default=None
            Index of symmetry operations whose description will be printed. 
        plotFile : str, default=None
            Name of file in which energy-levels and corresponding irreps will be 
            written to later place irreps in a band structure plot.

        Returns
        -------
        json_data : `json` object
            Object with output structured in `json` format.
        """
        #        if refUC is not None:
        #        self.spacegroup.show(refUC=refUC,shiftUC=shiftUC)
        #        self.spacegroup.show2(refUC=refUC)
        kpline = self.KPOINTSline()
        json_data = {}
        json_data[ "kpoints_line"] = kpline
        try:
            pFile = open(plotFile, "w")
        except BaseException:
            pFile = None
        NBANDINV = 0
        GAP = np.Inf
        Low = -np.Inf
        Up = np.inf
        json_data["k-points" ] = []
        if kpnames is not None:
            for kpname, KP in zip(kpnames, self.kpoints):
                irreps = self.spacegroup.get_irreps_from_table(kpname, KP.K)
                ninv, low, up , kdata = KP.write_characters(
                    degen_thresh,
                    irreptable=irreps,
                    symmetries=symmetries,
                    preline=preline,
                    efermi=self.efermi,
                    plotFile=pFile,
                    kpl=kpline,
                    symmetries_tables=self.spacegroup.symmetries_tables,
                    refUC=self.spacegroup.refUC,
                    shiftUC=self.spacegroup.shiftUC
                )
                kdata["kpname"] = kpname
                json_data["k-points" ].append(kdata)

                NBANDINV += ninv
                GAP = min(GAP, up - low)
                Up = min(Up, up)
                Low = max(Low, low)
        else:
            for KP, kpl in zip(self.kpoints, kpline):
                ninv, low, up , kdata = KP.write_characters(
                    degen_thresh,
                    symmetries=symmetries,
                    preline=preline,
                    efermi=self.efermi,
                    plotFile=pFile,
                    kpl=kpl,
                    symmetries_tables=self.spacegroup.symmetries_tables,
                    refUC=self.spacegroup.refUC,
                    shiftUC=self.spacegroup.shiftUC
                )
                kdata["kp in line"] = kpl
                json_data["k-points" ].append(kdata)
                NBANDINV += ninv
                GAP = min(GAP, up - low)
                Up = min(Up, up)
                Low = max(Low, low)

        if self.spinor:
            print(
                "number of inversion-odd Kramers pairs IN THE LISTED KPOINTS: ",
                int(NBANDINV / 2),
                "  Z4= ",
                int(NBANDINV / 2) % 4,
            )
            json_data["number of inversion-odd Kramers pairs"]  = int(NBANDINV / 2)
            json_data["Z4"] = int(NBANDINV / 2) % 4,
        else:
            print("number of inversion-odd states : ", NBANDINV)
            json_data["number of inversion-odd states"]  = NBANDINV

        #        print ("Total number of inversion-odd Kramers pairs IN THE LISTED KPOINTS: ",NBANDINV,"  Z4= ",NBANDINV%4)
        print("Minimal direct gap:", GAP, " eV")
        print("indirect  gap:", Up - Low, " eV")
        json_data["indirect gap (eV)"] =  Up-Low
        json_data["Minimal direct gap (eV)"] =  GAP
       
        return json_data

    def getNbands(self):
        """
        Return number of bands (if equal over all k-points), raise RuntimeError 
        otherwise.

        Returns
        -------
        int
            Number of bands in every k-point.
        """
        nbarray = [k.Nband for k in self.kpoints]
        if len(set(nbarray)) > 1:
            raise RuntimeError(
                "the numbers of bands differs over k-points:{0} \n cannot write trace.txt \n".format(
                    nbarray
                )
            )
        if len(nbarray) == 0:
            raise RuntimeError(
                "do we have any k-points??? NB={0} \n cannot write trace.txt \n".format(
                    nbarray
                )
            )
        return nbarray[0]

    def write_trace(
        self,
    ):
        """
        Generate `trace.txt` file to upload to the program `CheckTopologicalMat` 
        in `BCS <https://www.cryst.ehu.es/cgi-bin/cryst/programs/topological.pl>`_ .
        """

        f = open("trace.txt", "w")
        f.write(
            (
                " {0}  \n"
                + " {1}  \n"  # Number of bands below the Fermi level  # Spin-orbit coupling. No: 0, Yes: 1
            ).format(self.getNbands(), 1 if self.spinor else 0)
        )

        f.write(
                self.spacegroup.write_trace()
                )
        # Number of maximal k-vectors in the space group. In the next files
        # introduce the components of the maximal k-vectors))
        f.write("  {0}  \n".format(len(self.kpoints)))
        for KP in self.kpoints:
            f.write(
                "   ".join(
                    "{0:10.6f}".format(x)
                    for x in KP.K
                )
                + "\n"
            )
        for KP in self.kpoints:
            f.write(
                KP.write_trace()
            )

    def Separate(self, isymop, degen_thresh=1e-5, groupKramers=True):
        """
        Separate band structure according to the eigenvalues of a symmetry 
        operation.
        
        Parameters
        ----------
        isymop : int
            Index of symmetry used for the separation.
        degen_thresh : float, default=1e-5
            Energy threshold used to determine degeneracy of energy-levels.
        groupKramers : bool, default=True
            If `True`, states will be coupled by Kramers' pairs.

        Returns
        -------
        subspaces : dict
            Each key is an eigenvalue of the symmetry operation and the
            corresponding value is an instance of `class` `BandStructure` for 
            the subspace of that eigenvalue.
        """

        if isymop == 1:
            return {1: self}

        # Print description of symmetry used for separation
        symop = self.spacegroup.symmetries[isymop - 1]
        symop.show()

        # Separate each k-point
        kpseparated = [
            kp.Separate(symop, degen_thresh=degen_thresh, groupKramers=groupKramers)
            for kp in self.kpoints
        ] # each element is a dict with separated bandstructure of a k-point

        allvalues = np.array(sum((list(kps.keys()) for kps in kpseparated), []))
        #        print (allvalues)
        #        for kps in kpseparated :
        #            allvalues=allvalues | set( kps.keys())
        #        allvalues=np.array(allavalues)
        if groupKramers:
            allvalues = allvalues[np.argsort(np.real(allvalues))].real
            borders = np.hstack(
                (
                    [0],
                    np.where(abs(allvalues[1:] - allvalues[:-1]) > 0.01)[0] + 1,
                    [len(allvalues)],
                )
            )
            #            nv=len(allvalues)
            if len(borders) > 2:
                allvalues = set(
                    [allvalues[b1:b2].mean() for b1, b2 in zip(borders, borders[1:])]
                ) # unrepeated Re parts of all eigenvalues
                subspaces = {}
                for v in allvalues:
                    other = copy.copy(self)
                    other.kpoints = []
                    for K in kpseparated:
                        vk = list(K.keys())
                        vk0 = vk[np.argmin(np.abs(v - vk))]
                        if abs(vk0 - v) < 0.05:
                            other.kpoints.append(K[vk0])
                    subspaces[v] = other # unnecessary indent ?
                return subspaces
            else:
                return dict({allvalues.mean(): self})
        else:
            allvalues = allvalues[np.argsort(np.angle(allvalues))]
            print("allvalues:", allvalues)
            borders = np.where(abs(allvalues - np.roll(allvalues, 1)) > 0.01)[0]
            nv = len(allvalues)
            if len(borders) > 0:
                allvalues = set(
                    [
                        np.roll(allvalues, -b1)[: (b2 - b1) % nv].mean()
                        for b1, b2 in zip(borders, np.roll(borders, -1))
                    ]
                )
                print("distinct values:", allvalues)
                subspaces = {}
                for v in allvalues:
                    other = copy.copy(self)
                    other.kpoints = []
                    for K in kpseparated:
                        vk = list(K.keys())
                        vk0 = vk[np.argmin(np.abs(v - vk))]
                        #                    print ("v,vk",v,vk)
                        #                    print ("v,vk",v,vk[np.argmin(np.abs(v-vk))])
                        if abs(vk0 - v) < 0.05:
                            other.kpoints.append(K[vk0])
                        subspaces[v] = other
                return subspaces
            else:
                return dict({allvalues.mean(): self})

    def zakphase(self):
        """
        Calculate Zak phases along a path for a set of states.

        Returns
        -------
        z : array
            `z[i]` contains the total  (trace) zak phase (divided by 
            :math:`2\pi`) for the subspace of the first i-bands.
        array
            The :math:`i^{th}` element is the gap between :math:`i^{th}` and
            :math:`(i+1)^{th}` bands in the considered set of bands. 
        array
            The :math:`i^{th}` element is the mean energy between :math:`i^{th}` 
            and :math:`(i+1)^{th}` bands in the considered set of bands. 
        array
            Each line contains the local gaps between pairs of bands in a 
            k-point of the path. The :math:`i^{th}` column is the local gap 
            between :math:`i^{th}` and :math:`(i+1)^{th}` bands.
        """
        overlaps = [
            x.overlap(y)
            for x, y in zip(self.kpoints, self.kpoints[1:] + [self.kpoints[0]])
        ]
        print("overlaps")
        for O in overlaps:
            print(np.abs(O[0, 0]), np.angle(O[0, 0]))
        print("   sum  ", np.sum(np.angle(O[0, 0]) for O in overlaps) / np.pi)
        #        overlaps.append(self.kpoints[-1].overlap(self.kpoints[0],g=np.array( (self.kpoints[-1].K-self.kpoints[0].K).round(),dtype=int )  )  )
        nmax = np.min([o.shape for o in overlaps])
        # calculate zak phase in incresing dimension of the subspace (1 band,
        # 2 bands, 3 bands,...)
        z = np.angle(
            [[la.det(O[:n, :n]) for n in range(1, nmax + 1)] for O in overlaps]
        ).sum(axis=0) % (2 * np.pi)
        #        print (np.array([k.Energy[1:] for k in self.kpoints] ))
        #        print (np.min([k.Energy[1:] for k in self.kpoints],axis=0) )
        emin = np.hstack(
            (np.min([k.Energy[1:nmax] for k in self.kpoints], axis=0), [np.Inf])
        )
        emax = np.max([k.Energy[:nmax] for k in self.kpoints], axis=0)
        locgap = np.hstack(
            (
                np.min(
                    [k.Energy[1:nmax] - k.Energy[0 : nmax - 1] for k in self.kpoints],
                    axis=0,
                ),
                [np.Inf],
            )
        )
        return z, emin - emax, (emin + emax) / 2, locgap

    def wcc(self):
        """
        Calculate Wilson loops.

        Returns
        -------
        array
            Eigenvalues of the Wilson loop operator, divided by :math:`2\pi`.

        """
        overlaps = [
            x.overlap(y)
            for x, y in zip(self.kpoints, self.kpoints[1:] + [self.kpoints[0]])
        ]
        nmax = np.min([o.shape for o in overlaps])
        wilson = functools.reduce(
            np.dot,
            [functools.reduce(np.dot, np.linalg.svd(O)[0:3:2]) for O in overlaps],
        )
        return np.sort((np.angle(np.linalg.eig(wilson)) / (2 * np.pi)) % 1)

    def write_bands(self, locs=None):
        """
        Generate lines for a band structure plot, with cummulative length of the
        k-path as values for the x-axis and energy-levels for the y-axis.

        Returns
        -------
        str
            Lines to write into a file that will be parsed to plot the band 
            structure.
        """
        #        print (locs)
        kpline = self.KPOINTSline()
        nbmax = max(k.Nband for k in self.kpoints)
        EN = np.zeros((nbmax, len(kpline)))
        EN[:, :] = np.Inf
        for i, k in enumerate(self.kpoints):
            EN[: k.Nband, i] = k.Energy - self.efermi
        if locs is not None:
            loc = np.zeros((nbmax, len(kpline), len(locs)))
            for i, k in enumerate(self.kpoints):
                loc[: k.Nband, i, :] = k.getloc(locs).T
            return "\n\n\n".join(
                "\n".join(
                    (
                        "{0:8.4f}   {1:8.4f}  ".format(k, e)
                        + "  ".join("{0:8.4f}".format(l) for l in L)
                    )
                    for k, e, L in zip(kpline, E, LC)
                )
                for E, LC in zip(EN, loc)
            )
        else:
            return "\n\n\n".join(
                "\n".join(
                    ("{0:8.4f}   {1:8.4f}  ".format(k, e)) for k, e in zip(kpline, E)
                )
                for E in EN
            )

    def write_trace_all(
        self,
        degen_thresh=1e-8,
        symmetries=None,
        fname="trace_all.dat",
    ):
        """
        Write in a file the description of symmetry operations, energy-levels 
        and irreps calculated in all k-points.

        Parameters
        ----------
        degen_thresh : float, default=1e-8
            Threshold energy used to decide whether wave-functions are
            degenerate in energy.
        symmetries : list, default=None
            Index of symmetry operations whose traces will be printed. 
        fname : str, default=trace_all.dat
            Name of output file.
        """
        f = open(fname, "w")
        kpline = self.KPOINTSline()

        f.write(
            (
                "# {0}  # Number of bands below the Fermi level\n"
                + "# {1}  # Spin-orbit coupling. No: 0, Yes: 1\n"  #
            ).format(self.getNbands(), 1 if self.spinor else 0)
        )
        # add lines describing symmetry operations
        f.write(
            "\n".join(
                ("#" + l)
                for l in self.spacegroup.write_trace().split("\n")
            )
            + "\n\n"
        )
        for KP, KPL in zip(self.kpoints, kpline):
            f.write(
                KP.write_trace_all(
                    degen_thresh, symmetries=symmetries, efermi=self.efermi, kpline=KPL
                )
            )


    def KPOINTSline(self, kpred=None, breakTHRESH=0.1):
        """
        Calculate cumulative length along a path in reciprocal space.

        Parameters
        ----------
        kpred : list, default=None
            Each element contains the direct coordinates of a k-point in the
            attribute `kpoints`.
        breakTHRESH : float, default=0.1
            If the distance between two neighboring k-points in the path is 
            larger than `breakTHRESH`, it is taken to be 0.

        Returns
        -------
        array
            Each element is the cumulative distance along the path up to a 
            k-point. The first element is 0, so that the number of elements
            matches the number of k-points in the path.
        """
        if kpred is None:
            kpred = [k.K for k in self.kpoints]
        KPcart = np.array(kpred).dot(self.RecLattice)
        K = np.zeros(KPcart.shape[0])
        k = np.linalg.norm(KPcart[1:, :] - KPcart[:-1, :], axis=1)
        k[k > breakTHRESH] = 0.0
        K[1:] = np.cumsum(k)
        return K
