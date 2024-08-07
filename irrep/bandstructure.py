
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

from .readfiles import ParserAbinit, ParserVasp, ParserEspresso, ParserW90
from .kpoint import Kpoint
from .spacegroup import SpaceGroup
from .gvectors import sortIG, calc_gvectors
from .utility import get_block_indices, log_message


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
    trans_thresh : float, default=1e-5
        Threshold to compare translational parts of symmetries.
    degen_thresh : float, default=1e-8
        Threshold to determine the degeneracy of energy levels.
    calculate_traces : bool
        If `True`, traces of symmetries will be calculated. Useful to icreate 
        instances of `BandStructure` faster.
    save_wf : bool
        Whether wave functions should be kept as attribute after calculating 
        traces.
    verbosity : int, default=0
        Number controlling the verbosity. 
        0: minimalistic printing. 
        1: print info about decisions taken internally by the code, recommended 
        when the code runs without errors but the result is not the expected.
        2: print detailed info, recommended when the code stops with an error

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
        set, all k-points found in DFT files will be considered. ACCESSED 
        DIRECTLY BY BANDUPPY>=0.3.4. DO NOT CHANGE UNLESS NECESSARY. NOTIFY 
        THE DEVELOPERS IF ANY CHANGES ARE MADE.
    num_bandinvs : int
        Property that returns the number of inversion odd states in the 
        given TRIM.
    gap_direct : float
        Property that returns the smallest direct gap in the given k points.
    gap_indirect : float
        Property that returns the smallest indirect gap in the given k points.
    num_k : int
        Property that returns the number of k points in the attribute `kpoints`
    num_bands : int
        Property that returns the number of bands. Used to write trace.txt.
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
        calculate_traces=False,
        EF='0.0',
        onlysym=False,
        spin_channel=None,
        refUC = None,
        shiftUC = None,
        search_cell = False,
        trans_thresh=1e-5,
        degen_thresh=1e-8,
        save_wf=True,
        verbosity=0,
        alat=None,
        from_sym_file=None,
        normalize=True,
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
            self.Lattice, positions, typat = parser.parse_poscar(verbosity)
            if not onlysym:
                NK, NBin, self.Ecut0, lattice = parser.parse_header()
                if not np.allclose(self.Lattice, lattice):
                    raise RuntimeError("POSCAR and WAVECAR contain different lattices")
                EF_in = None  # not written in WAVECAR

        elif code == "abinit":

            parser = ParserAbinit(fWFK)
            (nband,
             NK,
             self.Lattice,
             self.Ecut0,
             self.spinor,
             typat,
             positions,
             EF_in) = parser.parse_header(verbosity=verbosity)
            NBin = max(nband)

        elif code == "espresso":

            parser = ParserEspresso(prefix)
            self.spinor = parser.spinor
            # alat is saved to be used to write the prefix.sym file
            self.Lattice, positions, typat, _alat = parser.parse_lattice()
            if alat is None:
                alat = _alat
            spinpol, self.Ecut0, EF_in, NK, NBin_list = parser.parse_header()

            # Set NBin
            if self.spinor and spinpol:
                raise RuntimeError("bandstructure cannot be both noncollinear and spin-polarised. Smth is wrong with the 'data-file-schema.xml'")
            elif spinpol:
                if spin_channel is None:
                    raise ValueError("Need to select a spin channel for spin-polarised calculations set  'up' or 'dw'")
                assert (spin_channel in ['dw','up'])
                if spin_channel == 'dw':
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
                              trans_thresh=trans_thresh,
                              verbosity=verbosity,
                              alat=alat,
                              from_sym_file=from_sym_file)
        if onlysym:
            return

        # Set Fermi energy
        if EF.lower() == "auto":
            if EF_in is None:
                self.efermi = 0.0
                msg = "WARNING : fermi-energy not found. Setting it as 0 eV"
                log_message(msg, verbosity, 1)
            else:
                self.efermi = EF_in
        else:
            try:
                self.efermi = float(EF)
            except:
                raise RuntimeError("Invalid value for keyword EF. It must be "
                                   "a number or 'auto'")

        log_message(f"Efermi: {self.efermi:.4f} eV", verbosity, 1)

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
        msg = ("WAVECAR contains {} k-points and {} bands.\n"
               "Saving {} bands starting from {} in the output"
               .format(NK, NBin, NBout, IBstart + 1))
        log_message(msg, verbosity, 1)
        msg = f"Energy cutoff in WAVECAR : {self.Ecut0}"
        log_message(msg, verbosity, 1)
        msg = f"Energy cutoff reduced to : {self.Ecut}"
        log_message(msg, verbosity, 1)

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
                msg = f'Parsing wave functions at k-point #{ik:>3d}'
                log_message(msg, verbosity, 2)
                WF, Energy, kpt, npw = parser.parse_kpoint(ik, NBin, self.spinor)
                kg = calc_gvectors(kpt,
                                   self.RecLattice,
                                   self.Ecut0,
                                   npw,
                                   self.Ecut,
                                   spinor=self.spinor,
                                   verbosity=verbosity
                                   )
                if not self.spinor:
                    selectG = kg[3]
                else:
                    selectG = np.hstack((kg[3], kg[3] + int(npw / 2)))
                WF = WF[:, selectG]

            elif code == 'abinit':
                NBin = parser.nband[ik]
                kpt = parser.kpt[ik]
                msg = f'Parsing wave functions at k-point #{ik:>3d}: {kpt}'
                log_message(msg, verbosity, 2)
                WF, Energy, kg = parser.parse_kpoint(ik)
                WF, kg = sortIG(ik, kg, kpt, WF, self.RecLattice, self.Ecut0, self.Ecut, self.spinor, verbosity=verbosity)

            elif code == 'espresso':
                msg = f'Parsing wave functions at k-point #{ik:>3d}'
                log_message(msg, verbosity, 2)
                WF, Energy, kg, kpt = parser.parse_kpoint(ik, NBin, spin_channel, verbosity=verbosity)
                WF, kg = sortIG(ik+1, kg, kpt, WF, self.RecLattice/2.0, self.Ecut0, self.Ecut, self.spinor, verbosity=verbosity)

            elif code == 'wannier90':
                kpt = kpred[ik]
                Energy = Energies[ik]
                ngx, ngy, ngz = parser.parse_grid(ik+1)
                kg = calc_gvectors(kpred[ik],
                                   self.RecLattice,
                                   self.Ecut,
                                   spinor=self.spinor,
                                   nplanemax=np.max([ngx, ngy, ngz]) // 2,
                                   verbosity=verbosity
                                   )
                selectG = tuple(kg[0:3])
                msg = f'Parsing wave functions at k-point #{ik:>3d}: {kpt}'
                log_message(msg, verbosity, 2)
                WF = parser.parse_kpoint(ik+1, selectG)

            # Pick energy of IBend+1 band to calculate gaps
            try:
                upper = Energy[IBend] - self.efermi
            except BaseException:
                upper = np.nan

            # Preserve only bands in between IBstart and IBend
            WF = WF[IBstart:IBend]
            Energy = Energy[IBstart:IBend] - self.efermi

            # saved to further use in Separate()
            self.kwargs_kpoint=dict(
                degen_thresh=degen_thresh,
                refUC=self.spacegroup.refUC,
                shiftUC=self.spacegroup.shiftUC,
                symmetries_tables=self.spacegroup.symmetries_tables,
                save_wf=save_wf,
                verbosity=verbosity,
                calculate_traces=calculate_traces,
                )

            kp = Kpoint(
                ik=ik,
                kpt=kpt,
                WF=WF,
                Energy=Energy,
                ig=kg,
                upper=upper,
                num_bands=NBout,
                RecLattice=self.RecLattice,
                symmetries_SG=self.spacegroup.symmetries,
                spinor=self.spinor,
                kwargs_kpoint=self.kwargs_kpoint,
                normalize=normalize,
                )
            self.kpoints.append(kp)
        del WF

    @property
    def num_k(self):
        '''Getter for the number of k points'''
        return len(self.kpoints)

    def identify_irreps(self, kpnames, verbosity=0):
        '''
        Identifies the irreducible representations of wave functions based on 
        the traces of symmetries of the little co-group. Each element of 
        `kpoints`  will be assigned the attribute `irreps` with the labels of 
        irreps.

        Parameters
        ----------
        kpnames : list
            List of labels of the maximal k points.
        verbosity : int, default=0
            Verbosity level. Default set to minimalistic printing
        '''

        for ik, KP in enumerate(self.kpoints):
            
            if kpnames is not None:
                irreps = self.spacegroup.get_irreps_from_table(kpnames[ik], KP.k, verbosity=verbosity)
            else:
                irreps = None
            KP.identify_irreps(irreptable=irreps)

    def write_characters(self):
        '''
        For each k point, write the energy levels, their degeneracies, traces 
        of the little cogroup's symmetries, and the direct and indirect gaps. 
        Also the irreps, if they have been identified. If the crystal is 
        inversion symmetries, the number of total inversion odd states, the 
        Z2 and Z4 numbers will be written.
        '''

        for KP in self.kpoints:

            # Print block of irreps and their characters
            KP.write_characters()

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
                print("Gap with upper bands: ", KP.upper - KP.Energy_mean[-1])
        
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
        '''
        Prepare a dictionary to save the data in JSON format.

        Parameters
        ----------
        kpnames : list
            List of labels of the maximal k points.

        Returns
        -------
        json_data : dict
            Dictionary with the data.
        '''

        kpline = self.KPOINTSline()
        json_data = {}
        json_data['kpoints line'] = kpline
        json_data['k points'] = []
        
        for ik, KP in enumerate(self.kpoints):
            json_kpoint = KP.json()
            json_kpoint['kp in line'] = kpline[ik]
            if kpnames is None:
                json_kpoint['kpname'] = None
            else:
                json_kpoint['kpname'] = kpnames[ik]
            json_data['k points'].append(json_kpoint)
        
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
        '''
        Getter for the direct gap

        Returns
        -------
        gap : float
            Smallest direct gap
        '''

        gap = np.inf
        for KP in self.kpoints:
            gap = min(gap, KP.upper-KP.Energy_mean[-1])
        return gap

    @property
    def gap_indirect(self):
        '''
        Getter for the indirect gap

        Returns
        -------
        gap : float
            Smallest indirect gap
        '''

        min_upper = np.inf  # smallest energy of bands above set
        max_lower = -np.inf  # largest energy of bands in the set
        for KP in self.kpoints:
            min_upper = min(min_upper, KP.upper)
            max_lower = max(max_lower, KP.Energy_mean[-1])
        return min_upper - max_lower

    @property
    def num_bandinvs(self):
        '''
        Getter for the total number of inversion odd states

        Returns
        -------
        num_bandinvs : int
            Total number of inversion odd states. 0 if the crystal is not 
            inversion symmetric.
        '''

        num_bandinvs = 0
        for KP in self.kpoints:
            if KP.num_bandinvs is not None:
                num_bandinvs += KP.num_bandinvs
        return num_bandinvs

    def write_irrepsfile(self):
        '''
        Write the file `irreps.dat` with the identified irreps.
        '''

        file = open('irreps.dat', 'w')
        for KP in self.kpoints:
            KP.write_irrepsfile(file)
        file.close()


    @property
    def num_bands(self):
        """
        Return number of bands. Raise RuntimeError if the number of bands 
        varies from on k-point to the other.

        Returns
        -------
        int
            Number of bands in every k-point.
        """
        nbarray = [k.num_bands for k in self.kpoints]
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

    def write_trace(self,):
        """
        Generate `trace.txt` file to upload to the program `CheckTopologicalMat` 
        in `BCS <https://www.cryst.ehu.es/cgi-bin/cryst/programs/topological.pl>`_ .
        """

        f = open("trace.txt", "w")
        f.write(
            (
                " {0}  \n"
                + " {1}  \n"  # Number of bands below the Fermi level  # Spin-orbit coupling. No: 0, Yes: 1
            ).format(self.num_bands, 1 if self.spinor else 0)
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
                    for x in KP.k
                )
                + "\n"
            )
        for KP in self.kpoints:
            f.write(
                KP.write_trace()
            )

    def Separate(self, isymop, groupKramers=True, verbosity=0):
        """
        Separate band structure according to the eigenvalues of a symmetry 
        operation.
        
        Parameters
        ----------
        isymop : int
            Index of symmetry used for the separation.
        groupKramers : bool, default=True
            If `True`, states will be coupled by Kramers' pairs.
        verbosity : int, default=0
            Verbosity level. Default is set to minimalistic printing

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
            kp.Separate(symop, groupKramers=groupKramers, verbosity=verbosity, kwargs_kpoint=self.kwargs_kpoint)
            for kp in self.kpoints
        ] # each element is a dict with separated bandstructure of a k-point

        allvalues = np.array(sum((list(kps.keys()) for kps in kpseparated), []))
        if groupKramers:
            allvalues = allvalues[np.argsort(np.real(allvalues))].real
            block_indices = get_block_indices(allvalues, thresh=0.01, cyclic=False)
            if len(block_indices) > 1:
                allvalues = set(
                    [allvalues[b1:b2].mean() for b1, b2 in block_indices]
                ) # unrepeated Re parts of all eigenvalues
                subspaces = {}
                for vv in allvalues:
                    other = copy.copy(self)
                    other.kpoints = []
                    for K in kpseparated:
                        vk = list(K.keys())
                        vk0 = vk[np.argmin(np.abs(vv - vk))]
                        if abs(vk0 - vv) < 0.05:
                            other.kpoints.append(K[vk0])
                    subspaces[vv] = other # unnecessary indent ?
                return subspaces
            else:
                return dict({allvalues.mean(): self})
        else:
            allvalues = allvalues[np.argsort(np.angle(allvalues))]
            log_message(f'allvalues: {allvalues}', verbosity, 2)
            block_indices = get_block_indices(allvalues, thresh=0.01, cyclic=True)
            nv = len(allvalues)
            if len(block_indices) > 1:
                allvalues = set( [ np.roll(allvalues, -b1)[: (b2 - b1) % nv].mean()
                                    for b1, b2 in block_indices ])
                log_message(f'Distinct values: {allvalues}', verbosity, 2)
                subspaces = {}
                for vv in allvalues:
                    other = copy.copy(self)
                    other.kpoints = []
                    for K in kpseparated:
                        vk = list(K.keys())
                        vk0 = vk[np.argmin(np.abs(vv - vk))]
                        if abs(vk0 - vv) < 0.05:
                            other.kpoints.append(K[vk0])
                        subspaces[vv] = other
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
            (np.min([k.Energy[1:nmax] for k in self.kpoints], axis=0), [np.inf])
        )
        emax = np.max([k.Energy[:nmax] for k in self.kpoints], axis=0)
        locgap = np.hstack(
            (
                np.min(
                    [k.Energy[1:nmax] - k.Energy[0 : nmax - 1] for k in self.kpoints],
                    axis=0,
                ),
                [np.inf],
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
        wilson = functools.reduce(
            np.dot,
            [functools.reduce(np.dot, np.linalg.svd(O)[0:3:2]) for O in overlaps],
        )
        return np.sort((np.angle(np.linalg.eig(wilson)) / (2 * np.pi)) % 1)

    def write_plotfile(self, filename='bands-tognuplot.dat'):
        """
        Generate lines for a band structure plot, with cummulative length of the
        k-path as values for the x-axis and energy-levels for the y-axis.

        Returns
        -------
        str
            Lines to write into a file that will be parsed to plot the band 
            structure.
        """

        kpline = self.KPOINTSline()

        # Prepare energies at each k point
        energies_expanded = np.full((self.num_bands, len(kpline)), np.inf)
        for ik, kp in enumerate(self.kpoints):
            count = 0
            for iset, deg in enumerate(kp.degeneracies):
                for i in range(deg):
                    energies_expanded[count,ik] = kp.Energy_mean[iset]
                    count += 1

        # Write energies of each band
        file = open(filename, 'w')
        file.write('column 1: k, column 2: energy in eV (w.r.t. Fermi level)')
        for iband in range(self.num_bands):
            file.write('\n')  # blank line separating blocks of k points
            for k, energy in zip(kpline, energies_expanded[iband]):
                s = '{:8.4f}    {:8.4f}\n'.format(k, energy)
                file.write(s)
        file.close()

    def KPOINTSline(self, kpred=None, supercell=None, breakTHRESH=0.1):
        """
        Calculate cumulative length along a k-path in cartesian coordinates.
        ACCESSED DIRECTLY BY BANDUPPY>=0.3.4. DO NOT CHANGE UNLESS NECESSARY. 
        NOTIFY THE DEVELOPERS IF ANY CHANGE IS MADE.

        Parameters
        ----------
        kpred : list, default=None
            Each element contains the direct coordinates of a k-point in the
            attribute `kpoints`.
        supercell : array, shape=(3,3), default=None
                Describes how the lattice vectors of the (super)cell used in the 
                calculation are expressed in the basis vectors of the primitive 
                cell. USED IN BANDUPPY. DO NOT CHANGE UNLESS NECESSARY.
        breakTHRESH : float, default=0.1
            If the distance between two neighboring k-points in the path is 
            larger than `breakTHRESH`, it is taken to be 0. Set `breakTHRESH` 
            to a large value if the unforlded kpoints line is continuous.

        Returns
        -------
        K : array
            Each element is the cumulative distance along the path up to a 
            k-point. The first element is 0, so that the number of elements
            matches the number of k-points in the path.
        """
        if kpred is None:
            kpred = [k.k for k in self.kpoints]
        if supercell is None:
            reciprocal_lattice = self.RecLattice
        else:
            reciprocal_lattice = supercell.T @ self.RecLattice 
        KPcart = np.dot(kpred, reciprocal_lattice)
        K = np.zeros(KPcart.shape[0])
        k = np.linalg.norm(KPcart[1:, :] - KPcart[:-1, :], axis=1)
        k[k > breakTHRESH] = 0.0
        K[1:] = np.cumsum(k)
        return K
