import numpy as np
import psi4
from bdmats import BDMatrix
import time
from scipy.linalg import block_diag
"""
Just testing some things, realizing I need a more integral handling/tensor transformation routines
Coded up for sto-3g water only, because once it works for that system, every other test case is error free

KNOWN LIMITATION: run_symm() and run_symm_block() build one big block via
block_diag(occ_C, virt_C) + BDMatrix.full_to_bd(ERI, [nbfxns]), which assumes
so_orbitals.C spans the full nbfxns dimension per irrep. That assumption
breaks when options.exploit_degen=True on a point group with a genuinely
degenerate irrep (e.g. methane's T2, ammonia's E): so_orbitals.C is then
compressed down to irreplength (one representative per degenerate set) while
the raw ERI tensor stays at full nbfxns, so the einsum contraction raises a
shape-mismatch ValueError. Verified correct (vs. Psi4 conventional MP2, to
~1e-13 Eh) for exploit_degen=False and for abelian point groups (e.g. water,
C2v) where exploit_degen has no compressing effect. Properly supporting
exploit_degen=True with true degeneracy would require MP2 to consume the
degeneracy-aware, DPD-repacked integrals the way build_fock_blocky_sym does,
rather than the raw dense ERI -- not yet implemented.
"""


class MP2():
    def __init__(self, mymol, options, so_orbitals, ERI, repacked_bigERI):
        self.options = options
        self.molecule = mymol
        self.so_orbitals = so_orbitals
        self.ERI = ERI
        self.repacked_bigERI = repacked_bigERI
    
    def run_symm_block(self):
        print("MP2 in the block-symmetrized basis")
        nbfxns = self.so_orbitals.nbfxns
        C = self.so_orbitals.C
        occ_C = C.slicev2([":", ":ndocc_ir"], self.so_orbitals.Orbs)
        virt_C = C.slicev2([":", "ndocc_ir:"], self.so_orbitals.Orbs)
        self.ERI = BDMatrix.full_to_bd(self.ERI, [nbfxns])
        self.G = self.ERI.transpose((0,2,1,3))

        occ_C = BDMatrix.full_to_bd(block_diag(*[block for block in occ_C.blocks if len(block) != 0]), [nbfxns])
        virt_C = BDMatrix.full_to_bd(block_diag(*[block for block in virt_C.blocks if len(block) != 0]), [nbfxns])
        self.IJAB = self.ERI.einsum('mnrs,mI,nA,rJ,sB -> IAJB', self.ERI, occ_C, virt_C, occ_C, virt_C)

        occ = []
        virt = []
        for o, orb in enumerate(self.so_orbitals.Orbs):
            if len(self.so_orbitals.eps[o]) != 0:
                occ.append(self.so_orbitals.eps[o][:orb.ndocc_ir])
                virt.append(self.so_orbitals.eps[o][orb.ndocc_ir:])
        Eocc = np.concatenate(occ, axis=None).ravel()
        Evirt = np.concatenate(virt, axis=None).ravel()

        IJAB = self.IJAB.blocks[0]  # axis order (I, A, J, B)
        denom = (Eocc[:, None, None, None] + Eocc[None, None, :, None]
                 - Evirt[None, :, None, None] - Evirt[None, None, None, :])
        E_2 = np.sum(IJAB * (2 * IJAB - IJAB.swapaxes(1, 3)) / denom)
        return E_2   # Total MP2 Correlation Energy

    def run_symm(self):
        print("MP2 in the symmetrized basis")
        nbfxns = self.so_orbitals.nbfxns
        C = self.so_orbitals.C
        occ_C = C.slicev2([":", ":ndocc_ir"], self.so_orbitals.Orbs)
        virt_C = C.slicev2([":", "ndocc_ir:"], self.so_orbitals.Orbs)
        self.ERI = BDMatrix.full_to_bd(self.ERI, [nbfxns])
        self.G = self.ERI.transpose((0,2,1,3))

        occ_C = BDMatrix.full_to_bd(block_diag(*[block for block in occ_C.blocks if len(block) != 0]), [nbfxns])
        virt_C = BDMatrix.full_to_bd(block_diag(*[block for block in virt_C.blocks if len(block) != 0]), [nbfxns])
        self.IJAB = self.ERI.einsum('mnrs,mI,nA,rJ,sB -> IAJB', self.ERI, occ_C, virt_C, occ_C, virt_C)

        occ = []
        virt = []
        for o, orb in enumerate(self.so_orbitals.Orbs):
            if len(self.so_orbitals.eps[o]) != 0:
                occ.append(self.so_orbitals.eps[o][:orb.ndocc_ir])
                virt.append(self.so_orbitals.eps[o][orb.ndocc_ir:])
        Eocc = np.concatenate(occ, axis=None).ravel()
        Evirt = np.concatenate(virt, axis=None).ravel()

        IJAB = self.IJAB.blocks[0]  # axis order (I, A, J, B)
        denom = (Eocc[:, None, None, None] + Eocc[None, None, :, None]
                 - Evirt[None, :, None, None] - Evirt[None, None, None, :])
        E_2 = np.sum(IJAB * (2 * IJAB - IJAB.swapaxes(1, 3)) / denom)
        return E_2   # Total MP2 Correlation Energy

    def run(self):
        print("MP2, c1 symmetry only")
        C = self.so_orbitals.C 
        #occ_C = C.slicev2([":", ":ndocc_ir"], self.so_orbitals.Orbs)
        #virt_C = C.slicev2([":", "ndocc_ir:"], self.so_orbitals.Orbs)
        occ_C = C.slice([":", ":ndocc_ir"], self.so_orbitals.Orbs)
        virt_C = C.slice([":", "ndocc_ir:"], self.so_orbitals.Orbs)
        self.ERI = BDMatrix.full_to_bd(self.ERI, self.so_orbitals.irreplength)
        self.G = self.ERI.transpose((0,2,1,3))
        self.IJAB = self.ERI.einsum('mnrs,mI,nJ,rA,sB -> IJAB', self.G, occ_C, occ_C, virt_C, virt_C)

        ndocc = self.so_orbitals.Orbs[0].ndocc_ir
        Eocc = self.so_orbitals.eps[0][:ndocc]
        Evirt = self.so_orbitals.eps[0][ndocc:]

        IJAB = self.IJAB.blocks[0]  # axis order (I, J, A, B)
        denom = (Eocc[:, None, None, None] + Eocc[None, :, None, None]
                 - Evirt[None, None, :, None] - Evirt[None, None, None, :])
        E_2 = np.sum(IJAB * (2 * IJAB - IJAB.swapaxes(2, 3)) / denom)
        print(E_2)
        return E_2   # Total MP2 Correlation Energy
