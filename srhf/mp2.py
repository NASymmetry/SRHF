import numpy as np
import psi4
from bdmats import BDMatrix
import time
from scipy.linalg import block_diag
"""
Just testing some things, realizing I need a more integral handling/tensor transformation routines
Coded up for sto-3g water only, because once it works for that system, every other test case is error free
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
        C = self.so_orbitals.C 
        occ_C = C.slicev2([":", ":ndocc_ir"], self.so_orbitals.Orbs)
        virt_C = C.slicev2([":", "ndocc_ir:"], self.so_orbitals.Orbs)
        self.ERI = BDMatrix.full_to_bd(self.ERI, [7])
        self.G = self.ERI.transpose((0,2,1,3))

        occ_C = BDMatrix.full_to_bd(block_diag(*[block for block in occ_C.blocks if len(block) != 0]), [7])
        virt_C = BDMatrix.full_to_bd(block_diag(*[block for block in virt_C.blocks if len(block) != 0]),[7])
        self.IJAB = self.ERI.einsum('mnrs,mI,nA,rJ,sB -> IAJB', self.ERI, occ_C, virt_C, occ_C, virt_C)
        
        occ = []
        virt = []
        for o, orb in enumerate(self.so_orbitals.Orbs):
            if len(self.so_orbitals.eps[o]) != 0:
                occ.append(self.so_orbitals.eps[o][:orb.ndocc_ir])
                virt.append(self.so_orbitals.eps[o][orb.ndocc_ir:])
        Eocc = np.concatenate(occ, axis=None).ravel()
        Evirt = np.concatenate(virt, axis=None).ravel()
        E_2 = 0
        for i in range(5):  
                for j in range(5): 
                        for a in range(2):
                                for b in range(2):
                                        E_2 += (self.IJAB.blocks[0][i,a,j,b] * ((2 * self.IJAB.blocks[0][i,a,j,b]) - self.IJAB.blocks[0][i,b,j,a])) / (Eocc[i] + Eocc[j] - Evirt[a] - Evirt[b])
        return E_2   # Total MP2 Correlation Energy
    
    def run_symm(self):
        print("MP2 in the symmetrized basis")
        C = self.so_orbitals.C 
        occ_C = C.slicev2([":", ":ndocc_ir"], self.so_orbitals.Orbs)
        virt_C = C.slicev2([":", "ndocc_ir:"], self.so_orbitals.Orbs)
        self.ERI = BDMatrix.full_to_bd(self.ERI, [7])
        self.G = self.ERI.transpose((0,2,1,3))

        occ_C = BDMatrix.full_to_bd(block_diag(*[block for block in occ_C.blocks if len(block) != 0]), [7])
        virt_C = BDMatrix.full_to_bd(block_diag(*[block for block in virt_C.blocks if len(block) != 0]),[7])
        self.IJAB = self.ERI.einsum('mnrs,mI,nA,rJ,sB -> IAJB', self.ERI, occ_C, virt_C, occ_C, virt_C)
        
        occ = []
        virt = []
        for o, orb in enumerate(self.so_orbitals.Orbs):
            if len(self.so_orbitals.eps[o]) != 0:
                occ.append(self.so_orbitals.eps[o][:orb.ndocc_ir])
                virt.append(self.so_orbitals.eps[o][orb.ndocc_ir:])
        Eocc = np.concatenate(occ, axis=None).ravel()
        Evirt = np.concatenate(virt, axis=None).ravel()
        E_2 = 0
        for i in range(5):  
                for j in range(5): 
                        for a in range(2):
                                for b in range(2):
                                        E_2 += (self.IJAB.blocks[0][i,a,j,b] * ((2 * self.IJAB.blocks[0][i,a,j,b]) - self.IJAB.blocks[0][i,b,j,a])) / (Eocc[i] + Eocc[j] - Evirt[a] - Evirt[b])
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

        ABIJ = self.IJAB.transpose((2,3,0,1))
        
        print(self.so_orbitals.eps)
        Eocc = self.so_orbitals.eps[0][:5]
        Evirt = self.so_orbitals.eps[0][5:]
        E_2 = 0
        for i in range(5): 
                for j in range(5): 
                        for a in range(2):
                                for b in range(2):
                                        E_2 += (self.IJAB.blocks[0][i,j,a,b] * ((2 * ABIJ.blocks[0][a,b,i,j]) - ABIJ.blocks[0][b,a,i,j])) / (Eocc[i] + Eocc[j] - Evirt[a] - Evirt[b])
        print(E_2)
        return E_2   # Total MP2 Correlation Energy

        print(stop)
        Eocc = self.energies[:self.nocc]   # Orbital energies of occupied orbitals
        Evir = self.energies[self.nocc:]   # Orbital energies of virtual orbitals
        E_2 = 0
        for i in range(self.nocc): 
                for j in range(self.nocc): 
                        for a in range(self.nvir):
                                for b in range(self.nvir):
                                        E_2 += (self.IJAB[i,j,a,b] * ((2 * ABIJ[a,b,i,j]) - ABIJ[b,a,i,j])) / (Eocc[i] + Eocc[j] - Evir[a] - Evir[b])
        return E_2   # Total MP2 Correlation Energy
