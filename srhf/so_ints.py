import numpy as np
from bdmats import BDMatrix
import time

class  SO_Ints():
    def __init__(self, symtext, so_orbitals, salcs, options):
        self.symtext = symtext
        self.so_orbitals = so_orbitals
        self.salcs = salcs
        self.options = options

    def rank2_transform(self, A):
        """
        AO->SO transformation for one electron integrals
        """
        B = []
        for i, salc in enumerate(self.salcs.salc_sets):
            #print(salc)
            #print(salc[: so_orbitals.irreplength[i]])
            temp1 = np.einsum('uv,ui->iv', A, salc[:self.so_orbitals.irreplength[i]].T, optimize ='optimal') 
            temp = np.einsum('iv,vj->ij', temp1, salc[:self.so_orbitals.irreplength[i]].T, optimize ='optimal')
            B.append(temp)
        return BDMatrix(B)


    def rank4_transform(self, A):
        fstart = time.time()
        """
        AO->SO transformation for two electron integrals
        """
        first = True
        for i, salc in enumerate(self.salcs.salc_sets):
            if first:
                s = salc.T
                first = False
            else:
                if len(salc) == 0:
                    print("This boi empty")
                else:
                    s = np.concatenate((s,salc.T), axis=1)
        temp1 = np.einsum("PQRS,Pp->pQRS", A, s, optimize='optimal')
        temp2 = np.einsum("pQRS,Qq->pqRS", temp1, s, optimize='optimal')
        temp3 = np.einsum("pqRS,Rr->pqrS", temp2, s, optimize='optimal')
        E = np.einsum("pqrS,Ss->pqrs", temp3, s, optimize='optimal')
        ffinish = time.time()
        print(f"Total ERI transform time took {ffinish - fstart:6.8f} seconds")
        print("SO INTS E FULL")
        print(E)
        self.blocks = self.get_blocks()
        E = self.indices(E)
        return E
    def indices(self, A):
        before = time.time()
        twod_tensor = []
        tot = 0
        for block in self.blocks:
            i_idx, j_idx, k_idx, l_idx = self.salcs.salcs_by_irrep[block[0]], self.salcs.salcs_by_irrep[block[1]], self.salcs.salcs_by_irrep[block[2]], self.salcs.salcs_by_irrep[block[3]]
            twod_tensor_b = np.zeros((len(i_idx) * len(j_idx), len(k_idx) * len(l_idx)))
            for i, ib in enumerate(i_idx): 
                for j, jb in enumerate(j_idx): 
                   for k, kb in enumerate(k_idx): 
                       for l, lb in enumerate(l_idx):
                           ij = len(j_idx) * i + j
                           kl = len(l_idx) * k + l
                           twod_tensor_b[ij,kl] = A[ib,jb,kb,lb]
            twod_tensor.append(twod_tensor_b)
        now = time.time()
        print(f"Manual Repacking Time for block {block} of shape {twod_tensor_b.shape} took {now - before:6.8f} seconds")

        return twod_tensor
     
    def get_blocks(self):
        before = time.time()
        blocks = []
        for i in range(0, len(self.salcs.salcs_by_irrep)):
            if len(self.salcs.salcs_by_irrep[i]) != 0:
                for j in range(0, len(self.salcs.salcs_by_irrep)):
                    if len(self.salcs.salcs_by_irrep[j]) != 0:
                        if self.dp_contains_tsir(i, j):
                            for k in range(0, len(self.salcs.salcs_by_irrep)):
                                if len(self.salcs.salcs_by_irrep[k]) != 0:
                                    for l in range(0, len(self.salcs.salcs_by_irrep)):
                                        if len(self.salcs.salcs_by_irrep[l]) != 0:
                                            if self.dp_contains_tsir(k, l):
                                               blocks.append([i, j, k, l])
        now = time.time()
        print(f"Total sym block lookup time took {now - before:6.8f} seconds")
        return blocks
    def fourd_2d(self, b, A):
        before = time.time()
        i_idx = self.salcs.salcs_by_irrep[b[0]]
        j_idx = self.salcs.salcs_by_irrep[b[1]]
        k_idx = self.salcs.salcs_by_irrep[b[2]]
        l_idx = self.salcs.salcs_by_irrep[b[3]]
        twod = np.zeros((len(i_idx) **2, len(k_idx) **2))
        for i in range(0, len(i_idx)):
            for j in range(0, len(j_idx)):
                for k in range(0, len(k_idx)):
                    for l in range(0, len(l_idx)):
                        #print(f"{i,j,k,l}") 
                        ij = len(j_idx) * i + j
                        kl = len(l_idx) * k + l
                        twod[ij,kl] = A[i,j,k,l]
        now = time.time()
        print(f"Repack for eri block {b} took {now - before:6.8f} seconds")
        return twod
    def rank4_block(self, b, A):
        before = time.time()
        
        i,j,k,l = b[0], b[1], b[2], b[3]
        IS = self.salcs.salc_sets[i].T 
        JS = self.salcs.salc_sets[j].T 
        KS = self.salcs.salc_sets[k].T 
        LS = self.salcs.salc_sets[l].T
  
        temp1 = np.einsum("PQRS,Pp->pQRS", A, IS, optimize='optimal')
        temp2 = np.einsum("pQRS,Qq->pqRS", temp1, JS, optimize='optimal')
        temp3 = np.einsum("pqRS,Rr->pqrS", temp2, KS, optimize='optimal')
        E = np.einsum("pqrS,Ss->pqrs", temp3, LS, optimize='optimal')
        now = time.time()
        print(f"Transform for eri block {b} took {now - before:6.8f} seconds")
        TD = True
        if TD:
            return self.fourd_2d(b, E)
        else:
            return E
    def transform(self, A):
        self.intlist = []
        for b in self.blocks:
            self.intlist.append(self.rank4_block(b, A))
         
    def rank4_transf_alt(self, A):
        fstart = time.time()
        """
        AO->SO transformation for two electron integrals
        """
        #first, fetch nonzero blocks to transform ERI by
        self.blocks = self.get_blocks()
        #next, look through list and transform ERI?
        self.transform(A)     
        return self.intlist
 
    def dp_contains_tsir(self, a, b, *args):
        #ctab = self.symtext.chartable
        ctab = self.symtext.character_table
        a = ctab.characters[a]
        b = ctab.characters[b]
        chars = a * b
        for arg in args:
            chars *= ctab.characters[arg]
        s = sum(chars * ctab.class_orders * ctab.characters[0])
        n = s / sum(ctab.class_orders)
        if np.isclose(n, 0, atol = 1e-4):
            return False
        return True
