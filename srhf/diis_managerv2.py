import numpy as np
from dataclasses import dataclass
from bdmats import BDMatrix

@dataclass
class DIIS_HISTORY:
    iteration:int
    dRMS:float
    fock_hist:list
    error_hist:list
    
    def __repr__(self):
        #return f"Irrep: {self.irrep}\n dRMS: {self.dRMS} \n fock_history {self.fock_hist[i] for i in range(0, len(self.fock_hist))}"
        return f"\niteration: {self.iteration}\n dRMS: {self.dRMS} \n Fock History: \n {self.fock_hist} \n Error History: \n {self.error_hist}\n"
class DIIS_Manager():
    def __init__(self, symtext):
        print("DIIS_Mangager Class Constructed") 
        self.symtext = symtext
        self.ctab = symtext.character_table
        self.diis = DIIS_HISTORY(None, None, [], [])

    def do_diis(self, F, D, S, A, i):
        #create full mat from bdmat
        self.og_F = F
        self.irreplength = []
        for b, block in enumerate(self.og_F.blocks):
            #print(block.size)
            if len(block) == 0:
                self.irreplength.append(0)
            else:

                sf = int(np.sqrt(block.size))
                #print(sf)
                self.irreplength.append(sf)
        self.F = F.full_mat() 
        self.D = D.full_mat() 
        self.S = S.full_mat() 
        self.A = A.full_mat()
        error = np.einsum('ij, jk, kl->il', self.F, self.D, self.S) - np.einsum('ij, jk, kl->il', self.S, self.D, self.F)
        error = self.A.dot(error).dot(self.A)
        self.error = error 
        self.diis.iteration = i
        self.diis.error_hist.append(error)
        self.diis.fock_hist.append(self.F)
        
        self.diis.dRMS = np.mean(error ** 2) ** 0.5


        #convert back to bdmatrix and return
        self.F_bd = BDMatrix.full_to_bd(self.F, self.irreplength)

    def check_error(self):
        error = False
        for i, err in enumerate(self.diis.error_hist):
            if np.any(err > 0.1):
                error = True
        return error


    def create_b(self): 
        if self.diis.iteration >= 2:
            self.B = []
            diis_count = len(self.diis.fock_hist)
            if diis_count > 6:
                diis_count -= 1
                del self.diis.error_hist[0]
                del self.diis.fock_hist[0]
            B = np.zeros((diis_count + 1, diis_count + 1))
            B[-1, :] = -1
            B[:, -1] = -1
            B[-1, -1] = 0
            for num1, e1 in enumerate(self.diis.error_hist):
                for num2, e2 in enumerate(self.diis.error_hist):
                    if num2 > num1: continue
                    val = np.einsum('ij,ij->', e1, e2)
                    B[num1, num2] = val
                    B[num2, num1] = val

            # normalize
            B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

            # Build residual vector, [Pulay:1980:393], Eqn. 6, RHS
            resid = np.zeros(diis_count + 1)
            resid[-1] = -1

            # Solve Pulay equations, [Pulay:1980:393], Eqn. 6
            ci = np.linalg.solve(B, resid)

            # Calculate new fock matrix as linear
            # combination of previous fock matrices
            F = np.zeros_like(self.F)
            for num, c in enumerate(ci[:-1]):
                F += c * self.diis.fock_hist[num]
            return BDMatrix.full_to_bd(F, self.irreplength)
        else:
            return self.og_F
