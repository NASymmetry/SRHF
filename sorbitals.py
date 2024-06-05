from dataclasses import dataclass
import warnings
import numpy as np
from typing import Optional
@dataclass
class ORB:
    irrep:int
    #coeffs:np.array
    #energies:np.array
    degen:int
    ndocc_ir:int

class SOrbitals():
    def __init__(self, symtext, salcs, ndocc):
        self.symtext = symtext
        self.salcs = salcs
        self.ndocc = ndocc

    def flatten(self, listoflist):
        return [x for xs in listoflist for x in xs]

    def count_ndocc(self):
        docc = 0
        for i, ir_docc in enumerate(self.sorted_eval_degen):
            docc += ir_docc
            if docc == self.ndocc:
                return i
            elif docc > self.ndocc:
                if type(self.DOCC) is None:
                    raise ValueError('This likely means a degenerate orbital is partially occupied')
                else:
                    warnings.warn("The initial guess has partially occupied a degenerate orbital. The user-input occupation vector {DOCC} will be used... you have been warned.")
                    #raise Warning("The initial guess has partially occupied a degenerate orbital. The user-input occupation vector {DOCC} will be used... you have been warned.")
                    return i

    def ndocc_irrep(self, C, eps, DOCC = None):
        print(f"Inside ndocc irrep. There are {self.ndocc} orbitals to sum over")
        print(f"The task at hand is to figure out how many orbitals are occupied per irrep, taking into account degeneracy")
        self.DOCC = DOCC
        Eval_irreps = []
        Evals = []
        Eval_degen = []
        self.Orbs = []
        for ir, e_ir in enumerate(eps):
            #degen = self.symtext.chartable.irrep_dims[self.salcs.irreps[ir]]
            degen = [irrep.d for irrep in self.symtext.irreps]
            orbs = C.blocks[ir]
            self.Orbs.append(ORB(ir, degen, Optional))
            #eval_irreps = [ir for i in range(0, degen) for o in range(0, len(orbs))]
            #eval_degen = [degen for i in range(0, degen) for o in range(0, len(orbs))]
            eval_irreps = [ir for o in range(0, len(orbs))]
            eval_degen = [degen for o in range(0, len(orbs))]
            
            #flat_evals = self.flatten([e_ir for d in range(0, degen)])
            flat_evals = self.flatten([e_ir])
            Eval_irreps.append(eval_irreps)
            Evals.append(flat_evals)
            Eval_degen.append(eval_degen)
        #flatten them
        self.Eval_irreps = np.array(self.flatten(Eval_irreps))
        self.Evals = np.array(self.flatten(Evals))
        self.Eval_degen = np.array(self.flatten(Eval_degen))
        #sort them by eval size
        sort = np.argsort(self.Evals)
        self.sorted_evals = self.Evals[sort]
        self.sorted_irreps = self.Eval_irreps[sort]
        self.sorted_eval_degen = self.Eval_degen[sort]
        i = self.count_ndocc()
        docc_list = list(self.sorted_irreps[:i + 1])
        for o, orbs in enumerate(self.Orbs):
            ndocc_ir = docc_list.count(o)
            orbs.ndocc_ir = ndocc_ir
