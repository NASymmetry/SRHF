#Holds all the classes for hf.py, formally their own python modules
from dataclasses import dataclass
import warnings
import numpy as np
from typing import Optional
import time
import copy
import psi4
from srhf.bdmats import BDMatrix
from copy import deepcopy
import scipy
#import srhf.atomic_configurations
from srhf import atomic_configurations
@dataclass
class ORB:
    irrep:int
    degen:int
    ndocc_ir:int
    nvirt_ir:int

#@dataclass
#class salc_element:
#    i_idx:int
#    s_idx:int
#    element:float

@dataclass
class salc_element:
    i_idx:int #index of the nonzero element within the salc
    s_idx:int #index of the salc within the irrep
    element:float #nonzero element of the salc
    i:int #PF index

class SOrbitals():
    def __init__(self, symtext, salcs, ndocc, options, nbfxns, fxn_list, basis, mol, basis_obj, bset):
        self.symtext = symtext
        self.salcs = salcs
        self.ndocc = ndocc
        self.options = options
        self.nbfxns = nbfxns
        self.fxn_list = fxn_list
        self.basis = basis
        self.process_salcs()
        self.mol = mol
        self.basis_obj = basis_obj
        self.bset = bset
        
        ints = psi4.core.MintsHelper(self.basis)
        S = ints.ao_overlap().np
        T = ints.ao_kinetic().np
        V = ints.ao_potential().np
        self.S = self.ao_to_so(S)
        #self.test_S = self.minimal_ao_to_so(S) 
        #self.find_sparse_salcs(1e-5)
        #self.test_S = self.sparse_twoD_transform(S)
        T = self.ao_to_so(T) 
        V = self.ao_to_so(V)
        #np.savez(
        #"nh3_ccpvdz_so_integrals.npz",
        #S_blocks=np.array(self.S.blocks, dtype=object),
        #T_blocks=np.array(T.blocks, dtype=object),
        #V_blocks=np.array(V.blocks, dtype=object),
        #)

        #self.S = np.array(self.S, dtype=object)
        #T = np.array(T, dtype=object)
        #V = np.array(V, dtype=object)
        #np.savez(
        #"nh3_so_integrals.npz",
        #S_blocks = self.S,
        #T_blocks = T,
        #V_blocks = V,
        #)
        #Do other stuff before TEI transform
        if self.options.guess == "core":
            #C, A, eps = self.rhf_core_guess(S, T, V)
            C, self.A, eps = self.rhf_core_guessv2(self.S, T, V)
        elif self.options.guess == "gwh":
            C, self.A, eps = self.gwh_guess(self.S, T, V)
        elif self.options.guess == "sad":
            self.D = self.sad_guess_v2(self.S, T, V)
            #print("self.D type")
            #D_blocks = np.array(self.D.blocks, dtype=object)
            #print(type(self.D))
            #np.savez(
            #"nh3_sad_data.npz",
            #D_blocks = D_blocks,
            #)
            A = []
            for i, s in enumerate(self.S.blocks):
                if len(s) == 0:
                    A.append(np.array([]))
                    continue
                else:
                    A.append(self.normalize(s))

            self.A = BDMatrix(A)
            self.Orbs = []
            for ir, d_ir in enumerate(self.D.blocks):
                degen = self.symtext.irreps[ir].d
                self.Orbs.append(ORB(ir, degen, None, None))
        self.H = T + V
        print("The initial core Hamiltonian")
        print(self.H)
        if self.options.guess == "sad":
            self.C = None
            self.eps = None
        else:
            self.ndocc_irrep(C, eps)
            self.C = C
            self.eps = eps
    
    #def get_sad_fock(self, vHF):
    #    print("inside get sad fock")
    #    h1e = self.T + self.V
    #    f = h1e + vHF
    #    return f

    def sad_unique_atoms(self):
        self.atomic_densities = {}
        natom = self.mol.natom()
        for n in range(natom):
            self.atomic_densities[self.mol.symbol(n)] = [self.bset[n], np.array([])]

    def construct_dm_from_SAD(self):
        densities = []
        for n in range(self.mol.natom()):
            densities.append(self.atomic_densities[self.mol.symbol(n)][1])
        if len(densities) != self.mol.natom():
            raise ValueError
        else:
            densities = scipy.linalg.block_diag(*densities)
        return densities
         

    def sad_guess_v2(self, S, T, V):
        print("Begin SAD Guess")
        self.sad_unique_atoms()
        for s_idx, atom_symbol in enumerate(self.atomic_densities):
            atom_psi4 = psi4.geometry(atom_symbol)
            self.atom_basis = psi4.core.BasisSet.build(atom_psi4, 'BASIS', self.basis_obj, puream = True)
            atom_ints = psi4.core.MintsHelper(self.atom_basis)
            S = atom_ints.ao_overlap().np
            T = atom_ints.ao_kinetic().np
            V = atom_ints.ao_potential().np
            I = atom_ints.ao_eri().np
            if str(atom_symbol) == 'H':
                h1e = T + V
                energies, coeffs = self.eig(h1e, S, atom_symbol)
                mo_occ = np.zeros_like(energies)
                ncore = 0
                nopen = 1
                open_idx = []
                core_sort = np.argsort(energies)
                core_idx = core_sort[:ncore]
                if nopen > 0:
                    open_idx = core_sort[ncore:]
                    open_sort = np.argsort(energies[open_idx])
                    open_idx = open_idx[open_sort[:nopen]]
                mo_occ[core_idx] = 2
                mo_occ[open_idx] = 1
                dm = np.dot(coeffs*mo_occ, coeffs.T)
                self.atomic_densities[atom_symbol][1] = dm
                continue
               
            F = T + V
            a = self.normalize(S)
            Ft = a.dot(F).dot(a)
            En, Ct = np.linalg.eigh(Ft)
            pre_ao_ang = self.atomic_densities[atom_symbol][0]
            pre_ao_ang = np.asarray(pre_ao_ang, dtype=int)
            ao_ang = np.repeat(pre_ao_ang, 2 * pre_ao_ang + 1)
            nao = len(ao_ang)
            l_max = 8
            mo_energy = []
            mo_coeff = []
            for l in range(l_max):
                degen = 2 * l + 1
                idx = np.where(ao_ang == l)[0]
                nao_l = len(idx)
                if nao_l > 0:
                    nsh = nao_l // degen
                    f_l = F[idx[:,None],idx].reshape(nsh, degen, nsh, degen)
                    s_l = S[idx[:,None],idx].reshape(nsh, degen, nsh, degen)
                    #avg over angular parts
                    f_l = np.einsum('piqi->pq', f_l) / degen
                    s_l = np.einsum('piqi->pq', s_l) / degen
                    e, c = scipy.linalg.eigh(f_l, s_l)
                    eig_idx = np.argmax(abs(c.real), axis=0)
                    c[:,c[eig_idx,np.arange(len(e))].real<0] *= -1
                    mo_energy.append(np.repeat(e, degen))
                    mo = np.zeros((nao, nsh, degen))
                    for i in range(degen):
                        mo[idx[i::degen], :, i] = c
                    mo_coeff.append(mo.reshape(nao, nao_l))
            energies = np.hstack(mo_energy) 
            coeffs = np.hstack(mo_coeff)
            occ = []
            for l in range(l_max):
                nuc = int(self.mol.charge(0))
                n2occ, frac = self.frac_occ(atom_symbol, l, nuc, atomic_configurations)
                degen = 2 * l + 1
                idx = np.array(self.atomic_densities[atom_symbol][0]) == l
                nbas_l = len(np.array(self.atomic_densities[atom_symbol][0])[idx])
                if l < 4:
                    assert n2occ <= nbas_l
                    #skip ecp treatment, we don't use them yet...
                    occ_l = np.zeros(nbas_l)
                    occ_l[:n2occ] = 2
                    if frac > 0:
                        occ_l[n2occ] = frac
                    occ.append(np.repeat(occ_l, degen))
                else:
                    occ.append(np.zeros(nbas_l * degen))
            mo_occ = np.hstack(occ)
            mocc = coeffs[:,mo_occ>0]

            dm = np.dot(mocc*mo_occ[mo_occ>0], mocc.T)
             
            h = T + V
            Ji = np.einsum('pqrs,rs->pq',I,dm)
            Ki = np.einsum('prqs,rs->pq',I,dm) #permute the indices here, b/c permuted in 2-e integral?
            #vhf = Ji - Ki * .5
            vhf = 2*Ji - Ki
            f = h + vhf
            E = (np.einsum('rs,rs->',dm,h) + np.einsum('rs,rs->',dm,f)) * .5 # arrow for scalar
            scf_conv = False
            for cycle in range(1, self.options.sad_cycles +1):
                dm_last = dm
                last_hf_e = E
                f = h + vhf
                energies, coeffs = self.eig(f, S, atom_symbol)
                mo_occ = self.get_occ(energies, coeffs, atom_symbol)
                mocc = coeffs[:,mo_occ>0]
                dm = np.dot(mocc*mo_occ[mo_occ>0], mocc.T)
                Ji = np.einsum('pqrs,rs->pq',I,dm)
                Ki = np.einsum('prqs,rs->pq',I,dm) #permute the indices here, b/c permuted in 2-e integral?
                #vhf = Ji - Ki * .5
                vhf = 2*Ji - Ki
                f = h + vhf
                E = (np.einsum('rs,rs->',dm,h) + np.einsum('rs,rs->',dm,f)) * .5 # arrow for scalar
                norm_ddm = np.linalg.norm(dm - dm_last)
                delta_E = np.format_float_scientific(np.absolute(last_hf_e) - np.absolute(E), unique=False, precision=15) #Formats delta_E to scientific notation, takes difference of iteration energies
                if float(delta_E) < 1e-8 and float(norm_ddm) < 1e-6:
                    scf_conv = True 
                if scf_conv:
                    print(f"SCF FINISHED in {cycle} iterations!!!")
                    self.atomic_densities[atom_symbol][1] = dm
                    break
                if cycle == 20 and not scf_conv:
                    print("SAD cycles did not converge")
                    break
        densities = self.construct_dm_from_SAD()
        print(f"The SAD Density Matrix") 
        densities = self.ao_to_so(densities)
        print(f"The Symmetry-Adapted SAD Density Matrix")
        print(densities) 
        return densities


    def get_occ(self, energies, coeffs, atom_symbol):
        print("inside get_occ")
        l_max = 8
        atom_symbol
        occ = []
        for l in range(l_max):
            nuc = int(self.mol.charge(0))
            n2occ, frac = self.frac_occ(atom_symbol, l, nuc, atomic_configurations)
            degen = 2 * l + 1
            idx = np.array(self.atomic_densities[atom_symbol][0]) == l
            nbas_l = len(np.array(self.atomic_densities[atom_symbol][0])[idx])
            if l < 4:
                assert n2occ <= nbas_l
                #skip ecp treatment, we don't use them yet...
                occ_l = np.zeros(nbas_l)
                occ_l[:n2occ] = 2
                if frac > 0:
                    occ_l[n2occ] = frac
                occ.append(np.repeat(occ_l, degen))
            else:
                occ.append(np.zeros(nbas_l * degen))
        mo_occ = np.hstack(occ)
        #do density in here too
        return mo_occ

    def eig(self, F, S, atom_symbol):
        pre_ao_ang = self.atomic_densities[atom_symbol][0]
        pre_ao_ang = np.asarray(pre_ao_ang, dtype=int)
        ao_ang = np.repeat(pre_ao_ang, 2 * pre_ao_ang + 1)
        nao = len(ao_ang)
        l_max = 8
        mo_energy = []
        mo_coeff = []
        for l in range(l_max):
            degen = 2 * l + 1
            idx = np.where(ao_ang == l)[0]
            nao_l = len(idx)
            if nao_l > 0:
                nsh = nao_l // degen
                f_l = F[idx[:,None],idx].reshape(nsh, degen, nsh, degen)
                s_l = S[idx[:,None],idx].reshape(nsh, degen, nsh, degen)
                #avg over angular parts
                f_l = np.einsum('piqi->pq', f_l) / degen
                s_l = np.einsum('piqi->pq', s_l) / degen
                e, c = scipy.linalg.eigh(f_l, s_l)
                print(f"The e and c {e} {c}")
                eig_idx = np.argmax(abs(c.real), axis=0)
                c[:,c[eig_idx,np.arange(len(e))].real<0] *= -1
                mo_energy.append(np.repeat(e, degen))
                mo = np.zeros((nao, nsh, degen))
                for i in range(degen):
                    mo[idx[i::degen], :, i] = c
                mo_coeff.append(mo.reshape(nao, nao_l))
        energies = np.hstack(mo_energy) 
        coeffs = np.hstack(mo_coeff)
        return energies, coeffs


 
    def frac_occ(self, symb, l, nuc, atomic_configurations):
        #print("inside frac occ")
        #print(atomic_configurations.atom_configs(nuc))
        #print(atomic_configurations.atom_configs(nuc)[l])
        #print(f"The l val {l}")
        #print(f"The nuc {nuc}")
        #print(atomic_configurations.atom_configs(nuc)[l]) 
        if l < 4 and atomic_configurations.atom_configs(nuc)[l] > 0:
            ne = atomic_configurations.atom_configs(nuc)[l]
            nd = (l * 2 + 1) * 2
            ndocc = ne.__floordiv__(nd)
            frac = (float(ne) / nd - ndocc) * 2
        else:
            ndocc = frac = 0
        #print(f"ndocc frac {ndocc} {frac}")
        return ndocc, frac 
        
    

    def rhf_core_guessv2(self, S, T, V):
        A = []
        F = T + V
        Ct = []
        En = []
        for i, s in enumerate(S.blocks):
            if len(s) == 0:
                A.append(np.array([]))
                En.append(np.array([]))
                Ct.append(np.array([]))
            else:
                a = self.normalize(s)
                #a = fractional_matrix_power(s, -0.5) 
                A.append(a)
        A = BDMatrix(A)
        Ft = A.dot(F).dot(A)
        En, Ct = Ft.eigh() 
        print("Initial Orbital Core Guess")
        print(En)
        C = A.dot(Ct)
        return C, A, En
    
    def normalize(self, s):
        normlist = 1.0 / np.sqrt(np.diag(s))
        over = np.outer(normlist, normlist) * s
        eigval, U = np.linalg.eigh(over)
        Us = (U / np.sqrt(eigval)[np.newaxis, :]) * normlist[:, np.newaxis]
        anti = Us @ U.T
        return anti
    
    def gwh_guess(self, S, T, V):
        F = []
        for I, X in enumerate(S.blocks):
            if len(X) == 0:
                F.append(np.array([]))
            else:
                F.append(np.zeros((len(X),len(X))))
        F = BDMatrix(F)
        
        gwh_k = 1.75
        dummy = 0.0
        H = T + V

        A = []
        Ct = []
        En = []
        for i, s in enumerate(S.blocks):
            if len(s) == 0:
                A.append(np.array([]))
                Ct.append(np.array([]))
                En.append(np.array([]))
                continue
            else:
                h_diag = np.diag(H.blocks[i])
                F.blocks[i] = 0.5 * gwh_k * np.add.outer(h_diag, h_diag) * s
                np.fill_diagonal(F.blocks[i], h_diag)
                #construct A, get intial orbital energies
                a = self.normalize(s)
                A.append(a)
                en_i, ct_i = np.linalg.eigh(F.blocks[i])
                Ct.append(ct_i)
                En.append(en_i)
        print("Initial energies from the GWH Guess")
        print(En)

        A = BDMatrix(A)
        Ct = BDMatrix(Ct)
        Ft = A.dot(F).dot(A)
        C = A.dot(Ct)
        return C, A, En

    def ao_to_mo(self, A, B, C):
        """
        General transformation of rank to operator where A is the matrix to be transformed,
        B and C are the left and right hand transformation matrices
        """
        B = []
        for h, c in enumerate(C.blocks):
            if len(c) == 0:
                B.append(np.array([]))
            else:
                string1 = 'ui,uv->iv'
                string2 = 'iv,vj->ij'
                test = np.einsum(string1, c, A.blocks[h], optimize ='optimal')
                test2 = np.einsum(string2, test, c, optimize ='optimal')
                idk =  np.einsum('ui,uv->iv', c, A.blocks[h], optimize ='optimal')
                idk1 =  np.einsum('iv,vj->ij',idk, c, optimize ='optimal')
                B.append(idk1)
        return BDMatrix(B)

    #def find_sparse_salcs(self, threshold):
    #    self.sparse_salcs = []
    #    for h, salc in enumerate(self.salcs.salc_sets):
    #        salcs = []
    #        for s_i, s in enumerate(salc[:self.irreplength[h]]):
    #            overall_s_idx = self.salcs.salcs_by_irrep[h][s_i]
    #            for i in range(len(s)):
    #                if abs(s[i]) > threshold:
    #                    salcs.append(salc_element(i, s_i, s[i], self.salcs.salcs[overall_s_idx].i))
    #        self.sparse_salcs.append(salcs)


    #def iterate_through_sblock(self, h, salc, A):
    #    B = np.zeros((len(salc),len(A)))
    #    for i in range(len(salc)):
    #        for j in range(salc.shape[1]):
    #            for k in range(salc.shape[1]):
    #                if abs(salc[i,k]) > 1e-5:
    #                    B[i,j] += salc[i,k] * A[k,j]
    #    

    #def minimal_ao_to_so(self, A):
    #    for h_i, salc in enumerate(self.salcs.salc_sets):
    #        self.iterate_through_sblock(h_i, salc, A)
    #    return BDMatrix(B)

    #def sparse_twoD_transform(self, A):
    #    #print("Sparse TwoD Transform")
    #    B = []
    #    for h, salcs in enumerate(self.sparse_salcs):
    #        temp = np.zeros((len(self.salcs.salc_sets[h][:self.irreplength[h]]), len(self.salcs.salc_sets[h][:self.irreplength[h]])))
    #        #print(temp.shape)
    #        #for i, salc_i in enumerate(salcs[:self.irreplength[h]]):
    #        for i, salc_i in enumerate(salcs):
    #            for j, salc_j in enumerate(salcs):
    #                temp[salc_i.s_idx, salc_j.s_idx] += A[salc_i.i_idx, salc_j.i_idx] * salc_i.element * salc_j.element
    #        B.append(temp)
    #    return BDMatrix(B)

    #def sparse_fourD_transform(self, A):
    #    print("Inside sparse FourD Transform")
    #    print("The eri dog")
    #    print(A)
    #    #first, flatten self.sparse_salcs
    #    print(self.sparse_salcs)
    #    #redefine sparse_salcs because we are doing the full ERI this time, so ...
    #    sparse_salcs_full = []
    #    offset = 0
    #    for h, salc in enumerate(self.salcs.salc_sets):
    #        for s_i, s in enumerate(salc):
    #            for I in range(len(s)):
    #                if abs(s[I]) > 1e-5:
    #                    sparse_salcs_full.append(salc_element(I, s_i + offset, s[I], salc.i))
    #        offset += self.irreplength[h]
    #    print("sparse salcs full")
    #    print(sparse_salcs_full)
    #    print(len(sparse_salcs_full))
    #    #flat_s = self.flatten(self.sparse_salcs)
    #    temp = np.zeros((A.shape))
    #    for i, salc_i in enumerate(sparse_salcs_full):
    #        for j, salc_j in enumerate(sparse_salcs_full):
    #            for k, salc_k in enumerate(sparse_salcs_full):
    #                for l, salc_l in enumerate(sparse_salcs_full):
    #                    #print(f"i j k l {i, j, k, l}")
    #                    temp[salc_i.s_idx, salc_j.s_idx, salc_k.s_idx, salc_l.s_idx] += A[salc_i.i_idx, salc_j.i_idx, salc_k.i_idx, salc_l.i_idx] * salc_i.element * salc_j.element * salc_k.element * salc_l.element
    #    print(temp)
    #    #B.append(temp)
    #    return BDMatrix(B)
                    


    def ao_to_so(self, A):
        """
        AO->SO transformation for one electron integrals
        """
        B = []
        for i, salc in enumerate(self.salcs.salc_sets):
            temp1 = np.einsum('uv,ui->iv', A, salc[:self.irreplength[i]].T, optimize ='optimal') 
            temp = np.einsum('iv,vj->ij', temp1, salc[:self.irreplength[i]].T, optimize ='optimal')
            #print(f"The temp {temp} {type(temp)}")
            B.append(temp)
        return BDMatrix(B)
    
    def flatten(self, listoflist):
        return [x for xs in listoflist for x in xs]

    def count_ndocc(self):
        docc = 0
        for i, ir_docc in enumerate(self.sorted_eval_degen):
            docc += ir_docc
            if docc == self.ndocc:
                return i
            elif docc > self.ndocc:
                if type(self.options.docc) is None:
                    raise ValueError('This likely means a degenerate orbital is partially occupied')
                else:
                    warnings.warn("The initial guess has partially occupied a degenerate orbital. The user-input occupation vector {DOCC} will be used... you have been warned.")
                    #raise Warning("The initial guess has partially occupied a degenerate orbital. The user-input occupation vector {DOCC} will be used... you have been warned.")
                    return i

    def process_salcs(self):
        #TEMP self.partner_functions()
        print(self.salcs.salc_sets)
        self.irreplength = []
        if self.options.exploit_degen:
            for s, SET in enumerate(self.salcs.salc_sets):
                degen = self.symtext.irreps[s].d
                #degen = self.symtext.chartable.irrep_dims[self.salcs.irreps[s]]
                if SET.shape[0] == 0:
                    self.irreplength.append(0)
                else:
                    self.irreplength.append(SET.shape[0] // degen)
        else: 
            for SET in self.salcs.salc_sets:
                self.irreplength.append(SET.shape[0])
    
    #a super convoluted function that counts the number of occupied and virtual orbitals per irrep
    #also accounts for degeneracy, if we want to exploit it   
    #nonetheless it sucks and should be addressed soon 
    def ndocc_irrep(self, C, eps):
        Eval_irreps = []
        Evals = []
        Eval_degen = []
        self.Orbs = []
        for ir, e_ir in enumerate(eps):
            degen = self.symtext.irreps[ir].d
            #degen = self.symtext.chartable.irrep_dims[self.salcs.irreps[ir]]
            if self.options.exploit_degen:
                factor = degen
            else:
                factor = 1
            orbs = C.blocks[ir]
            self.Orbs.append(ORB(ir, degen, Optional, Optional))
            
            eval_irreps = [ir for o in range(0, len(orbs))]
            eval_degen = [factor for o in range(0, len(orbs))]
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
        virt_list = list(self.sorted_irreps[i+1:])
        for o, orbs in enumerate(self.Orbs):
            ndocc_ir = docc_list.count(o)
            orbs.ndocc_ir = ndocc_ir
            orbs.nvirt_ir = virt_list.count(o)

#Class for repacking TEIs using symmetry arguments
#optionally exploiting degeneracy and repacking even further
class DPD():
    def __init__(self, orb_idx, symtext, salcs, so_orbitals, D, options):
        self.orb_idx = orb_idx
        self.symtext = symtext
        self.salcs = salcs
        self.fxn_list = so_orbitals.fxn_list
        self.irreplength = so_orbitals.irreplength
        self.D = D
        self.options = options
        self._pair_cache = {}

    def _get_all_pairs(self, ir_left, ir_right, sparse_salcs_full):
        key = ("all", ir_left, ir_right)
    
        if key not in self._pair_cache:
            self._pair_cache[key] = self._all_pairs(
                ir_left, ir_right, sparse_salcs_full
            )
    
        return self._pair_cache[key]
    
    
    def _get_same_partner_pairs(self, ir_left, ir_right, sparse_salcs_full, nfunx):
        key = ("same", ir_left, ir_right, nfunx)
    
        if key not in self._pair_cache:
            self._pair_cache[key] = self._same_partner_pairs(
                ir_left, ir_right, sparse_salcs_full, nfunx
            )
    
        return self._pair_cache[key]

    def dp_contains(self, irrep, a, b, *args):
        ctab = self.symtext.character_table
        a = ctab[a, :]
        b = ctab[b, :]
        chars = a * b
        for arg in args:
            chars *= ctab[arg, :]
        s = sum(chars * self.symtext.class_orders * ctab[irrep, :])
        n = s / sum(self.symtext.class_orders)
        if np.isclose(n, 0, atol = 1e-4):
            return False
        return True
    
    def dp_contains_tsir(self, a, b, *args):
        #ctab = self.symtext.chartable
        ctab = self.symtext.character_table
        a = ctab[a,:]
        b = ctab[b,:]
        chars = a * b
        for arg in args:
            chars *= ctab[arg, :]
        s = sum(chars * self.symtext.class_orders * ctab[0, :])
        n = s / sum(self.symtext.class_orders)
        if np.isclose(n, 0, atol = 1e-4):
            return False
        return True

    def sparse_ERI_transform(self, tensor, swap):
        sparse_salcs_full = []
        offset = 0
        for h, salc in enumerate(self.salcs.salc_sets):
            salcs = []
            for s_i, s in enumerate(salc):
                overall_s_idx = self.salcs.salcs_by_irrep[h][s_i]
                for I in range(len(s)):
                    if abs(s[I]) > 1e-5:
                        salcs.append(salc_element(I, s_i, s[I], self.salcs.salcs[overall_s_idx].i))
            sparse_salcs_full.append(salcs)
            offset += self.irreplength[h]
        nonzero_blocks = self.nonzero_tiles()
        self.nonzero_blocks = nonzero_blocks
        B = []
        self.braket = []
        for b, block in enumerate(nonzero_blocks):
            #if comparing to the old algorithm, you'll notice we skipped the "indices" function call
            #which nonintuitively repacks the ERI tensor into a 2D array
            #this step will be skipped for now, because repacking will occur at time of sparse ERI  transform
            if self.options.exploit_degen:
                print("Going to maximally exploit degeneracy!")
                bra_degen, ket_degen = self.lookup_braket_degen(block)

            else:
                bra_degen, ket_degen = 1, 1
                print("Even if we have degeneracy, not gonna take advantage of it")
                #self.braket is used within the JK build function, which allows the fock build JK contributions to be scaled by the degeneracy
                #IF the degeneracy is being exploited
                #self.braket = []
                for b, block in enumerate(self.nonzero_blocks):
                    self.braket.append(0)
                    twod_eri = self.sparse_nondegen_transform(block, b, sparse_salcs_full, tensor, swap)
                    B.append(twod_eri)
                    #incorporate code that performs the transformation and repacks into a 2D tensor, nothing further
                    #theoretically this could be a sparse or np.einsum transform (or some hybrid)

            #check for degeneracy in the bra or ket
            #condition 1, no degeracy in the bra or ket
            if self.nor((bra_degen > 1), (ket_degen > 1)):
                self.braket.append(0)
                twod_eri = self.sparse_nondegen_transform(block, b, sparse_salcs_full, tensor, swap)
                B.append(twod_eri)
                #the function call to transform the ERI should be uniform with the call above that doesn't exploit degeneracy

            #condition 2, either the bra or ket has degeneracy to be exploited in transform
            elif self.xor((bra_degen > 1), (ket_degen > 1)):
                if bra_degen > 1:
                    self.braket.append(1)
                    twod_eri = self.sparse_degen_bra(block, b, bra_degen, sparse_salcs_full, tensor, swap)
                    B.append(twod_eri)
                elif ket_degen > 1:
                    self.braket.append(2)
                    twod_eri = self.sparse_degen_ket(block, b, ket_degen, sparse_salcs_full, tensor, swap)
                    B.append(twod_eri)
            elif self.And((bra_degen > 1), (ket_degen > 1)): 
                self.braket.append(3)
                twod_eri = self.sparse_degen_braket(block, b, bra_degen, ket_degen, sparse_salcs_full, tensor, swap)
                B.append(twod_eri)
        return B

    def sparse_nondegen_transform(self, block, b, sparse_salcs_full, tensor, swap):
        ir1, ir2, ir3, ir4 = block[0], block[1], block[2], block[3]
        temp = np.zeros((len(self.orb_idx[ir1]) * len(self.orb_idx[ir2]), len(self.orb_idx[ir3]) * len(self.orb_idx[ir4])))
        ir1_fxns = len(self.orb_idx[ir1])
        ir3_fxns = len(self.orb_idx[ir3])
        #evaluate this true statement at the top of the list to prevent slowdown if it was at the bottom of 4 for-loops
        if swap:
            for i, salc_i in enumerate(sparse_salcs_full[ir1]):
                for j, salc_j in enumerate(sparse_salcs_full[ir2]):
                    for k, salc_k in enumerate(sparse_salcs_full[ir3]):
                        for l, salc_l in enumerate(sparse_salcs_full[ir4]):
                            ij = ir1_fxns * salc_i.s_idx + salc_j.s_idx
                            kl = ir3_fxns * salc_k.s_idx + salc_l.s_idx
                            #index ERI by I[i,k,j,l]
                            temp[ij,kl] += tensor[salc_i.i_idx, salc_k.i_idx, salc_j.i_idx, salc_l.i_idx] * salc_i.element * salc_j.element * salc_k.element * salc_l.element
        else:
            for i, salc_i in enumerate(sparse_salcs_full[ir1]):
                for j, salc_j in enumerate(sparse_salcs_full[ir2]):
                    for k, salc_k in enumerate(sparse_salcs_full[ir3]):
                        for l, salc_l in enumerate(sparse_salcs_full[ir4]):
                            ij = ir1_fxns * salc_i.s_idx + salc_j.s_idx
                            kl = ir3_fxns * salc_k.s_idx + salc_l.s_idx
                            #index ERI by I[i,j,k,l]
                            temp[ij,kl] += tensor[salc_i.i_idx, salc_j.i_idx, salc_k.i_idx, salc_l.i_idx] * salc_i.element * salc_j.element * salc_k.element * salc_l.element

        return temp


    def _same_partner_pairs(self, ir_left, ir_right, sparse_salcs_full, nfunx):
        """
        Build compressed pair list for degenerate irrep blocks.
    
        Returns tuples:
            (ao_left, ao_right, packed_index, coeff_product)
        """
        pairs = []
        right_fxns = self.irreplength[ir_right]
    
        for a in sparse_salcs_full[ir_left]:
            for b in sparse_salcs_full[ir_right]:
                if a.i != b.i:
                    continue
    
                # Preserve your original compressed-index logic
                factor_a = a.s_idx // nfunx
                factor_b = b.s_idx // nfunx
    
                packed = (
                    right_fxns * (a.s_idx - factor_a)
                    + b.s_idx
                    - (right_fxns * nfunx * factor_b)
                )
    
                pairs.append((a.i_idx, b.i_idx, packed, a.element * b.element))
    
        return pairs
    
    
    def _all_pairs(self, ir_left, ir_right, sparse_salcs_full):
        """
        Build ordinary uncompressed pair list.
    
        Returns tuples:
            (ao_left, ao_right, packed_index, coeff_product)
        """
        pairs = []
        right_fxns = len(self.orb_idx[ir_right])
    
        for a in sparse_salcs_full[ir_left]:
            for b in sparse_salcs_full[ir_right]:
                packed = right_fxns * a.s_idx + b.s_idx
                pairs.append((a.i_idx, b.i_idx, packed, a.element * b.element))
    
        return pairs
    
    
    def _contract_pair_lists(self, bra_pairs, ket_pairs, tensor, out_shape, swap):
        temp = np.zeros(out_shape)
    
        if swap:
            for p, q, pq, cpq in bra_pairs:
                for r, s, rs, crs in ket_pairs:
                    temp[pq, rs] += tensor[p, r, q, s] * cpq * crs
        else:
            for p, q, pq, cpq in bra_pairs:
                for r, s, rs, crs in ket_pairs:
                    temp[pq, rs] += tensor[p, q, r, s] * cpq * crs
    
        return temp

    def sparse_degen_bra(self, block, b, bra_degen, sparse_salcs_full, tensor, swap):
        ir1, ir2, ir3, ir4 = block
    
        nfunx = self.irreplength[ir1]
    
        #bra_pairs = self._same_partner_pairs(ir1, ir2, sparse_salcs_full, nfunx)
        #ket_pairs = self._all_pairs(ir3, ir4, sparse_salcs_full)
        bra_pairs = self._get_same_partner_pairs(ir1, ir2, sparse_salcs_full, nfunx)
        ket_pairs = self._get_all_pairs(ir3, ir4, sparse_salcs_full)
    
        out_shape = (
            self.irreplength[ir1] ** 2,
            len(self.orb_idx[ir3]) * len(self.orb_idx[ir4]),
        )
    
        return self._contract_pair_lists(
            bra_pairs, ket_pairs, tensor, out_shape, swap
        ) / bra_degen
    
    
    def sparse_degen_ket(self, block, b, ket_degen, sparse_salcs_full, tensor, swap):
        ir1, ir2, ir3, ir4 = block
    
        nfunx = self.irreplength[ir3]
    
        #bra_pairs = self._all_pairs(ir1, ir2, sparse_salcs_full)
        #ket_pairs = self._same_partner_pairs(ir3, ir4, sparse_salcs_full, nfunx)
        bra_pairs = self._get_all_pairs(ir1, ir2, sparse_salcs_full)
        ket_pairs = self._get_same_partner_pairs(ir3, ir4, sparse_salcs_full, nfunx)
    
        out_shape = (
            len(self.orb_idx[ir1]) * len(self.orb_idx[ir2]),
            self.irreplength[ir3] ** 2,
        )
    
        return self._contract_pair_lists(
            bra_pairs, ket_pairs, tensor, out_shape, swap
        ) / ket_degen
    
    
    def sparse_degen_braket(self, block, b, bra_degen, ket_degen, sparse_salcs_full, tensor, swap):
        ir1, ir2, ir3, ir4 = block
    
        #bra_pairs = self._same_partner_pairs(
        #    ir1, ir2, sparse_salcs_full, self.irreplength[ir1]
        #)
        #ket_pairs = self._same_partner_pairs(
        #    ir3, ir4, sparse_salcs_full, self.irreplength[ir3]
        #)
        bra_pairs = self._get_same_partner_pairs(
            ir1, ir2, sparse_salcs_full, self.irreplength[ir1]
        )
        ket_pairs = self._get_same_partner_pairs(
            ir3, ir4, sparse_salcs_full, self.irreplength[ir3]
        )
    
        out_shape = (
            self.irreplength[ir1] ** 2,
            self.irreplength[ir3] ** 2,
        )
    
        # Keeping your original scaling behavior: divide only by ket_degen.
        return self._contract_pair_lists(
            bra_pairs, ket_pairs, tensor, out_shape, swap
        ) / ket_degen

#    def sparse_degen_braket(self, block, b, bra_degen, ket_degen, sparse_salcs_full, tensor, swap):
#        ir1, ir2, ir3, ir4 = block[0], block[1], block[2], block[3]
#        #since ket degen, #self.irreplength[ir3] is the number of functions when exploiting degen
#        nfunx_ij = self.irreplength[ir1]
#        nfunx_kl = self.irreplength[ir3]
#
#        ir1_fxns = self.irreplength[ir1]
#        ir2_fxns = self.irreplength[ir2]
#        ir3_fxns = self.irreplength[ir3]
#        ir4_fxns = self.irreplength[ir4]
#        #setup 4 for-loops to create the nonzero ERI tensor blocks
#        temp = np.zeros((self.irreplength[ir1] **2,  self.irreplength[ir3] **2))
#        if swap:
#            for i, salc_i in enumerate(sparse_salcs_full[ir1]):
#                for j, salc_j in enumerate(sparse_salcs_full[ir2]):
#                    for k, salc_k in enumerate(sparse_salcs_full[ir3]):
#                        for l, salc_l in enumerate(sparse_salcs_full[ir4]):
#                            if (salc_i.i == salc_j.i) and (salc_k.i == salc_l.i):
#                                #checks the "column" of the irrep matrice the function belongs to, instead of indexing the i/j value
#                                #the quotient of the salc index and number of minimal # of functions 
#                                ijfactor = salc_i.s_idx // nfunx_ij
#                                ijfactor2 = salc_j.s_idx // nfunx_ij
#                                klfactor = salc_k.s_idx // nfunx_kl
#                                klfactor2 = salc_l.s_idx // nfunx_kl
#                                ij2 = ir2_fxns * (salc_i.s_idx - ijfactor)  + salc_j.s_idx - (ir2_fxns * nfunx_ij *ijfactor2) #- factor2 * self.irreplength[ir4] #+ salc_l.s_idx 
#                                kl2 = ir4_fxns * (salc_k.s_idx - klfactor)  + salc_l.s_idx - (ir4_fxns * nfunx_kl *klfactor2) #- factor2 * self.irreplength[ir4] #+ salc_l.s_idx 
#                                
#                                temp[ij2,kl2] += tensor[salc_i.i_idx, salc_k.i_idx, salc_j.i_idx, salc_l.i_idx] * salc_i.element * salc_j.element * salc_k.element * salc_l.element
#        else:
#            for i, salc_i in enumerate(sparse_salcs_full[ir1]):
#                for j, salc_j in enumerate(sparse_salcs_full[ir2]):
#                    for k, salc_k in enumerate(sparse_salcs_full[ir3]):
#                        for l, salc_l in enumerate(sparse_salcs_full[ir4]):
#                            if (salc_i.i == salc_j.i) and (salc_k.i == salc_l.i):
#                                #checks the "column" of the irrep matrice the function belongs to, instead of indexing the i/j value
#                                #the quotient of the salc index and number of minimal # of functions 
#                                ijfactor = salc_i.s_idx // nfunx_ij
#                                ijfactor2 = salc_j.s_idx // nfunx_ij
#                                klfactor = salc_k.s_idx // nfunx_kl
#                                klfactor2 = salc_l.s_idx // nfunx_kl
#                                ij2 = ir2_fxns * (salc_i.s_idx - ijfactor)  + salc_j.s_idx - (ir2_fxns * nfunx_ij *ijfactor2) #- factor2 * self.irreplength[ir4] #+ salc_l.s_idx 
#                                kl2 = ir4_fxns * (salc_k.s_idx - klfactor)  + salc_l.s_idx - (ir4_fxns * nfunx_kl *klfactor2) #- factor2 * self.irreplength[ir4] #+ salc_l.s_idx 
#                                
#                                temp[ij2,kl2] += tensor[salc_i.i_idx, salc_j.i_idx, salc_k.i_idx, salc_l.i_idx] * salc_i.element * salc_j.element * salc_k.element * salc_l.element
#        return temp /ket_degen
#
#    def sparse_degen_bra(self, block, b, bra_degen, sparse_salcs_full, tensor, swap):
#        ir1, ir2, ir3, ir4 = block[0], block[1], block[2], block[3]
#        #since ket degen, #self.irreplength[ir3] is the number of functions when exploiting degen
#        nfunx = self.irreplength[ir1]
#        #setup 4 for-loops to create the nonzero ERI tensor blocks
#        ir2_fxns = self.irreplength[ir2]
#        ir3_fxns = len(self.orb_idx[ir3])
#         
#        temp = np.zeros((self.irreplength[ir1] **2,  self.irreplength[ir3] **2))
#        if swap:
#            for i, salc_i in enumerate(sparse_salcs_full[ir1]):
#                for j, salc_j in enumerate(sparse_salcs_full[ir2]):
#                    for k, salc_k in enumerate(sparse_salcs_full[ir3]):
#                        for l, salc_l in enumerate(sparse_salcs_full[ir4]):
#                            if salc_i.i == salc_j.i:
#
#                                #checks the "column" of the irrep matrice the function belongs to, instead of indexing the i/j value
#                                factor = salc_i.s_idx // nfunx
#                                factor2 = salc_j.s_idx // nfunx
#                                
#                                ij2 = ir2_fxns * (salc_i.s_idx - factor)  + salc_j.s_idx - (ir2_fxns * nfunx *factor2) #- factor2 * self.irreplength[ir4] #+ salc_l.s_idx 
#                                kl = ir3_fxns * salc_k.s_idx + salc_l.s_idx
#                                #the quotient of the salc index and number of minimal # of functions 
#                                factor = salc_i.s_idx // nfunx
#                                
#                                temp[ij2,kl] += tensor[salc_i.i_idx, salc_k.i_idx, salc_j.i_idx, salc_l.i_idx] * salc_i.element * salc_j.element * salc_k.element * salc_l.element
#        else:
#            for i, salc_i in enumerate(sparse_salcs_full[ir1]):
#                for j, salc_j in enumerate(sparse_salcs_full[ir2]):
#                    for k, salc_k in enumerate(sparse_salcs_full[ir3]):
#                        for l, salc_l in enumerate(sparse_salcs_full[ir4]):
#                            if salc_i.i == salc_j.i:
#
#                                #checks the "column" of the irrep matrice the function belongs to, instead of indexing the i/j value
#                                factor = salc_i.s_idx // nfunx
#                                factor2 = salc_j.s_idx // nfunx
#                                
#                                
#                                ij2 = ir2_fxns * (salc_i.s_idx - factor)  + salc_j.s_idx - (ir2_fxns * nfunx *factor2) #- factor2 * self.irreplength[ir4] #+ salc_l.s_idx 
#                                kl = ir3_fxns * salc_k.s_idx + salc_l.s_idx
#                                #the quotient of the salc index and number of minimal # of functions 
#                                factor = salc_i.s_idx // nfunx
#                                
#                                temp[ij2,kl] += tensor[salc_i.i_idx, salc_j.i_idx, salc_k.i_idx, salc_l.i_idx] * salc_i.element * salc_j.element * salc_k.element * salc_l.element
#        return temp / bra_degen
#
#    def sparse_degen_ket(self, block, b, ket_degen, sparse_salcs_full, tensor, swap):
#        ir1, ir2, ir3, ir4 = block[0], block[1], block[2], block[3]
#        #since ket degen, #self.irreplength[ir3] is the number of functions when exploiting degen
#        nfunx = self.irreplength[ir3]
#        #setup 4 for-loops to create the nonzero ERI tensor blocks
#        ir2_fxns = len(self.orb_idx[ir2])
#        ir4_fxns = self.irreplength[ir4]
#        temp = np.zeros((self.irreplength[ir1] **2,  self.irreplength[ir3] **2))
#        if swap:
#            for i, salc_i in enumerate(sparse_salcs_full[ir1]):
#                for j, salc_j in enumerate(sparse_salcs_full[ir2]):
#                    for k, salc_k in enumerate(sparse_salcs_full[ir3]):
#                        for l, salc_l in enumerate(sparse_salcs_full[ir4]):
#                            if salc_k.i == salc_l.i:
#
#                                #checks the "column" of the irrep matrice the function belongs to, instead of indexing the i/j value
#                                factor = salc_k.s_idx // nfunx
#                                factor2 = salc_l.s_idx // nfunx
#                                ij = ir2_fxns * salc_i.s_idx + salc_j.s_idx
#                                kl2 = ir4_fxns * (salc_k.s_idx - factor)  + salc_l.s_idx - (ir4_fxns * nfunx *factor2) #- factor2 * self.irreplength[ir4] #+ salc_l.s_idx 
#                                #the quotient of the salc index and number of minimal # of functions 
#                                factor = salc_k.s_idx // nfunx
#                                #function_factor = nfunx **2 * ket_degen
#                                
#                                temp[ij,kl2] += tensor[salc_i.i_idx, salc_k.i_idx, salc_j.i_idx, salc_l.i_idx] * salc_i.element * salc_j.element * salc_k.element * salc_l.element
#        else:
#            for i, salc_i in enumerate(sparse_salcs_full[ir1]):
#                for j, salc_j in enumerate(sparse_salcs_full[ir2]):
#                    for k, salc_k in enumerate(sparse_salcs_full[ir3]):
#                        for l, salc_l in enumerate(sparse_salcs_full[ir4]):
#                            if salc_k.i == salc_l.i:
#
#                                #checks the "column" of the irrep matrice the function belongs to, instead of indexing the i/j value
#                                factor = salc_k.s_idx // nfunx
#                                factor2 = salc_l.s_idx // nfunx
#                                ij = ir2_fxns * salc_i.s_idx + salc_j.s_idx
#                                kl2 = ir4_fxns * (salc_k.s_idx - factor)  + salc_l.s_idx - (ir4_fxns * nfunx *factor2) #- factor2 * self.irreplength[ir4] #+ salc_l.s_idx 
#                                #the quotient of the salc index and number of minimal # of functions 
#                                factor = salc_k.s_idx // nfunx
#                                
#                                temp[ij,kl2] += tensor[salc_i.i_idx, salc_j.i_idx, salc_k.i_idx, salc_l.i_idx] * salc_i.element * salc_j.element * salc_k.element * salc_l.element
#        return temp / ket_degen

    def _find_nonzero_blocks(self):
        #combinations of i, j, k, and l that are nonzero for HF ERI
        #checks if direct product of ij (bra) and kl (ket) contain TSIR within themselves
        nonzero_blocks = []
        for i in range(0, len(self.orb_idx)):
            if len(self.orb_idx[i]) != 0:
                for j in range(0, len(self.orb_idx)):
                    if len(self.orb_idx[j]) != 0:
                        if self.dp_contains_tsir(i, j):
                            for k in range(0, len(self.orb_idx)):
                                if len(self.orb_idx[k]) != 0:
                                    for l in range(0, len(self.orb_idx)):
                                        if len(self.orb_idx[l]) != 0:
                                            if self.dp_contains_tsir(k, l):
                                                nonzero_blocks.append([i, j, k, l])
        return nonzero_blocks

    def nonzero_tiles(self):
        return self._find_nonzero_blocks()

####

#Everything below this point is redudant functionality. legacy code that can still be called as opposed to the sparse stuff above

    def lookup_hf_ERI(self, tensor):
        self.tensor = tensor
        before = time.time()
        self.nonzero_blocks = self._find_nonzero_blocks()
        self.i_symmetry = [block[0] for block in self.nonzero_blocks]
        self.k_symmetry = [block[2] for block in self.nonzero_blocks]
        self.twod_tensor = self.indices()
        if self.options.exploit_degen:
            self.lookup_degen()
        else:
            self.braket = []
            for b, block in enumerate(self.nonzero_blocks):
                self.braket.append(0)
    def compute_offsets(self):
        self.idk =[]
        for o in self.orb_idx:
            self.idk.append(len(o)) 
        self.offsets = []
        offset = 0
        for ir, orbs in enumerate(self.idk):
            self.offsets.append(offset)
            offset += orbs
       
    #same as "indices" function, but uses numpy.reshape so its hopefully faster? 
    def indices_v2(self):
        before = time.time()
        twod_tensor = []
        tot = 0
        self.compute_offsets()
        offset = sum(self.offsets[:0])
        for block in self.nonzero_blocks:
            #offsets are 0
            i,j,k,l = self.idk[block[0]], self.idk[block[1]], self.idk[block[2]], self.idk[block[3]]
            ir,jr,kr,lr = block[0], block[1], block[2], block[3]
            oi = sum(self.offsets[:ir]) 
            oj = sum(self.offsets[:jr]) 
            ok = sum(self.offsets[:kr]) 
            ol = sum(self.offsets[:lr]) 
            sliceit = self.tensor[oi:oi + i, oj:oj +j, ok:ok + k, ol:ol + l]
            pp = np.reshape(sliceit, (sliceit.shape[0]**2, sliceit.shape[3] **2))
            twod_tensor.append(pp)            
        now = time.time()
        print(f"NUMPY  Repacking Time for all blocks took {now - before:6.8f} seconds")
        return twod_tensor
  
    #makes twod tensor by gathering with np.ix_ and reshaping (vectorized, no python-level loop over AOs)
    def indices(self):
        before = time.time()
        twod_tensor = []
        for block in self.nonzero_blocks:
            i_idx, j_idx, k_idx, l_idx = self.orb_idx[block[0]], self.orb_idx[block[1]], self.orb_idx[block[2]], self.orb_idx[block[3]]
            sliceit = self.tensor[np.ix_(i_idx, j_idx, k_idx, l_idx)]
            twod_tensor.append(sliceit.reshape(len(i_idx) * len(j_idx), len(k_idx) * len(l_idx)))
        now = time.time()
        print(f"Repacking Time for all blocks took {now - before:6.8f} seconds")
        #test alternate implementation
        #self.alternate()
        return twod_tensor

    def transform(self, A, i,j,k,l):
        E = np.einsum("PQRS,Pp,Qq,Rr,Ss", A, i.T,j.T,k.T,l.T, optimize ='optimal')
        return E 
    
    def lookup_braket_degen(self, block):
        #set initial degeneracies to minimal values, loop thru possible irreps within the group and check if contained in the direct product
        #as of now, like the previous code, this code will only work if irreps in bra are the same and if the irreps in ket are the same...
        #will need a generalized code for when irreps within bra (2 or more) are not necessarily the same...
        bra_degen, ket_degen = 1, 1
        for ir, irrep in enumerate(self.symtext.irreps):
            if self.symtext.irreps[ir].d > 1:
                #assign variables to bra and ket irreps
                ir0, ir1 = block[0], block[1]
                ir2, ir3 = block[2], block[3]
                #is irrep contained in direct product of bra?
                if (ir == ir0) and (ir == ir1):
                    if self.symtext.irreps[ir].d >= bra_degen:
                        bra_degen = self.symtext.irreps[ir].d
                #is irrep contained in direct product of ket?
                if (ir == ir2) and (ir == ir3):
                    if self.symtext.irreps[ir].d >= ket_degen:
                        ket_degen = self.symtext.irreps[ir].d
        return bra_degen, ket_degen

    def lookup_degen(self):
        ctab = self.symtext.character_table
        self.Doned = []
        self.braket = []
        for d in self.D.blocks:
            self.Doned.append(np.reshape(d, d.shape[0] **2))
        for b, block in enumerate(self.nonzero_blocks):
            #print(f"The shape {self.twod_tensor[b].shape}")
            #print(f"The block {block}")
            bra_degen = 1
            bra_i = 0
            ket_degen = 1
            ket_i = 0
            #This is where Ih breaks the code. Because T1g x T1g contains up to Hg. 
            #Change this to look for degeneracy of direct product

            #loop over irreps to check for presence in direct product 
            for ir, irrep in enumerate(self.symtext.irreps):
                #print(f"ir {ir} {irrep}")
                #if ctab.irrep_dims[irrep] > 1:
                #if self.symtext.irreps[irrep].d > 1:
                if self.symtext.irreps[ir].d > 1:
                    #assign variables to bra and ket irreps
                    ir0, ir1 = block[0], block[1]
                    ir2, ir3 = block[2], block[3]
                    #is irrep contained in direct product of bra?
                    if (ir == ir0) and (ir == ir1):
                    #if self.dp_contains(ir, ir0, ir1):
                        if self.symtext.irreps[ir].d >= bra_degen:
                        #if ctab.irrep_dims[irrep] >= bra_degen:
                            #print(f"the irrep {irrep} for bra")
                            #bra_degen = ctab.irrep_dims[irrep]
                            bra_degen = self.symtext.irreps[ir].d
                            bra_i = ir
                    #is irrep contained in direct product of ket?
                    if (ir == ir2) and (ir == ir3):
                    #if self.dp_contains(ir, ir2, ir3):
                        #if ctab.irrep_dims[irrep] >= ket_degen:
                        if self.symtext.irreps[ir].d >= ket_degen:
                            #print(f"the irrep {irrep} for ket")
                            #ket_degen = ctab.irrep_dims[irrep]
                            ket_degen = self.symtext.irreps[ir].d
                            ket_i = ir
            #WORKING CODE, SAVE THIS!!
            #for ir, irrep in enumerate(ctab.irreps):
            #    if ctab.irrep_dims[irrep] > 1:
            #        ir0, ir1 = block[0], block[1]
            #        ir2, ir3 = block[2], block[3]
            #        if self.dp_contains(ir, ir0, ir1):
            #            if ctab.irrep_dims[irrep] >= bra_degen:
            #                print(f"the irrep {irrep} for bra")
            #                bra_degen = ctab.irrep_dims[irrep]
            #                bra_i = ir
            #        if self.dp_contains(ir, ir2, ir3):
            #            if ctab.irrep_dims[irrep] >= ket_degen:
            #                print(f"the irrep {irrep} for ket")
            #                ket_degen = ctab.irrep_dims[irrep]
            #                ket_i = ir
            if self.nor((bra_degen > 1), (ket_degen > 1)):
                self.braket.append(0)
                #print("no degeneracy!")
            elif self.xor((bra_degen > 1), (ket_degen > 1)):
                #print("one or the other!")
                if bra_degen > 1:
                    self.braket.append(1)
                    self.degen_bra(block, b, bra_degen)
                elif ket_degen > 1:
                    self.braket.append(2)
                    #print(f"The bra ket degens {bra_degen} {ket_degen}")
                    self.degen_ket(block, b, ket_degen)
            elif self.And((bra_degen > 1), (ket_degen > 1)): 
                self.braket.append(3)
                self.degen_braket(block, b, bra_degen, ket_degen)

    def degen_ket(self, block, b, degen):
        print("Inside degen ket")
        print(block)
        d_sym = block[3]
        oned_d_s = self.Doned[d_sym] 
        neri = self.ket_iter_salc_irrep_v2(block[3], b, oned_d_s, degen)
        #print(self.twod_tensor)
        print(f"The neri")
        print(neri)
        self.twod_tensor[b] = neri
    
    def degen_bra(self, block, b, degen):
        print("Inside degen bra")
        print(block)
        d_sym = block[3]
        oned_d_s = self.Doned[d_sym] 
        #neri = self.bra_iter_salc_irrep(block[0], b, oned_d_s, degen)
        neri = self.bra_iter_salc_irrep_v2(block[0], b, oned_d_s, degen)
        print("The neri")
        print(neri)
        self.twod_tensor[b] = neri
    
    def degen_braket(self, block, b, bra_degen, ket_degen):
        #print("Inside degen braket")
        #print(block)
        d_sym = block[3]
        oned_d_s = self.Doned[d_sym] 
        neri = self.iter_dual_salc_irrep_new(block[0], block[3], b, oned_d_s, bra_degen, ket_degen)
        print("The neri")
        print(neri)
        self.twod_tensor[b] = neri
    
    def ket_iter_salc_irrep_v2(self, index, b, density, degen):
        print("The function to be repacked")
        print(self.twod_tensor[b])
        nfxn = self.irreplength[index]
        #nfxn = len(self.fxn_list[index])
        limit = (nfxn **2) * degen
        new = np.zeros((self.twod_tensor[b].shape[0], nfxn ** 2))
        print(f"the number of functions {nfxn}, the limit is {limit}")
        if int(np.sqrt(self.twod_tensor[b].shape[1]) / degen) != nfxn:
            print(f"# of SALCs in the KET is {int(np.sqrt(self.twod_tensor[b].shape[1])/degen)}") 
            print(f"# of SALCs in the fxn_list is {nfxn}")
            print("""


            """)
            print(f"shape of ket {self.twod_tensor[b].shape[1]}")
            print(f"The degeneracy {degen}")
            #print(vars(self.salcs))
            print(self.salcs.salc_sets[1])
            print(self.salcs.salc_sets[6])
            raise ValueError("The shape of the old ERI Bra and the new ERI Bra is mismatched. This could be an issue with how partner functions were arranged in the new SALC sets/lists.")
        total = 0
        rs_new = 0
        for ri, r in enumerate(self.orb_idx[index]):
            for si, s in enumerate(self.orb_idx[index]):
                #salc_r = self.salcs.salc_list[r]
                #salc_s = self.salcs.salc_list[s]
                salc_r = self.salcs.salcs[r]
                salc_s = self.salcs.salcs[s]
                rs = ri * len(self.orb_idx[index]) + si
                #print(f"ind rs {rs} r_bfxn {salc_r.bfxn} s_bfxn {salc_s.bfxn} i [{salc_r.i}, {salc_s.i}]")
                if rs >= limit:
                    return new
                else:
                    if (salc_r.i == salc_s.i):
                        #print(f" {self.twod_tensor[b][:, rs]}")
                        offset_rs = salc_r.i * (nfxn ** 2)
                        #print(f"total {total} rs_new {rs_new} offset {offset_rs}")
                        new[:,rs_new - offset_rs] = self.twod_tensor[b][:,rs]
                        rs_new += 1
                        total += 1
    
    def bra_iter_salc_irrep_v2(self, index, b, density, degen):
        print("The tensor to be repacked (bra)")
        print(self.twod_tensor[b])
        nfxn = self.irreplength[index]
        #nfxn = len(self.fxn_list[index])
        limit = (nfxn **2) * degen
        new = np.zeros((nfxn **2, self.twod_tensor[b].shape[1]))

        if int(np.sqrt(self.twod_tensor[b].shape[0]) / degen) != nfxn:
            print(f"# of SALCs in the BRA is {int(np.sqrt(self.twod_tensor[b].shape[0])/degen)}") 
            print(f"# of SALCs in the fxn_list is {nfxn}") 
            raise ValueError("The shape of the old ERI Bra and the new ERI Bra is mismatched. This could be an issue with how partner functions were arranged in the new SALC sets/lists.")
        print(new.shape)
        print(f"the number of functions {nfxn}, the limit is {limit}")
        total = 0
        pq_new = 0
        print(self.orb_idx)
        print("orb idx ^^")
        for pi, p in enumerate(self.orb_idx[index]):
            for qi, q in enumerate(self.orb_idx[index]):
                #salc_p = self.salcs.salc_list[p]
                #salc_q = self.salcs.salc_list[q]
                salc_p = self.salcs.salcs[p]
                salc_q = self.salcs.salcs[q]
                pq = pi * len(self.orb_idx[index]) + qi
                #print(f"ind pq {pq} p_bfxn {salc_p.bfxn} q_bfxn {salc_q.bfxn} i [{salc_p.i}, {salc_q.i}]")
                if pq >= limit:
                    return new
                else:
                    if (salc_p.i == salc_q.i):
                        offset_pq = salc_p.i * (nfxn ** 2)
                        new[pq_new - offset_pq, :] = self.twod_tensor[b][pq,:]
                        pq_new += 1
                        total += 1
    def iter_dual_salc_irrep_new(self, index1, index2, b, density, bra_degen, ket_degen):
        print(f"Degen braket repacking")
        #bra_nfxn = len(self.fxn_list[index1])
        #ket_nfxn = len(self.fxn_list[index2])
        bra_nfxn = self.irreplength[index1]
        ket_nfxn = self.irreplength[index2]
        bra_index = self.twod_tensor[b].shape[0]
        bra_limit = (bra_nfxn **2) * bra_degen
        ket_limit = (ket_nfxn **2) * ket_degen
        new = np.zeros((self.Doned[index1].shape[0], density.shape[0] * ket_degen))
        total = 0
        pq_new = 0
        rs_new = 0
        for pi, p in enumerate(self.orb_idx[index1]):
            for qi, q in enumerate(self.orb_idx[index1]):
                pq = pi * len(self.orb_idx[index1]) + qi
                #salc_p = self.salcs.salc_list[p]
                #salc_q = self.salcs.salc_list[q]
                salc_p = self.salcs.salcs[p]
                salc_q = self.salcs.salcs[q]
                if salc_p.i == salc_q.i:
                    rs_new = 0
                    for ri, r in enumerate(self.orb_idx[index2]):
                        for si, s in enumerate(self.orb_idx[index2]):
                            rs = ri * len(self.orb_idx[index2]) + si
                            #salc_r = self.salcs.salc_list[r]
                            #salc_s = self.salcs.salc_list[s]
                            salc_r = self.salcs.salcs[r]
                            salc_s = self.salcs.salcs[s]
                            if salc_r.i == salc_s.i:
                                if pq >= bra_limit:
                                    #print("neri")
                                    #print(new)
                                    new = self.compress_neri(new, ket_nfxn, ket_degen)
                                    return new
                                else:
                                    #this is a valid density matrix block
                                    new[pq_new, rs_new] = self.twod_tensor[b][pq, rs]
                                    rs_new += 1
                                    total += 1 
                    pq_new += 1  

    def compress_neri(self, new, ket_nfxn, ket_degen):
        neri = np.zeros((new.shape[0], new.shape[1] // ket_degen))
        offset = ket_nfxn**2
        for x in range(0, ket_degen):
            start  = x * ket_nfxn ** 2
            finish = (x * ket_nfxn ** 2) + ket_nfxn **2 
            if x == 0:
                ref = new[:,start:finish]
            else:
                ref += new[:,start:finish] 
        return ref
    
    def xor(self, a, b):
        if a != b:
            return True
        return False

    def nor(self, a, b):
        if (a == 0) and (b == 0):
            return True
        return False

    def And(self, a, b):
        if (a == 1) and (b == 1):
            return True
        return False
    
    def OR(self, a, b):
        if (a == 1) and (b == 0):
            return True
        elif (a == 0) and (b == 1):
            return True
        else:
            return False
     
    #def partner_functions(self):
    #    new_salcs = []
    #    fxn_list = []
    #    new_salc_list = []
    #    for s, salc in enumerate(self.salcs.salc_sets):
    #        #degen = self.symtext.chartable.irrep_dims[self.salcs.irreps[s]]
    #        degen = self.symtext.irreps[s].d
    #        #degen = self.symtext.character_table.irrep_dims[self.salcs.irreps[s]]
    #        if salc is None:
    #            new_salcs.append(np.zeros((0, self.nbfxns)))
    #            fxn_list.append([])
    #        elif degen > 1:
    #            new = np.zeros((len(self.salcs.partner_function_sets_by_irrep[s]) * degen, self.nbfxns))
    #            for pf in range(0, degen):
    #                for ss, Set in enumerate(self.salcs.partner_function_sets_by_irrep[s]):
    #                    cpi = pf * len(self.salcs.partner_function_sets_by_irrep[s]) + ss
    #                    function = self.salcs.partner_function_sets_by_irrep[s][ss][pf]
    #                    new[cpi, :] = self.salcs.salc_list[function].coeffs
    #                    new_salc_list.append(self.salcs.salc_list[function]) 
    #            fxn_list.append([1 for i in range(0, (len(salc) // degen))])
    #            new_salcs.append(new)    
    #        else:
    #            for ss, Set in enumerate(self.salcs.partner_function_sets_by_irrep[s]):
    #                function = self.salcs.partner_function_sets_by_irrep[s][ss][0]
    #                new_salc_list.append(self.salcs.salc_list[function])
    #            new_salcs.append(salc)
    #            fxn_list.append([1 for i in range(0, len(salc))])
    #    self.salcs.salc_list = new_salc_list
    #    self.salcs.salc_sets = new_salcs
    #    self.fxn_list = fxn_list
    #    #print(self.fxn_list)

#    def alternate(self, A):
#        self.trial = []
#        for block in self.nonzero_blocks:
#            si = self.salcs.salc_sets[block[0]]
#            sj = self.salcs.salc_sets[block[1]]
#            sk = self.salcs.salc_sets[block[2]]
#            sl = self.salcs.salc_sets[block[3]]
#            eri_s = self.transform(A, si, sj, sk, sl)
#            bra_s = eri_s.shape[0] 
#            ket_s = eri_s.shape[2]
#            twod = np.reshape(eri_s, (bra_s **2, ket_s **2))
#            self.trial.append(twod)
#        #test against good method
#        self.twod_tensor = copy.deepcopy(self.trial)
#        if self.options.exploit_degen:
#            print("Going to maximally exploit degeneracy!")
#            self.lookup_degen()
#        else:
#            print("Even if we have degeneracy, not gonna take advantage of it")
#            self.braket = []
#            for b, block in enumerate(self.nonzero_blocks):
#                self.braket.append(0)

#    def alternate_swap(self, A):
#        self.trial_swap = []
#        for block in self.nonzero_blocks:
#            si = self.salcs.salc_sets[block[0]]
#            sj = self.salcs.salc_sets[block[1]]
#            sk = self.salcs.salc_sets[block[2]]
#            sl = self.salcs.salc_sets[block[3]]
#            #IF INPUT IS NOT SWAPPED, THIS WORKS
#            
#            eri_s = self.transform(A, si, sk, sj, sl)
#            bra_s = eri_s.shape[0] 
#            ket_s = eri_s.shape[3]
#            eri_s = np.swapaxes(eri_s, 1,2)
#            ##IF INPUT IS NOT SWAPPED, THIS WORKS V2
#            
#            #eri_s = self.transform(A, si, sj, sk, sl)
#            #bra_s = eri_s.shape[0] 
#            #ket_s = eri_s.shape[3]
#            ##eri_s = np.swapaxes(eri_s, 1,2)
#            #print(eri_s.shape) 
#            #self.trial_swap.append(pp)
#            
#            #TRY THIS THO
#             
#            #eri_s = self.transform(A, si, sj, sk, sl)
#            #bra_s = eri_s.shape[0] 
#            #ket_s = eri_s.shape[3]
#            ##eri_s = np.swapaxes(eri_s, 1,2)
#            ##print(eri_s.shape) 
#            #self.trial_swap.append(pp)
#        #test against good method
#        self.twod_tensor = copy.deepcopy(self.trial_swap)
#        if self.options.exploit_degen:
#            print("Going to maximally exploit degeneracy!")
#            self.lookup_degen()
#        else:
#            print("Even if we have degeneracy, not gonna take advantage of it")
#            self.braket = []
#            for b, block in enumerate(self.nonzero_blocks):
#                self.braket.append(0)
