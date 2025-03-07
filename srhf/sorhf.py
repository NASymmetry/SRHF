import numpy as np
import copy
import sys
import time
from copy import deepcopy
from scipy.linalg import fractional_matrix_power
import psi4

import molsym
from molsym.molecule import Molecule
from molsym.salcs.spherical_harmonics import SphericalHarmonics
from molsym.salcs.projection_op import ProjectionOp

from bdmats import BDMatrix
from diis_managerv2 import DIIS_Manager
from srhf_helper import SOrbitals
from srhf_helper import DPD
#from so_ints import SO_Ints 
#from mo_transform import MO_Trans

np.set_printoptions(precision=5, linewidth=200, suppress=True)


class SO_RHF():
    def __init__(self, mymol, basis_input, options):
        self.molecule = psi4.geometry(mymol)
        self.molecule.update_geometry()
        self.basis_input = basis_input
        self.options = options
    
    def run(self):
        print("Run the SO_RHF code!")
        self.ndocc = self.process_input() // 2
        #molecule stuff
        schema = self.qc()
        qcmol = Molecule.from_schema(schema)
        self.symtext = molsym.Symtext.from_molecule(qcmol)
        if self.options.subgroup:
            print(f"The symmetry code is running in subgroup {self.options.subgroup}")
            self.symtext = self.symtext.subgroup_symtext(self.options.subgroup)
        mol = self.symtext.mol
        self.molecule.set_geometry(psi4.core.Matrix.from_array(mol.coords))
        #basis set
        self.basis = psi4.core.BasisSet.build(self.molecule, 'BASIS', self.basis_input, puream = True)
        self.enuc = self.molecule.nuclear_repulsion_energy()
        #integrals
        ints = psi4.core.MintsHelper(self.basis)
          
        bset, nbas_vec = self.get_basis()
        #bset, nbas_vec = get_basis(molecule, basis)
        coords = SphericalHarmonics(self.symtext, bset)
        
        self.salcs = ProjectionOp(self.symtext, coords)
        #pass off salcs of the full point group as c1
        if self.options.fg_as_c1:
            self.fgtoc1_salcs(bset)

        self.nbfxns = psi4.core.BasisSet.nbf(self.basis)
        print(f"There are {self.nbfxns} AOs in this calculation")

        #self.so_orbitals = []
        
        #align salcs to maximally block-diagonalize our operators
        self.salcs.sort_to('blocks')
        
        #not sure why I've created this object. Perhaps recent MolSym updates have made this obsolete. 
        self.salcs.salc_sets = []
        fxn_list = []
        for ir, irrep in enumerate(self.symtext.irreps):
            if len(self.salcs.salcs_by_irrep[ir]) == 0:
                self.salcs.salc_sets.append(np.zeros((0, self.nbfxns)))
            else:
                self.salcs.salc_sets.append(np.row_stack([self.salcs[i].coeffs for i in self.salcs.salcs_by_irrep[ir]]))

        print(self.salcs.salc_sets)

        #Initialize the orbitals in the helper object. Take the initial guess. GWH and Core guesses are implemented.
        #Going to pass in a fake fxn_list argument for now, see if I can replace it later on...
        so_orbitals = SOrbitals(self.symtext, self.salcs, self.ndocc, self.options, self.nbfxns, fxn_list, self.basis)
        #so_orbitals.process_salcs()
        
        iter_type = "DIAG"
        D_i, docc_vector = self.build_D(so_orbitals)
        ERI = ints.ao_eri().np
        print("Repacking and symmetry blocking ERI")
        before = time.time()
        self.dpd = DPD(self.salcs.salcs_by_irrep, self.symtext, self.salcs, so_orbitals, D_i, self.options)
        #repacked_bigERI_swapped = self.dpd.trial_swap
        bigERI = self.aotoso_2(ERI)
        self.dpd.lookup_hf_ERI(bigERI)
        #twod pre J
        repacked_bigERI = self.dpd.twod_tensor
        #twod pre K
        ERI_swapped = np.swapaxes(bigERI, 1, 2)
        self.dpd.lookup_hf_ERI(ERI_swapped)
        repacked_bigERI_swapped = self.dpd.twod_tensor
        now = time.time()
        print(f"Finished repack {now - before:6.3f}")
        print("Starting SCF Iterations")
        print("Initiating DIIS Manager")
        diis_m = DIIS_Manager(self.symtext)
        start = time.time()
        E_i = 0
        #Begin SCF iterations
        for i in range(1, self.options.scf_max_iter + 1):
            before = time.time()
            F, ftime = self.build_fock_blocky_sym(so_orbitals.H, D_i, repacked_bigERI, repacked_bigERI_swapped)
            diis_m.do_diis(F, D_i, so_orbitals.S, so_orbitals.A, i)
            E_new = self.degen_rhf_energy(D_i, so_orbitals.H, F, so_orbitals) + self.enuc
            if self.options.diis:
                dRMS = diis_m.diis.dRMS 
            print(f"Iter {i:>3} SCF energy {E_new:>.10f} Delta(E) {E_new - E_i:^+.10f} RMS(D) {dRMS} {docc_vector} {iter_type} took {now - before:.7f} seconds")
            if (abs(E_new - E_i) < self.options.e_convergence) and (dRMS < self.options.d_convergence):
                print(stop)
                break
            E_i = E_new
            if np.any(diis_m.error > 0.1):
                F = diis_m.create_b()
                Fs = so_orbitals.A.transpose().dot(F.dot(so_orbitals.A))
                eps, Cs = Fs.eigh()
                C = so_orbitals.A.dot(Cs)
                so_orbitals.C = C
                D_new, docc_vector = self.build_D(so_orbitals)
                D_i = D_new
                iter_type = "DIIS"
            else:
                
                moF = F.einsum('ui,vj,uv', C, C, F)
                gn = -4 * moF.slicev2([":ndocc_ir", "ndocc_ir:"], so_orbitals.Orbs)
                occ_C = C.slicev2([":", ":ndocc_ir"], so_orbitals.Orbs)
                I = BDMatrix.full_to_bd(ERI, so_orbitals.irreplength)
                MO = I.einsum("PQRS,Pp,Qq,Rr,Ss", I, occ_C, C, C, C)

                eye_diag_occ = BDMatrix([np.diag(np.ones(so_orbitals.Orbs[0].ndocc_ir))])
                eye_diag_virt = BDMatrix([np.diag(np.ones(so_orbitals.Orbs[0].nvirt_ir))])

                Biajb = moF.einsum('ab,ij->iajb', moF.slice(["ndocc_ir:", "ndocc_ir:"], so_orbitals.Orbs), eye_diag_occ)
                Biajb -= moF.einsum('ij,ab->iajb', moF.slice([":ndocc_ir", ":ndocc_ir"], so_orbitals.Orbs), eye_diag_virt)
                Biajb += 4 * MO.slice([":", "ndocc_ir:", ":ndocc_ir", "ndocc_ir:"], so_orbitals.Orbs)
                Biajb -= MO.slice([":", "ndocc_ir:", ":ndocc_ir", "ndocc_ir:"], so_orbitals.Orbs).swapaxes(0, 2)
                Biajb -= MO.slice([":", ":ndocc_ir", "ndocc_ir:", "ndocc_ir:"], so_orbitals.Orbs).swapaxes(1, 2)
                Biajb *= 4

                oXv_idx = []                
                ovov_idx = []                
                for o, orb in enumerate(so_orbitals.Orbs):
                    oXv_idx.append([orb.ndocc_ir * orb.nvirt_ir, -1])
                    ovov_idx.append([orb.ndocc_ir, orb.nvirt_ir, orb.ndocc_ir, orb.nvirt_ir])


                # Invert B, (o^3 v^3); solves Newton equations H*x = B
                Binv = BDMatrix.inv(Biajb.reshape(oXv_idx)).reshape(ovov_idx)

                x = Binv.einsum('iajb,ia->jb', Binv, gn)
                U = []
                for h, Cirrep in enumerate(C.blocks):
                    if len(Cirrep) == 0:
                        U.append(np.array([])) 
                    else:
                        U.append(np.zeros(Cirrep.shape))
                U = BDMatrix(U)
                
                U.slice([":ndocc_ir", "ndocc_ir:"], so_orbitals.Orbs, x)
                U.slice(["ndocc_ir:", ":ndocc_ir"], so_orbitals.Orbs, -1*x.transpose())
                U += 0.5 * U.dot(U)
                for ui, u in enumerate(U.blocks):
                    if len(u) == 0:
                        pass
                    else:
                        U.blocks[ui][np.diag_indices_from(so_orbitals.A.blocks[ui])] += 1
                U, r = (U.transpose()).qr()
                C = C.dot(U)
                iter_type = 'SOSCF'
                so_orbitals.C = C
                D_new, docc_vector = self.build_D(so_orbitals)
                D_i = D_new

    def create_slices(self, slice_args, Orbs):
        #for now, Orbs only supports ndocc_irrep objects
        trials = []
        for i, s_arg in enumerate(slice_args):
            try:
                test = []
                for x in s_arg:
                    if x is not None:
                        test.append(getattr(Orbs[0], x))
                    else:
                        test.append(x)
                test_s = slice(*test)
                trials.append(test_s)
            except:
                raise ValueError(f"It is possible that of the slice arguments within {s_arg} is not a valid attribute of the Orbs object or is not None")
        return tuple(trials)

    def degen_rhf_energy(self, D, H, F, SOrbs):
        """
        Calculate HF energy
        """
        if isinstance(D, BDMatrix):
            E = 0
            for h, d in enumerate(D.blocks):
                if len(D.blocks[h]) == 0:
                    continue
                else:
                    if self.options.exploit_degen:
                        degen = SOrbs.Orbs[h].degen
                        e = degen * sum(sum(np.multiply(D.blocks[h],(H.blocks[h]+F.blocks[h]))))
                        E += e #*sum(sum(np.multiply(D.blocks[h],(H.blocks[h]+F.blocks[h]))))
                    else:
                        e = sum(sum(np.multiply(D.blocks[h],(H.blocks[h]+F.blocks[h]))))
                        E += e #*sum(sum(np.multiply(D.blocks[h],(H.blocks[h]+F.blocks[h]))))

        else:
            E = sum(sum(np.multiply(D,(H+F))))
        return E

    def build_fock_blocky_sym(self, H, Dp, repacked_bigERI, repacked_bigERI_swapped):
        start = time.time()
        #broadcast h d and f to oned. fock should really be the only one packed and unpacked each time, could be fed into this function
        before = time.time()
        oned_h, oned_f, oned_d = self.build_d_h_f(Dp, H)
        now = time.time() 
        fstart = time.time()
        jktime_total = 0
        for b, block in enumerate(self.dpd.nonzero_blocks):
            #f_sym and d_sym are the irrep of mu and sigma, respectively
            f_sym, d_sym = block[0], block[3]
            #index h and d of the proper symmetry to form fock and contract with eri, respectively
            oned_h_s, oned_d_s = oned_h[f_sym], oned_d[d_sym]
            #form j and k 
            jkstart = time.time()
            j, k = self.jk(repacked_bigERI[b], repacked_bigERI_swapped[b], oned_d_s, self.dpd.braket[b], block)
            jkfinish = time.time()
            jktime_total += (jkfinish - jkstart)
            #print(f"jk time took {jkfinish - jkstart:6.5f} seconds for block {self.tensor_sym_string(block, symtext)} {repacked_bigERI[b].shape}")
            #construct fock
            oned_f[f_sym] += 2 * j - k
        ffinish = time.time()
        #print(f"Fock loop time took {ffinish - fstart:6.5f} seconds, {jktime_total:6.5f} seconds for jk")
        F = BDMatrix(self.repack_fock(oned_f, oned_h))
        finish = time.time()
        #print(f"Total fock build time took {finish - start:6.8f} seconds")
        return F, finish - start
    
    def repack_fock(self, oned_f, oned_h):
        F = []
        for z, hs in enumerate(oned_h):
            oned_f[z] += hs
            if len(hs) == 0:
                F.append(np.array([]))
            else:
                F.append(self.oned_twod(oned_f[z]))
        return F
    
    def jk(self, ERI,ERI_swap, d, braket, block):
        if braket == 2:
            degen = self.symtext.irreps[block[3]].d
            #degen = self.symtext.chartable.irrep_dims[self.salcs.irreps[block[3]]]
            j = degen * np.einsum('pr,r->p', ERI, d)
            k = degen * np.einsum('pr,r->p', ERI_swap, d)
        else:
            j = np.einsum('pr,r->p', ERI, d)
            k = np.einsum('pr,r->p', ERI_swap, d)
        return j, k
    
    def build_d_h_f(self, Dp, H):
        oned_h = [] 
        oned_f = [] 
        oned_d = [] 
        for hi, h in enumerate(H.blocks):
            if len(h) == 0:
                oned_h.append(np.array([]))
                oned_f.append(np.array([]))
            else:
                oned_h.append(self.twod_oned(h))
                #f = np.zeros((oned_h[hi].shape))
                oned_f.append(np.zeros((oned_h[hi].shape)))
        for d in Dp.blocks:
            if len(d) == 0:
                oned_d.append(np.array([]))
            else:
                oned_d.append(self.twod_oned(d))
        return oned_h, oned_f, oned_d
    def twod_oned(self, mat):
        if len(mat) == 0:
            pass
        else:
            oned_mat = np.zeros((mat.shape[0] * mat.shape[1]))
            for i in range(0, mat.shape[0]):  
                for j in range(0, mat.shape[1]):  
                    ij = mat.shape[1] * i + j
                    oned_mat[ij] = mat[i,j]
            return oned_mat
    
    def oned_twod(self, mat):
        root = int(np.sqrt(mat.shape[0]))
        twod_mat = np.zeros((root, root))
        #lned_mat = np.zeros((mat.shape[0],  mat.shape[1]))
        for i in range(0, twod_mat.shape[0]):  
            for j in range(0, twod_mat.shape[1]):  
                ij = root * i + j
                twod_mat[i,j] = mat[ij]
        return twod_mat
        
    def build_D(self, SOrbs):
        docc_vector = []
        blocks = []
        for h, Cirrep in enumerate(SOrbs.C.blocks):
            if self.options.docc is not None:
                nir = self.options.docc[h]
            else:
                nir = SOrbs.Orbs[h].ndocc_ir
            docc_vector.append(nir)
            if len(Cirrep) == 0:
                blocks.append(np.array([])) 
            elif nir == 0:
                blocks.append(np.zeros(Cirrep.shape))
            else:
                blocks.append(np.einsum('pi,qi->pq', Cirrep[:,:nir], Cirrep[:,:nir]))
        return BDMatrix(blocks), docc_vector  

    def fgtoc1_salcs(self, bset):
        self.salcs = ProjectionOp(self.symtext, coords)
        self.salcs_fg = copy.deepcopy(self.salcs)
        self.symtext = self.symtext.subgroup_symtext("C1")
        coords = SphericalHarmonics(self.symtext, bset)
        self.salcs = ProjectionOp(self.symtext, coords)
        
        for s, salc in enumerate(self.salcs_fg):
            if s < (len(self.salcs.salcs)):
                self.salcs.salcs[s].coeffs = self.salcs_fg.salcs[s].coeffs
    
    def process_input(self):
        electrons = 0
        for atom in range(0, self.molecule.natom()):
            electrons += self.molecule.ftrue_atomic_number(atom)
        electrons -= self.molecule.molecular_charge()
        #Need something for processing charge as well... do that when you want to test a molecule like that
        return electrons

    def get_basis(self):
        nbas_vec = []
        molecule_basis = []
        counter = 0
        for x in range(0, self.molecule.natom()):
            atom_basis = []
            for y in range(0, self.basis.nshell_on_center(x)):
                atom_basis.append(self.basis.shell(y+counter).am)
                print(self.basis.shell(y+counter).am)
            counter += self.basis.nshell_on_center(x)
            L = 0
            for l in atom_basis:
                L += 2*l + 1
            molecule_basis.append(atom_basis)
            nbas_vec.append(L)
        return molecule_basis, nbas_vec
    
    def qc(self):
        qc_obj = {
            "symbols": [self.molecule.symbol(x) for x in range(0, self.molecule.natom())] ,
            "geometry": self.molecule.geometry(),
        }
        return qc_obj
    def aotoso_2(self, ERI):
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
        temp1 = np.einsum("PQRS,Pp->pQRS", ERI, s, optimize='optimal')
        temp2 = np.einsum("pQRS,Qq->pqRS", temp1, s, optimize='optimal')
        temp3 = np.einsum("pqRS,Rr->pqrS", temp2, s, optimize='optimal')
        E = np.einsum("pqrS,Ss->pqrs", temp3, s, optimize='optimal')
        return E
