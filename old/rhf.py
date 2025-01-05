import psi4
import numpy as np
#from input import Settings
import copy
from copy import deepcopy
import molsym
from molsym.molecule import Molecule
#from molsym.salcs.RSH_SALCs import Project
from molsym.salcs.spherical_harmonics import SphericalHarmonics
from molsym.salcs.projection_op import ProjectionOp
from bdmats import BDMatrix
#from dpd import DPD
from diis_managerv2 import DIIS_Manager
#from sorbitals import SOrbitals
import sys
import time
#from options import Options
#from mendeleev.fetch import fetch_table
#from mendeleev import element
from srhf_helper import SOrbitals
from srhf_helper import DPD
from so_ints import SO_Ints 
from scipy.linalg import fractional_matrix_power
from mo_transform import MO_Trans

np.set_printoptions(threshold=sys.maxsize, linewidth=12000, precision=10)

class SRHF():
    def __init__(self, mymol, basis_input, options):
        print("Nothing to init!!")
        self.molecule = psi4.geometry(mymol)
        self.molecule.update_geometry()
        self.basis_input = basis_input
        self.options = options
    def process_input(self):
        electrons = 0
        for atom in range(0, self.molecule.natom()):
            electrons += self.molecule.ftrue_atomic_number(atom)
        print(f"Total electrons, before charge process {electrons}")
        electrons -= self.molecule.molecular_charge()
        print(f"Total electrons, afta {electrons}")
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
    def handle_salcs(self):
        if self.options.subgroup:
            print(f"The symmetry code is running in subgroup {self.options.subgroup}")
            self.symtext = self.symtext.subgroup_symtext(self.options.subgroup) 
    def run(self):
        print("Run the RHF code!")
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
        self.nbfxns = psi4.core.BasisSet.nbf(self.basis)
        print(f"There are {self.nbfxns} AOs in this calculation")
        #orbital_idxs = self.salcs.salcs_by_irrep
        self.so_orbitals = []
        print("salc list now boi")
        #print(vars(self.salcs))
        self.salcs.sort_to('blocks')
        #print("salcs lists (self.salcs.salcs)")
        #print(self.salcs.salcs)
        self.salcs.salc_sets = []
        for salc in self.salcs.salcs:
            print(salc)
        print("salcs by irrep (self.orb_idxs)")
        print(self.salcs.salcs_by_irrep)
        fxn_list = []
        for ir, irrep in enumerate(self.symtext.irreps):
            if len(self.salcs.salcs_by_irrep[ir]) == 0:
                #ir_salcs = None
                ir_salcs = np.zeros((0, self.nbfxns))
                fxn_list.append([])
                #needs to be array([], shape=(0, 9), dtype=float64), array([], shape=(0, 9), dtype=float64), array([], shape=(0, 9), dtype=float64)
                self.salcs.salc_sets.append(ir_salcs)
            else:
                ir_salcs = [self.salcs[i].coeffs for i in self.salcs.salcs_by_irrep[ir]]
                ir_salcs = np.row_stack(ir_salcs)
                self.salcs.salc_sets.append(ir_salcs)
                fxn_list.append([1 for i in range(0, (len(ir_salcs) // self.symtext.irreps[ir].d))])
        #FXN LIST [[1, 1, 1], [], [], [], [1, 1]]
        print(fxn_list) 
        #print(bibba)
        
        #print(vars(self.salcs))
        #print(self.salcs.salc_sets)
        #print(self.salcs.salc_list)
        #print(len(self.salcs.salc_list))
        #print(usse) 
        #print(self.salcs)
        #print(self.salcs.sort_to('blocks'))
        #print(self.salcs)
        #print(self.salcs.salcs_by_irrep)
        ##print(for salc in self.salcs
        #self.salcs.salc_sets = []
        #for ir, irrep in enumerate(self.symtext.irreps):
        #    if len(self.salcs.salcs_by_irrep[ir]) == 0:
        #        ir_salcs = None
        #        self.salcs.salc_sets.append(ir_salcs)
        #    else:
        #        ir_salcs = [self.salcs[i].coeffs for i in self.salcs.salcs_by_irrep[ir]]
        #        ir_salcs = np.row_stack(ir_salcs)
        #        self.salcs.salc_sets.append(ir_salcs)
        #print(self.salcs.salc_sets)
        #print("salc list")
        #print(self.salcs.salc_list) 
        for salc in self.salcs.salc_sets:
            print(salc)
            if salc is None:
                self.so_orbitals.append(0)
            else:
                self.so_orbitals.append(salc.shape[0])
        print(f"The number of salcs per irrep: {self.so_orbitals}") 
        #process salcs
        #so_orbitals = SOrbitals(self.symtext, self.salcs, self.ndocc, self.options, self.nbfxns)
        so_orbitals = SOrbitals(self.symtext, self.salcs, self.ndocc, self.options, self.nbfxns, fxn_list)
        so_orbitals.process_salcs()
        #so_orbitals.ndocc_irrep(None, None, None)

        #compute integrals
        S = ints.ao_overlap().np
        T = ints.ao_kinetic().np
        V = ints.ao_potential().np
        ERI = ints.ao_eri().np
        
         
        #symmetry adapt
        #for s in S.blocks:
        #    print(s)
        if self.options.intsdpd:
            S = self.ao_to_so(S, so_orbitals) 
            T = self.ao_to_so(T, so_orbitals) 
            V = self.ao_to_so(V, so_orbitals)
        
        else:
            symmetry_adapt = SO_Ints(self.symtext, so_orbitals, self.salcs, self.options)
            S = symmetry_adapt.rank2_transform(S)
            T = symmetry_adapt.rank2_transform(T)
            V = symmetry_adapt.rank2_transform(V)
        #Do other stuff before TEI transform
        if self.options.guess == "core":
            #C, A, eps = self.rhf_core_guess(S, T, V)
            C, A, eps = self.rhf_core_guessv2(S, T, V)
        elif self.options.guess == "gwh":
            C, A, eps = self.gwh_guess(S, T, V)
        else:
            raise Exception("NEED TO IMPLEMENT SAD GUESS")
        #print(f"The core guess orbitals used for SO-convergence")
        #print(C.blocks)
        #Build initial Density from DOCC or initial guess
        so_orbitals.ndocc_irrep(C, eps)
        D_i, docc_vector = self.build_D(C, eps, so_orbitals)
        print(f"The docc_vector {docc_vector}")
        #build core hamiltonian
        H = T + V
      
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
        self.fock_timings = []
        for i in range(1, self.options.scf_max_iter + 1):
            before = time.time()
            F, ftime = self.build_fock_blocky_sym(H, D_i, repacked_bigERI, repacked_bigERI_swapped)
            if self.options.diis:
                diis_m.do_diis(F, D_i, S, A, i)
                F = diis_m.create_b() 
            E_new = self.degen_rhf_energy(D_i, H, F, so_orbitals) + self.enuc
            Fs = A.transpose().dot(F.dot(A))
            eps, Cs = Fs.eigh()
            C = A.dot(Cs)
            D_new, docc_vector = self.build_D(C, eps, so_orbitals)
             
            delta_e = np.format_float_scientific(np.absolute(E_new) - np.absolute(E_i), unique = False, precision=3)
            delta_d = D_new - D_i
            d_rms = np.format_float_scientific(delta_d.frob_norm(), precision = 3)
            D_i = D_new
            E_i = E_new
            
            now = time.time()

            print(f"Iter {i:>3} SCF energy {E_new:>.10f} Delta(E) {delta_e} RMS(D) {d_rms} {docc_vector} took {now - before:.7f} seconds")
            #if (float(delta_e)) < Settings["e_converge"] and float(d_rms) < Settings["d_converge"]:
            if (float(delta_e)) < self.options.e_convergence and float(d_rms) < self.options.d_convergence: 
                finished = time.time() 
                print(f"SCF Cycles Converged In {i} Iterations")
                print(f"Final RHF energy {E_new} which took {finished - start:6.3f} seconds")
                if self.options.compare_psi:
                    self.compute_psi(E_new)
                break
                #print(self.symtext.chartable.irreps)
                #self.compute_psi(E_new)
            if i == self.options.scf_max_iter:
                print(f"SCF Cycles Did Not Converge In {i} Iterations, Donate To The Developers' Patreon")
                self.avg_ftime = np.average(self.fock_timings)
                if self.options.compare_psi:
                    self.compute_psi(E_new)


            #Fock_list, DIIS_error, dRMS = DIIS(F, D_i, S, A, Fock_list, DIIS_error)
            #diis_m.do_diis(F, D_i, S, A, i)
            #s_type = "Diag"
            #F = diis_m.create_b()
            #Set this to greater than 0 to initiate a second order step, not ready yet tho
            #if np.any(diis_m.diis.error_hist[-1] > 0):
            #    s_type = "DIIS"
            #    ##updated fock from diis
            #    F = diis_m.create_b()
            #else:
            #    #This doesn't work at the moment, don't try it Steef
            #    D_i, docc_vector = self.second_order(F, C, A, docc_vector, so_orbitals, ERI)
            #  
            #E_new = self.degen_rhf_energy(D_i, H, F, so_orbitals) + self.enuc

            #D_i, docc_vector = self.diag(F, A, so_orbitals)
            #
            ##if i > 1:
            ##    self.second_order(F, Cs, docc_vector)
            ##D_i = D_new
            #delta_e = np.absolute(E_new) - np.absolute(E_i)
            #E_i = E_new 
            #now = time.time()
            ##print(f"Iter {i:>3} SCF energy {E_new:>.10f} Delta(E) {delta_e:+.3e} dRMS {diis_m.diis.dRMS:>.3e} {s_type}, occupation {docc_vector}, took {now - before:.7f} seconds")
            #print(f"Iter {i:>3} SCF energy {E_new:>.10f} Delta(E) {delta_e:+.3e}, occupation {docc_vector}, took {now - before:.7f} seconds")
            ##if (float(delta_e)) < Settings["e_converge"] and float(d_rms) < Settings["d_converge"]:
            ##if (abs(float(delta_e)) < self.options.e_convergence) and (float(diis_m.diis.dRMS) < self.options.d_convergence):
            #if (abs(float(delta_e)) < self.options.e_convergence):
            #    if self.options.compare_psi:
            #        self.compute_psi(E_new)
            #    break
            #if i > self.options.scf_max_iter + 1:
            #    print("Bruh")

    def diag(self, F, A, so_orbitals):
        Fs = A.transpose().dot(F.dot(A))
        eps, Cs = Fs.eigh()
        C = A.dot(Cs)
        #D_new, docc_vector = self.build_D(C, eps, so_orbitals)
        return self.build_D(C, eps, so_orbitals)

    def second_order(self, F, Cs, A, docc_vector, so_orbitals, ERI):
        print("SECOND ORDER UPDATE")
        print(Cs)
        print("The F")
        print(F)
        #F = A.dot(F).dot(A)
        moF = Cs.transpose().dot(F).dot(Cs)
        print("Fock in MO basis")
        print(moF)
        gn = []
        #for i, f in enumerate(moF.blocks):
        #    gn_i = -4 * f[:docc_vector[i], docc_vector[i]:]
        #    gn.append(gn_i)
        #gn = BDMatrix(gn)
        #npC = []
        #for i, c in enumerate(Cs.blocks):
        #    npc[:] = c
        #    print("c and npc")
        #    print(npc)
        #    print(c)
        mot = MO_Trans(docc_vector, so_orbitals, self.symtext)
        mot.mo_eri(ERI, Cs)
    def degen_rhf_energy(self, D, H, F, SOrbs):
        """
        Calculate HF energy
        """
        if isinstance(D, BDMatrix):
            #print("are we bd?")
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
    def jk(self, ERI,ERI_swap, d, braket, block):
        if braket == 2:
            degen = self.symtext.irreps[block[3]].d
            #degen = self.symtext.chartable.irrep_dims[self.salcs.irreps[block[3]]]
            #print(f"The degen is {degen}") 
            j = degen * np.einsum('pr,r->p', ERI, d)
            k = degen * np.einsum('pr,r->p', ERI_swap, d)
        else:
            j = np.einsum('pr,r->p', ERI, d)
            k = np.einsum('pr,r->p', ERI_swap, d)
        return j, k
    
    def tensor_sym_string(self, block, symtext):
        #irreps = symtext.chartable.irreps
        irreps = symtext.irreps
        bra = irreps[block[0]] + ' ' + irreps[block[1]]
        ket = irreps[block[2]] + ' ' + irreps[block[3]]
        braket = '< ' + bra + ' | ' + ket + ' >'
        return braket 

    #part of this function needs to be moved to the bdmatrix class
    #add core hamiltonian to fock 
    def repack_fock(self, oned_f, oned_h):
        F = []
        for z, hs in enumerate(oned_h):
            oned_f[z] += hs
            if len(hs) == 0:
                F.append(np.array([]))
            else:
                F.append(self.oned_twod(oned_f[z]))
        return F
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

    def get_norm(self, i):
        return 1/np.sqrt(i)

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
                for fi in range(0, len(s)):
                    F.blocks[i][fi, fi] = H.blocks[i][fi, fi]
                    for fj in range(0, fi):
                        dummy = 0.5 * gwh_k * (H.blocks[i][fi, fi] + H.blocks[i][fj, fj]) * s[fi, fj]
                        F.blocks[i][fi, fj] = dummy
                        F.blocks[i][fj, fi] = dummy
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
        #print("THE RIGHT C")
        #print(C)
        return C, A, En

    def normalize(self, s):
        news = copy.deepcopy(s)
        over = np.zeros(s.shape)
        normlist = []
        for i in range(len(s)):
            norm1 = self.get_norm(s[i, i])
            normlist.append(norm1)
            for j in range(len(s)):
                norm2 = self.get_norm(s[j,j])
                over[i,j] = s[i,j] * norm1 * norm2
        eigval, U = np.linalg.eigh(over)
        Us = deepcopy(U)
        for i in range(len(eigval)):
            Us[:,i] = U[:,i] * 1.0/np.sqrt(eigval[i])
        for i in range(len(eigval)):
            Us[i,:] = Us[i,:] * normlist[i]
        anti = np.dot(Us, U.T)
        return anti
    
    def build_D(self, C, eps, SOrbs):
        #print("Inside Build D")
        docc_vector = []
        blocks = []
        for h, Cirrep in enumerate(C.blocks):
            if self.options.docc is not None:
                nir = self.options.docc[h]
            else:
                nir = SOrbs.Orbs[h].ndocc_ir
            docc_vector.append(nir)
            #print(f"{h} {nir}")
            if len(Cirrep) == 0:
                blocks.append(np.array([])) 
            elif nir == 0:
                blocks.append(np.zeros(Cirrep.shape))
            else:
                blocks.append(np.einsum('pi,qi->pq', Cirrep[:,:nir], Cirrep[:,:nir]))
        return BDMatrix(blocks), docc_vector  
    def compute_psi(self, E_new):
        #psi4.set_options({'basis': Settings["basis"],
        psi4.set_options({'basis': self.basis_input,
                              'scf_type': 'pk',
                              'mp2_type': 'conv',
                              'e_convergence': 1e-12,
                              'd_convergence': 1e-12,
                             'reference': 'rhf',
                             'guess' : 'core',
                             "puream": True,
                             "print" : 7,
                             "freeze_core": False})
        #ndocc = Settings["nalpha"]
        #nbfxns = psi4.core.BasisSet.nbf(basis)
        pe, wfn = psi4.energy('scf', return_wfn=True)
         
        print(f"Difference between us and PSI4: {abs(E_new-pe)}")
        #print(wfn.Fa_subset("MO").nph)
        #print(wfn.Fa_subset("AO").nph)


    def ao_to_so(self, A, so_orbitals):
        """
        AO->SO transformation for one electron integrals
        """
        B = []
        for i, salc in enumerate(self.salcs.salc_sets):
            #print(salc)
            #print(salc[: so_orbitals.irreplength[i]])
            temp1 = np.einsum('uv,ui->iv', A, salc[:so_orbitals.irreplength[i]].T, optimize ='optimal') 
            temp = np.einsum('iv,vj->ij', temp1, salc[:so_orbitals.irreplength[i]].T, optimize ='optimal')
            B.append(temp)
        return BDMatrix(B)
    
    def aotoso_2(self, ERI):
        """
        AO->SO transformation for two electron integrals
        """
        first = True
        for i, salc in enumerate(self.salcs.salc_sets):
            print(f"{i} {salc}")
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
