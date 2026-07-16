import numpy as np
import copy
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from scipy.linalg import fractional_matrix_power, block_diag
import psi4

import molsym
from molsym.molecule import Molecule
from molsym.salcs.spherical_harmonics import SphericalHarmonics
from molsym.salcs.projection_op import ProjectionOp

from srhf.bdmats import BDMatrix
from srhf.diis_managerv2 import DIIS_Manager
from srhf.srhf_helper import SOrbitals
from srhf.srhf_helper import DPD
#from so_ints import SO_Ints 
#from mo_transform import MO_Trans

np.set_printoptions(precision=5, linewidth=200, suppress=True)


@dataclass
class TiledMOData:
    """
    Combined-and-tiled AO->MO integral/Fock data: every irrep's
    representative-partner block duplicated degen_h times (1 if
    exploit_degen=False for that irrep -- so_orbitals.irreplength[h] is
    already the full, uncompressed size in that case, and tiling it
    further would over-index; degen_h = symtext.irreps[h].d only when
    exploit_degen=True actually compressed this irrep down to one
    representative partner). See SO_RHF._build_tiled_mo_data's docstring
    for the full rationale and who consumes this.
    """
    I_tiled: np.ndarray
    MO_tiled: np.ndarray
    C_tiled: np.ndarray
    occ_C_tiled: np.ndarray
    moF_tiled: np.ndarray
    tile_start: dict
    occ_tile_start: dict
    populated: list
    irreplength: list
    full_offsets: list


@dataclass
class RHFStabilityResult:
    """
    Result of SO_RHF.rhf_stability_analysis(). eigenvalues/eigenvectors
    are from np.linalg.eigh (ascending order) of the full, un-pooled real
    singlet (1A+1B) RHF stability Hessian -- every occupied x virtual
    orbital pair, every degenerate-irrep partner independent (see that
    method's docstring). occ_irrep_of[k]/virt_irrep_of[k] give the point-
    group irrep index of the occupied/virtual orbital in rotation-index
    k of the Hessian (same ordering as the Hessian's rows/columns, NOT
    the same as eigenvector component ordering unless you're indexing
    eigenvectors[:, i][k]).
    """
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    occ_irrep_of: np.ndarray
    virt_irrep_of: np.ndarray


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
        #so_orbitals = SOrbitals(self.symtext, self.salcs, self.ndocc, self.options, self.nbfxns, fxn_list, self.basis)
        so_orbitals = SOrbitals(self.symtext, self.salcs, self.ndocc, self.options, self.nbfxns, fxn_list, self.basis, self.molecule, self.basis_input, bset)
        #so_orbitals.process_salcs()
        
        iter_type = "DIAG"
        if self.options.guess == 'sad':
            D_i, docc_vector = so_orbitals.D, None
        else:
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
        if self.options.guess == 'sad':
            # Unlike rhf.py's guess="sad" branch, D_i here is still the raw
            # SAD density -- so_orbitals.C/eps/Orbs[h].ndocc_ir are all still
            # None (see SOrbitals.__init__), and nothing below ever
            # populates them otherwise: build_D() just reads
            # Orbs[h].ndocc_ir back out, and the SOSCF Newton step needs a
            # real ndocc_ir per irrep for its Hessian shapes. Build an
            # initial Fock from the raw SAD density, diagonalize it to get
            # a real C/eps, and use that to assign ndocc_ir per irrep --
            # mirroring rhf.py's own guess="sad" initialization.
            F_guess, _ = self.build_fock_blocky_sym(so_orbitals.H, D_i, repacked_bigERI, repacked_bigERI_swapped)
            e_guess = self.degen_rhf_energy(D_i, so_orbitals.H, F_guess, so_orbitals) + self.enuc
            print(f"The initial SCF energy via SAD {e_guess}")
            Fs = so_orbitals.A.transpose().dot(F_guess.dot(so_orbitals.A))
            eps, Cs = Fs.eigh()
            C = so_orbitals.A.dot(Cs)
            if so_orbitals.Orbs[0].ndocc_ir is None:
                so_orbitals.ndocc_irrep(C, eps)
            so_orbitals.C = C
            D_i, docc_vector = self.build_D(so_orbitals)
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
                self.wfn_energy = E_new
                self.n_iterations = i
                # Mirrors rhf.py's SRHF.run() convention -- stashed for
                # post-hoc inspection (test/test_soscf_hessian.py builds
                # the SOSCF Hessian directly from these after convergence).
                self.so_orbitals = so_orbitals
                self.bigERI = bigERI
                self.C = C
                self.F = F
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
                # Newton-Raphson (SOSCF) orbital-rotation step. gn is
                # correctly restricted to same-irrep occ-virt pairs only
                # (standard fact: only same-irrep rotations have nonzero
                # orbital gradient for a closed-shell singlet reference).
                # The Hessian Biajb built below is block-diagonal BY IRREP
                # (every BDMatrix contraction here shares one irrep h across
                # all axes -- see bdmats.py) and so structurally excludes
                # coupling between same-irrep rotation pairs of DIFFERENT
                # irreps (h != h'). Such cross-irrep two-electron coupling
                # is generically nonzero by the standard selection rule
                # (h⊗h always contains the totally symmetric irrep, for any
                # h), so this Hessian is a genuine approximation, not just a
                # shape convenience. It's still correct to use as a Newton
                # step: the fixed point (gn=0) doesn't depend on which
                # Hessian approximation produced the step, only on the
                # gradient being exact -- confirmed empirically in
                # test/smoke_so_rhf.py (methane/Td, water/C2v, ammonia/C3v
                # with a genuinely degenerate E irrep, all converging to
                # match Psi4 to ~1e-9 or better under both exploit_degen
                # settings).
                moF = F.einsum('ui,vj,uv', C, C, F)
                occ_C = C.slicev2([":", ":ndocc_ir"], so_orbitals.Orbs)

                x_flat, active_by_irrep = self.soscf_newton_step(
                    bigERI, moF, occ_C, C, so_orbitals
                )

                U = []
                for h, Cirrep in enumerate(C.blocks):
                    if len(Cirrep) == 0:
                        U.append(np.array([]))
                    else:
                        U.append(np.zeros(Cirrep.shape))
                U = BDMatrix(U)

                pos = 0
                for h, orb in enumerate(so_orbitals.Orbs):
                    n = active_by_irrep[h]
                    if n == 0:
                        continue
                    block_x = x_flat[pos:pos + n].reshape(orb.ndocc_ir, orb.nvirt_ir)
                    U.blocks[h][:orb.ndocc_ir, orb.ndocc_ir:] = block_x
                    U.blocks[h][orb.ndocc_ir:, :orb.ndocc_ir] = -block_x.T
                    pos += n
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

    def soscf_newton_step(self, bigERI, moF, occ_C, C, so_orbitals):
        """
        Solve the Newton-Raphson orbital-rotation equations Biajb @ x = gn
        for the same-irrep occupied-virtual rotation parameters, using a
        SINGLE dense Hessian spanning every irrep at once -- unlike the
        original per-irrep BDMatrix construction (still visible in git
        history), which structurally excluded coupling between same-irrep
        rotation pairs (i,a)/(j,b) belonging to DIFFERENT irreps h != h'.

        That coupling is real: by character orthogonality h⊗h always
        contains the totally symmetric irrep for ANY irrep h (this holds
        even for abelian point groups with no degenerate irreps at all --
        h⊗h = A1 exactly for any 1D irrep), so the two-electron integral
        (ia|jb) is generically nonzero for two same-irrep rotation pairs
        even when their irreps differ. The old block-diagonal-by-irrep
        Hessian still converged to the correct final energy (the fixed
        point gn=0 doesn't depend on which Hessian approximation produced
        the step, only on the gradient being exact) but needed far more
        Newton iterations for real point-group symmetry than for C1 (see
        test/smoke_so_rhf.py's convergence-rate assertions).

        The rotation PARAMETERS themselves are still exactly one shared
        kappa per representative (i,a) pair per irrep, matching
        so_orbitals.C/Orbs[h].ndocc_ir's existing compressed-size storage --
        that part was already correct and is unchanged. What was missing
        under exploit_degen=True was accounting for what that ONE shared
        kappa physically means: it's applied IDENTICALLY to all degen_h
        partners of a degenerate irrep at once (that's what "exploiting
        degeneracy" means), not just to the one stored representative
        partner in isolation. This is now handled by _build_soscf_hessian
        (below): every irrep is tiled degen_h times (mirroring
        MP2.run_degen_tensor()'s tiling recipe -- see that method's
        docstring for why simple tiling, not group-representation-matrix
        machinery, is the right tool here too, confirmed numerically this
        session), the Hessian/gradient are computed in that fully-tiled
        space, and then summed ("pooled") back down to one entry per
        representative pair. Verified numerically (ammonia's E irrep,
        degen=2) that this reduces to gn_shared = degen_h * gn_representative
        for the gradient (no cross-partner term possible for a first
        derivative of a sum) and Biajb_shared = degen_h * Biajb_representative
        + degen_h*(degen_h-1) * Biajb_cross_partner for the Hessian (a
        naive "just multiply by degen_h" fix would have been wrong here --
        confirmed the true value differs substantially from that guess).
        When exploit_degen=False (or for a nondegenerate irrep, degen_h=1
        always), tiling is a no-op and this reduces exactly to the
        cross-irrep-only fix already validated in test/smoke_so_rhf.py.

        bigERI is the full, UNCOMPRESSED AO->SO ERI (built by aotoso_2 from
        every SALC partner row, not just the compressed representative
        ones) -- so locating irrep h's representative-partner block within
        it requires the irrep's FULL salc_set row count (self.salcs.
        salc_sets[h].shape[0]) to find where it STARTS, then extracting
        only so_orbitals.irreplength[h] rows (the representative partner)
        from that point. This is the same "full offsets vs. compressed
        irreplength" split srhf/degen_tensor.py's DegenIntegralFactory
        already uses (_full_offsets vs _compute_offsets(irreplength)) --
        confirmed equivalent to DegenIntegralFactory._transform(ERI_ao)'s
        diagonal (same-irrep) blocks by test/test_soscf_hessian.py, since
        slicing commutes with a linear transform's output columns.

        Split into _build_soscf_hessian (below) + this thin solve wrapper
        so test/test_soscf_hessian.py can inspect Biajb_dense/gn_flat/
        MO_dense/I_dense directly without re-deriving this logic
        independently in the test.
        """
        Biajb_dense, active_by_irrep, gn_flat, _, _, _, _, _ = self._build_soscf_hessian(
            bigERI, occ_C, C, moF, so_orbitals
        )
        if Biajb_dense.size == 0:
            return np.array([]), active_by_irrep
        x_flat = np.linalg.solve(Biajb_dense, gn_flat)
        return x_flat, active_by_irrep

    def _build_tiled_mo_data(self, bigERI, occ_C, C, moF, so_orbitals):
        """
        Build the combined-and-TILED AO->MO integral/Fock data (see
        TiledMOData's docstring for the tiling policy). Shared by
        _build_soscf_hessian (Newton step -- further restricts to
        same-irrep pairs and pools partners back down to one shared
        rotation parameter, see its own docstring for why) and
        rhf_stability_analysis (uses every tiled occ/virt index
        independently, no restriction, no pooling -- see its docstring
        for why pooling would be physically wrong for stability testing).

        Mirrors MP2.run_degen_tensor()'s tiling recipe (srhf/mp2.py) --
        see its docstring for the general "full SALC offsets vs
        compressed irreplength" distinction this also relies on to
        locate each partner's own rows within the full, uncompressed
        bigERI.
        """
        irreplength = so_orbitals.irreplength
        full_sizes = [salc.shape[0] for salc in self.salcs.salc_sets]
        full_offsets = BDMatrix.irrep_offsets(full_sizes)
        populated = [h for h in range(len(irreplength)) if irreplength[h] > 0]

        # Tiled combined space: irrep h contributes degen_h copies of its
        # representative-partner block, located via the FULL (not
        # irreplength-based) per-partner offset -- same distinction
        # DegenIntegralFactory's _full_offsets vs _compute_offsets(
        # irreplength) already draws. degen_h is gated by
        # self.options.exploit_degen because so_orbitals.irreplength[h]
        # (and hence C.blocks[h]/occ_C.blocks[h]/moF.blocks[h]) is ALREADY
        # the full, uncompressed size when exploit_degen=False -- tiling
        # further in that case would over-index (confirmed the hard way:
        # this is not a policy choice, it's reading the data correctly).
        idx_tiled_parts, occ_C_tiled_blocks, C_tiled_blocks, moF_tiled_blocks = [], [], [], []
        tile_start = {}  # irrep h -> this irrep's first-partner offset within the tiled combined space
        occ_tile_start = {}  # irrep h -> list of this irrep's per-partner offsets within the tiled occ-only axis
        o, oo = 0, 0
        for h in populated:
            degen_h = so_orbitals.symtext.irreps[h].d if self.options.exploit_degen else 1
            il_h = irreplength[h]
            ndocc_h = so_orbitals.Orbs[h].ndocc_ir
            tile_start[h] = o
            occ_tile_start[h] = []
            for mu in range(degen_h):
                idx_tiled_parts.append(np.arange(full_offsets[h] + mu * il_h, full_offsets[h] + (mu + 1) * il_h))
                occ_C_tiled_blocks.append(occ_C.blocks[h])
                C_tiled_blocks.append(C.blocks[h])
                moF_tiled_blocks.append(moF.blocks[h])
                occ_tile_start[h].append(oo)
                o += il_h
                oo += ndocc_h
        idx_tiled = np.concatenate(idx_tiled_parts)
        I_tiled = bigERI[np.ix_(idx_tiled, idx_tiled, idx_tiled, idx_tiled)]
        occ_C_tiled = block_diag(*occ_C_tiled_blocks)
        C_tiled = block_diag(*C_tiled_blocks)
        moF_tiled = block_diag(*moF_tiled_blocks)

        MO_tiled = np.einsum(
            'PQRS,Pp,Qq,Rr,Ss->pqrs', I_tiled, occ_C_tiled, C_tiled, C_tiled, C_tiled,
            optimize='optimal',
        )

        return TiledMOData(
            I_tiled=I_tiled, MO_tiled=MO_tiled, C_tiled=C_tiled, occ_C_tiled=occ_C_tiled,
            moF_tiled=moF_tiled, tile_start=tile_start, occ_tile_start=occ_tile_start,
            populated=populated, irreplength=irreplength, full_offsets=full_offsets,
        )

    def _build_soscf_hessian(self, bigERI, occ_C, C, moF, so_orbitals):
        """
        Build the dense, cross-irrep-AND-cross-partner-coupled orbital-
        rotation Hessian Biajb_dense and gradient gn_flat (shape
        (n_active, n_active) / (n_active,), row/column order given by
        active_by_irrep -- see soscf_newton_step's docstring for the full
        design rationale), plus representative-partner-only MO_dense/
        I_dense/occ_num/comb_occ/comb_virt (unaffected by the tiling below
        -- kept for test/test_soscf_hessian.py's cross-irrep-focused
        checks and the DegenIntegralFactory diagonal-block regression).

        Internally builds a TILED combined space -- every irrep's
        representative-partner block duplicated degen_h times (1 if
        exploit_degen=False, matching MP2.run_degen_tensor()'s recipe --
        see its docstring in mp2.py) -- computes the Hessian/gradient
        formula there (same formula as the representative-only case,
        automatically picking up real cross-partner two-electron coupling
        for pairs living in different partners of the same degenerate
        irrep), then SUMS ("pools") each representative pair's degen_h
        tiled copies back down to one Biajb_dense/gn_flat entry -- this is
        the "shared kappa applied identically to every partner" physics,
        not a new degeneracy-aware parametrization (the active-pair count
        stays exactly ndocc_h*nvirt_h per irrep, never scaled by degen_h).
        """
        tiled = self._build_tiled_mo_data(bigERI, occ_C, C, moF, so_orbitals)
        irreplength = tiled.irreplength
        full_offsets = tiled.full_offsets
        populated = tiled.populated
        tile_start = tiled.tile_start
        occ_tile_start = tiled.occ_tile_start
        I_tiled = tiled.I_tiled
        MO_tiled = tiled.MO_tiled
        moF_tiled = tiled.moF_tiled

        combined_offsets = BDMatrix.irrep_offsets(irreplength)
        occ_offsets = BDMatrix.irrep_offsets([orb.ndocc_ir for orb in so_orbitals.Orbs])

        # Representative-partner-only (degen_h=1 slice of the above) --
        # exactly reproduces the previous, already-validated construction;
        # returned for test/test_soscf_hessian.py's cross-irrep-only checks.
        idx = np.concatenate([
            np.arange(full_offsets[h], full_offsets[h] + irreplength[h]) for h in populated
        ])
        I_dense = bigERI[np.ix_(idx, idx, idx, idx)]
        occ_C_combined = block_diag(*[occ_C.blocks[h] for h in populated])
        C_combined = block_diag(*[C.blocks[h] for h in populated])
        MO_dense = np.einsum(
            'PQRS,Pp,Qq,Rr,Ss->pqrs', I_dense, occ_C_combined, C_combined, C_combined, C_combined,
            optimize='optimal',
        )

        occ_num, comb_occ, comb_virt, active_by_irrep = [], [], [], []
        occ_num_tiled, comb_occ_tiled, comb_virt_tiled, pool_pair = [], [], [], []
        pair_id = 0
        for h, orb in enumerate(so_orbitals.Orbs):
            ndocc_h, nvirt_h = orb.ndocc_ir, orb.nvirt_ir
            if ndocc_h == 0 or nvirt_h == 0:
                active_by_irrep.append(0)
                continue
            active_by_irrep.append(ndocc_h * nvirt_h)
            degen_h = so_orbitals.symtext.irreps[h].d if self.options.exploit_degen else 1
            il_h = irreplength[h]
            for i_local in range(ndocc_h):
                for a_local in range(nvirt_h):
                    occ_num.append(occ_offsets[h] + i_local)
                    comb_occ.append(combined_offsets[h] + i_local)
                    comb_virt.append(combined_offsets[h] + ndocc_h + a_local)
                    for mu in range(degen_h):
                        occ_num_tiled.append(occ_tile_start[h][mu] + i_local)
                        comb_occ_tiled.append(tile_start[h] + mu * il_h + i_local)
                        comb_virt_tiled.append(tile_start[h] + mu * il_h + ndocc_h + a_local)
                        pool_pair.append(pair_id)
                    pair_id += 1

        if not occ_num_tiled:
            empty = np.array([])
            return empty, active_by_irrep, empty, np.array(occ_num), np.array(comb_occ), np.array(comb_virt), MO_dense, I_dense

        occ_num_tiled = np.array(occ_num_tiled)
        comb_occ_tiled = np.array(comb_occ_tiled)
        comb_virt_tiled = np.array(comb_virt_tiled)
        pool_pair = np.array(pool_pair)
        n_active = pair_id

        delta_occ = occ_num_tiled[:, None] == occ_num_tiled[None, :]
        delta_virt = comb_virt_tiled[:, None] == comb_virt_tiled[None, :]
        F_vv = moF_tiled[np.ix_(comb_virt_tiled, comb_virt_tiled)]
        F_oo = moF_tiled[np.ix_(comb_occ_tiled, comb_occ_tiled)]
        fock_term = np.where(delta_occ, F_vv, 0.0) - np.where(delta_virt, F_oo, 0.0)

        # (ia|jb), (ij|ab), (ib|ja) over every TILED (partner-resolved)
        # pair combination -- row p = tile (i,a), column q = tile (j,b).
        # For p, q belonging to different partners of the same (or a
        # different) irrep, this is exactly the same formula already used
        # for cross-irrep coupling -- the Fock-delta terms above are
        # exactly zero for any cross-partner combination (occ_num_tiled/
        # comb_virt_tiled never coincide across distinct partners), so only
        # the genuine two-electron coupling survives, automatically.
        ia_jb = MO_tiled[occ_num_tiled[:, None], comb_virt_tiled[:, None], comb_occ_tiled[None, :], comb_virt_tiled[None, :]]
        ij_ab = MO_tiled[occ_num_tiled[:, None], comb_occ_tiled[None, :], comb_virt_tiled[:, None], comb_virt_tiled[None, :]]
        ib_ja = MO_tiled[occ_num_tiled[:, None], comb_virt_tiled[None, :], comb_occ_tiled[None, :], comb_virt_tiled[:, None]]

        Biajb_tiled = 4.0 * (fock_term + 4.0 * ia_jb - ib_ja - ij_ab)
        gn_tiled = -4.0 * moF_tiled[comb_occ_tiled, comb_virt_tiled]

        # Pool: sum every tiled entry belonging to representative pair p
        # (the "shared kappa applied identically to all degen_h partners"
        # physics) down to one Biajb_dense/gn_flat entry per pair.
        n_tiled = len(occ_num_tiled)
        P = np.zeros((n_active, n_tiled))
        P[pool_pair, np.arange(n_tiled)] = 1.0
        Biajb_dense = P @ Biajb_tiled @ P.T
        gn_flat = P @ gn_tiled

        return Biajb_dense, active_by_irrep, gn_flat, np.array(occ_num), np.array(comb_occ), np.array(comb_virt), MO_dense, I_dense

    def _gather_stability_pairs(self, tiled, so):
        """
        Shared index-gathering for rhf_stability_analysis (singlet) and
        uhf_stability_analysis (triplet): every occupied x virtual orbital
        pair (every irrep combination, every degenerate-irrep partner
        independent -- NOT restricted to same-irrep "active" pairs, and
        NOT pooled across partners, unlike _build_soscf_hessian's Newton-
        step Hessian). Restricting to same-irrep pairs would silently hide
        a direction with zero gradient (by symmetry) but negative
        curvature -- a point-group instability. Pooling degenerate-irrep
        partners would make it structurally impossible to detect a
        distortion that breaks a partner's own degeneracy (a
        Jahn-Teller-type instability) -- so this always explores every
        partner independently.

        occ_num: occ-only numbering, matching MO_tiled's axis 0 (built
        from occ_C_tiled). comb_occ/comb_virt: full combined-space
        numbering, matching MO_tiled's other 3 axes (built from C_tiled)
        and moF_tiled (both axes) -- same two-numbering-system distinction
        _build_soscf_hessian already draws for its own occ_num vs
        comb_occ/comb_virt, just without the same-irrep restriction and
        without pooling.

        fock_term/ij_ab/ib_ja are shared verbatim between the singlet and
        triplet stability formulas; the (ia|jb) Coulomb term is singlet-
        only (triplet/spin-flip excitations have no same-spin Coulomb
        ladder contribution) and is left to rhf_stability_analysis to
        gather itself from the returned i_num_arr/i_comb_arr/a_arr.

        Returns (occ_irrep_of, virt_irrep_of, i_num_arr, i_comb_arr, a_arr,
        fock_term, ij_ab, ib_ja); all eight are empty arrays when there are
        no active occupied-virtual pairs to test.
        """
        occ_num, comb_occ, comb_virt, occ_irrep, virt_irrep = [], [], [], [], []
        for h in tiled.populated:
            degen_h = so.symtext.irreps[h].d if self.options.exploit_degen else 1
            ndocc_h, nvirt_h = so.Orbs[h].ndocc_ir, so.Orbs[h].nvirt_ir
            il_h = tiled.irreplength[h]
            for mu in range(degen_h):
                base = tiled.tile_start[h] + mu * il_h
                occ_base = tiled.occ_tile_start[h][mu]
                for i_local in range(ndocc_h):
                    occ_num.append(occ_base + i_local)
                    comb_occ.append(base + i_local)
                    occ_irrep.append(h)
                for a_local in range(nvirt_h):
                    comb_virt.append(base + ndocc_h + a_local)
                    virt_irrep.append(h)

        empty = np.array([])
        if not occ_num or not comb_virt:
            return empty, empty, empty, empty, empty, empty, empty, empty

        occ_num = np.array(occ_num)
        comb_occ = np.array(comb_occ)
        comb_virt = np.array(comb_virt)
        occ_irrep = np.array(occ_irrep)
        virt_irrep = np.array(virt_irrep)

        n_occ_full, n_virt_full = len(occ_num), len(comb_virt)
        i_num_arr = np.repeat(occ_num, n_virt_full)     # for MO_tiled's occ-only axis
        i_comb_arr = np.repeat(comb_occ, n_virt_full)   # for MO_tiled's other axes / moF_tiled
        a_arr = np.tile(comb_virt, n_occ_full)
        occ_irrep_of = np.repeat(occ_irrep, n_virt_full)
        virt_irrep_of = np.tile(virt_irrep, n_occ_full)

        MO_tiled = tiled.MO_tiled
        moF_tiled = tiled.moF_tiled

        delta_occ = i_comb_arr[:, None] == i_comb_arr[None, :]
        delta_virt = a_arr[:, None] == a_arr[None, :]
        F_vv = moF_tiled[np.ix_(a_arr, a_arr)]
        F_oo = moF_tiled[np.ix_(i_comb_arr, i_comb_arr)]
        fock_term = np.where(delta_occ, F_vv, 0.0) - np.where(delta_virt, F_oo, 0.0)

        ij_ab = MO_tiled[i_num_arr[:, None], i_comb_arr[None, :], a_arr[:, None], a_arr[None, :]]
        ib_ja = MO_tiled[i_num_arr[:, None], a_arr[None, :], i_comb_arr[None, :], a_arr[:, None]]

        return occ_irrep_of, virt_irrep_of, i_num_arr, i_comb_arr, a_arr, fock_term, ij_ab, ib_ja

    def rhf_stability_analysis(self):
        """
        Real, singlet RHF->RHF wavefunction stability analysis: diagonalize
        the full (1A+1B) orbital-rotation Hessian over EVERY occupied x
        virtual orbital pair (see _gather_stability_pairs's docstring for
        why every irrep combination and every degenerate-irrep partner is
        explored independently). A negative eigenvalue means there is an
        occupied-virtual rotation that lowers the energy -- the wavefunction
        is a saddle point, not a true minimum, in the tested space.

        Formula is the standard real singlet (1A+1B) TDHF/RPA stability
        matrix:
            H_iajb = delta_ij F_ab - delta_ab F_ij + 4(ia|jb) - (ij|ab) - (ib|ja)
        identical to _build_soscf_hessian's Biajb_dense formula (up to that
        method's x4 Newton-step scaling, which doesn't affect eigenvalue
        sign) -- confirmed by direct comparison, and the resulting
        eigenvalue spectrum matches Psi4's own stability_analysis='check'
        output to 5-6 decimal places for every molecule tested (see
        test/test_rhf_stability.py), independent of whether this job used
        exploit_degen=True or False (confirmed bit-identical, ~1e-14,
        either way -- as it must be, since both describe the same physical
        wavefunction; _build_tiled_mo_data already handles the bookkeeping
        for correctly reconstructing the full space regardless of which
        setting this job used).

        Must be called after run() has converged (uses the same
        self.bigERI/self.C/self.F/self.so_orbitals/self.salcs stashed at
        the convergence break, already relied on by
        test/test_soscf_hessian.py's external _build_soscf_hessian calls).
        """
        if not hasattr(self, "wfn_energy"):
            raise RuntimeError("rhf_stability_analysis() requires a converged run() first")

        so = self.so_orbitals
        occ_C = self.C.slicev2([":", ":ndocc_ir"], so.Orbs)
        moF = self.F.einsum('ui,vj,uv', self.C, self.C, self.F)
        tiled = self._build_tiled_mo_data(self.bigERI, occ_C, self.C, moF, so)

        occ_irrep_of, virt_irrep_of, i_num_arr, i_comb_arr, a_arr, fock_term, ij_ab, ib_ja = \
            self._gather_stability_pairs(tiled, so)

        if occ_irrep_of.size == 0:
            empty = np.array([])
            return RHFStabilityResult(eigenvalues=empty, eigenvectors=empty,
                                       occ_irrep_of=empty, virt_irrep_of=empty)

        ia_jb = tiled.MO_tiled[i_num_arr[:, None], a_arr[:, None], i_comb_arr[None, :], a_arr[None, :]]

        H_iajb = fock_term + 4.0 * ia_jb - ib_ja - ij_ab
        eigvals, eigvecs = np.linalg.eigh(H_iajb)

        return RHFStabilityResult(eigenvalues=eigvals, eigenvectors=eigvecs,
                                   occ_irrep_of=occ_irrep_of, virt_irrep_of=virt_irrep_of)

    def uhf_stability_analysis(self):
        """
        Real, triplet RHF->UHF wavefunction stability analysis: diagonalize
        the full (3A+3B) orbital-rotation Hessian over the same tiled,
        un-pooled occupied x virtual space as rhf_stability_analysis (see
        _gather_stability_pairs's docstring). A negative eigenvalue means
        allowing the alpha and beta orbitals to split (breaking spin
        symmetry) lowers the energy -- an unrestricted, UHF-type solution
        exists below this RHF one.

        Formula is the standard real triplet (3A+3B) TDHF/RPA stability
        matrix:
            H_iajb = delta_ij F_ab - delta_ab F_ij - (ij|ab) - (ib|ja)
        i.e. rhf_stability_analysis's H_iajb with the 4(ia|jb) Coulomb term
        dropped entirely -- triplet/spin-flip excitations have no same-spin
        Coulomb ladder contribution. Validated three independent ways: (1)
        finite difference of a from-scratch UHF energy functional
        (kappa_alpha=+t*v, kappa_beta=-t*v for a single occ-virt pair)
        against this formula's diagonal entries, (2) the same finite-
        difference technique with two simultaneously-perturbed pairs
        against off-diagonal entries, and (3) the full eigenvalue spectrum
        against Psi4's own stability_analysis='check' "Lowest triplet
        (RHF->UHF) stability eigenvalues:" output (water/STO-3G, no
        truncation; ammonia/STO-3G, genuinely degenerate E irrep, both
        exploit_degen settings) -- exact match to 5-6 decimal places in
        every case (see test/test_rhf_stability.py). Note: the raw
        orbital-rotation Hessian (what a naive finite difference computes
        directly, matching _build_soscf_hessian's Biajb_dense) is 4x this
        H_iajb -- the "stability eigenvalue" convention (matching Psi4's
        own reported numbers, both singlet and triplet) carries no outer
        factor of 4.

        Must be called after run() has converged, same as
        rhf_stability_analysis.
        """
        if not hasattr(self, "wfn_energy"):
            raise RuntimeError("uhf_stability_analysis() requires a converged run() first")

        so = self.so_orbitals
        occ_C = self.C.slicev2([":", ":ndocc_ir"], so.Orbs)
        moF = self.F.einsum('ui,vj,uv', self.C, self.C, self.F)
        tiled = self._build_tiled_mo_data(self.bigERI, occ_C, self.C, moF, so)

        occ_irrep_of, virt_irrep_of, _, _, _, fock_term, ij_ab, ib_ja = \
            self._gather_stability_pairs(tiled, so)

        if occ_irrep_of.size == 0:
            empty = np.array([])
            return RHFStabilityResult(eigenvalues=empty, eigenvectors=empty,
                                       occ_irrep_of=empty, virt_irrep_of=empty)

        H_iajb = fock_term - ij_ab - ib_ja
        eigvals, eigvecs = np.linalg.eigh(H_iajb)

        return RHFStabilityResult(eigenvalues=eigvals, eigenvectors=eigvecs,
                                   occ_irrep_of=occ_irrep_of, virt_irrep_of=virt_irrep_of)

    def _report_stability(self, result, title, verdict_unstable, verdict_stable,
                           n=10, neg_threshold=-1e-6, cluster_tol=1e-6):
        """
        Shared clustering/formatting body for report_rhf_stability (singlet)
        and report_uhf_stability (triplet).

        The "(occ irrep, virt irrep) block weight" column is a cheap,
        HONEST diagnostic, not a rigorous irreducible-representation label:
        it's each near-degenerate eigenvalue CLUSTER's weight (summed over
        the whole cluster, not per individual eigenvector -- np.linalg.eigh
        returns an arbitrary orthonormal basis WITHIN a repeated eigenspace,
        e.g. ammonia's genuine E-irrep degeneracy, so a per-eigenvector
        weight would be a basis artifact, not physics) living in each
        (occupied irrep, virtual irrep) pair block. A single such block can
        itself further decompose into multiple irreducible components for
        non-Abelian groups (e.g. C3v's E(x)E = A1+A2+E) -- getting the TRUE
        irreducible label of each mode needs a Clebsch-Gordan/projection-
        operator decomposition (feasible via symtext.irrep_mats, confirmed
        available in an earlier investigation, but not implemented here --
        an explicit follow-on, not this pass).
        """
        so = self.so_orbitals
        irrep_symbols = [ir.symbol for ir in so.symtext.irreps]
        eigvals, eigvecs = result.eigenvalues, result.eigenvectors
        n_total = len(eigvals)
        if n_total == 0:
            print("No active occupied-virtual rotations to test.")
            return

        clusters = []
        start = 0
        for k in range(1, n_total):
            if eigvals[k] - eigvals[start] > cluster_tol:
                clusters.append((start, k))
                start = k
        clusters.append((start, n_total))

        def clean(symbol):
            return symbol.replace("_", "")

        print(title)
        print(f"  {'eigenvalue':>12s}  {'deg':>3s}  dominant (occ,virt) blocks (fraction of character)")
        shown = 0
        for lo, hi in clusters:
            if shown >= n:
                break
            avg_eig = np.mean(eigvals[lo:hi])
            deg = hi - lo
            sq = np.sum(eigvecs[:, lo:hi] ** 2, axis=1)
            weights = {}
            for ho in range(len(irrep_symbols)):
                for hv in range(len(irrep_symbols)):
                    mask = (result.occ_irrep_of == ho) & (result.virt_irrep_of == hv)
                    if not mask.any():
                        continue
                    # Normalize by deg -- fraction of character (0..1),
                    # independent of how many roots are in this cluster,
                    # rather than a raw sum that grows with deg.
                    w = float(np.sum(sq[mask])) / deg
                    if w > 0.02:
                        weights[(ho, hv)] = w
            dominant = sorted(weights.items(), key=lambda kv: -kv[1])[:3]
            label = ", ".join(f"{clean(irrep_symbols[ho])}x{clean(irrep_symbols[hv])}={w:.2f}" for (ho, hv), w in dominant)
            print(f"  {avg_eig:>12.6f}  {deg:>3d}  {label}")
            shown += 1

        lowest = eigvals[0]
        if lowest < neg_threshold:
            print(f"{verdict_unstable} lowest eigenvalue {lowest:.6f} < {neg_threshold}")
        else:
            print(f"{verdict_stable} lowest eigenvalue {lowest:.6f}")

    def report_rhf_stability(self, result, n=10, neg_threshold=-1e-6, cluster_tol=1e-6):
        """Print a Psi4-style stability report ("Lowest singlet (RHF->RHF)
        stability eigenvalues:") for a result from rhf_stability_analysis().
        See _report_stability's docstring for the block-weight diagnostic's
        meaning and limitations."""
        self._report_stability(
            result,
            title="Lowest RHF singlet (RHF->RHF) stability eigenvalues:",
            verdict_unstable="RHF determinant is UNSTABLE:",
            verdict_stable="RHF determinant is stable:",
            n=n, neg_threshold=neg_threshold, cluster_tol=cluster_tol,
        )

    def report_uhf_stability(self, result, n=10, neg_threshold=-1e-6, cluster_tol=1e-6):
        """Print a Psi4-style stability report ("Lowest triplet (RHF->UHF)
        stability eigenvalues:") for a result from uhf_stability_analysis().
        See _report_stability's docstring for the block-weight diagnostic's
        meaning and limitations."""
        self._report_stability(
            result,
            title="Lowest triplet (RHF->UHF) stability eigenvalues:",
            verdict_unstable="RHF determinant is UNSTABLE to spin-symmetry breaking (RHF->UHF):",
            verdict_stable="RHF determinant is stable to spin-symmetry breaking (RHF->UHF):",
            n=n, neg_threshold=neg_threshold, cluster_tol=cluster_tol,
        )

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
