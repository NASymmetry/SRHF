import numpy as np
from srhf.srhf_helper import SOrbitals

def test_ao_to_so_transform(nh3_ccpvdz_data, nh3_ccpvdz_integrals, nh3_ccpvdz_so_integrals):
    """Ensure AO→SO transformation preserves structure and matches reference SO integrals."""
    data = nh3_ccpvdz_data
    ao_ints = nh3_ccpvdz_integrals
    so_ref = nh3_ccpvdz_so_integrals

    # Initialize full SOrbitals object
    so = SOrbitals(
        data["symtext"],
        data["salcs"],
        data["ndocc"],
        data["options"],
        data["nbfxns"],
        data["fxn_list"],
        data["basis"],
        data["molecule"],
        data["basis_input"],
        data["bset"],
    )

    # Transform AO integrals to SO basis
    for key in ["S", "T", "V"]:
        ao = ao_ints[key]
        so_mat = so.ao_to_so(ao)

        # Compare each block to reference .npz file
        ref_blocks = so_ref[f"{key}_blocks"]
        for block, ref_block in zip(so_mat.blocks, ref_blocks):
            assert np.allclose(block, ref_block, atol=1e-6), f"{key} block mismatch"

