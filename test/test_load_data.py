import numpy as np

data = np.load("data/nh3_ccpvdz_diis_fixture.npz")
for k in data:
    print(k, data[k].shape)
