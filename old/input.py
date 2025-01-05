Settings = dict()

Settings["basis"] = "sto-3g"
Settings["molecule"] = """
  noreorient
  0 1
  units bohr
O       0.00000000     0.00000000     0.12820053
H      -0.00000000    -1.47972477    -1.01731828
H       0.00000000     1.47972477    -1.01731828
"""
Settings["guess"] = "core"
#Settings["algo"] = "blocky_sym"
Settings["algo"] = "sparse"
Settings["nalpha"] = 5
Settings["nbeta"] = 5
Settings["scf_max_iter"] = 100
Settings["e_converge"] = 1e-12
Settings["d_converge"] = 1e-12
Settings["DOCC"] = None
#Settings["DOCC"] = [3, 0, 1, 1]

Settings = dict()

Settings["basis"] = "sto-3g"
Settings["molecule"] = """
  noreorient
  0 1
O            0.000000000000     0.000000000000    -0.068516219320
H            0.000000000000    -0.790689573744     0.543701060715
H            0.000000000000     0.790689573744     0.543701060715
"""
Settings["guess"] = "core"
#Settings["algo"] = "blocky_sym"
Settings["algo"] = "sparse"
Settings["nalpha"] = 5
Settings["nbeta"] = 5
Settings["scf_max_iter"] = 100
Settings["e_converge"] = 1e-12
Settings["d_converge"] = 1e-12
Settings["DOCC"] = None
#Settings["DOCC"] = [3, 0, 1, 1]
