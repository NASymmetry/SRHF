class Options(object):
    def __init__(self, **kwargs):
        #Symmetry
        self.subgroup = kwargs.pop("subgroup", False)
        self.exploit_degen = kwargs.pop("exploit_degen", True)
        self.fg_as_c1 = kwargs.pop("fg_as_c1", False)

        #Starting Guess
        self.docc = kwargs.pop("docc", None)
        self.guess = kwargs.pop("guess", "core")

        #SCF Iterations
        self.scf_max_iter = kwargs.pop("scf_max_iter", 50)
        self.e_convergence = kwargs.pop("e_convergence", 1e-7)
        self.d_convergence = kwargs.pop("d_convergence", 1e-7)

        #DIIS
        self.diis = kwargs.pop("diis", True)
        self.diis_start = kwargs.pop("diis_start", 2)
        self.diis_length = kwargs.pop("diis_length", 6)

        #benchmark
        self.benchmark = kwargs.pop("benchmark", False)
        self.compare_psi = kwargs.pop("compare_psi", True)
        
        #Ints source
        self.intsdpd = kwargs.pop("intsdpd", True)


