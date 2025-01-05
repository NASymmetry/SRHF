import numpy as np

class MO_Trans():
    def __init__(self, docc_vector, so_orbitals, symtext):
        self.docc_vector = docc_vector
        self.so_orbitals = so_orbitals
        self.symtext = symtext

    def mo_eri(self, E, C):

        #first,get blocks
        bmo_eri = self.blocks_oggg(E,C)
        print("Now transform slices")

    def blocks_oggg(self, E, C):
        print("Transform TEI AO-MO")
        print(self.so_orbitals.irreplength)
        print("Docc vector")
        print(self.docc_vector)
        self.virt_vector = list(np.array(self.so_orbitals.irreplength) - np.array(self.docc_vector))
        print(self.virt_vector)
        self.blocks = [] 
        for i, ir in enumerate(self.docc_vector):
            if ir != 0:
                for j, jr in enumerate(self.so_orbitals.irreplength):
                    if jr != 0:
                        for a, ar in enumerate(self.so_orbitals.irreplength):
                            if ar != 0:
                                for b, br in enumerate(self.so_orbitals.irreplength):
                                    if br != 0:
                                        if self.dp_contains_tsir(i, j, a, b):
                                            print(f"{ir, jr, ar, br}")
                                            self.blocks.append([i, j, a, b])
        print("The blocks")
        print(self.blocks)
        


        return self.blocks
    def blocks_oovv(self, E, C):
        print("Transform TEI AO-MO")
        print(self.so_orbitals.irreplength)
        print("Docc vector")
        print(self.docc_vector)
        self.virt_vector = list(np.array(self.so_orbitals.irreplength) - np.array(self.docc_vector))
        print(self.virt_vector)
        self.blocks = [] 
        for i, ir in enumerate(self.docc_vector):
            if ir != 0:
                for j, jr in enumerate(self.docc_vector):
                    if jr != 0:
                        for a, ar in enumerate(self.virt_vector):
                            if ar != 0:
                                for b, br in enumerate(self.virt_vector):
                                    if br != 0:
                                        if self.dp_contains_tsir(i, j, a, b):
                                            print(f"{ir, jr, ar, br}")
                                            self.blocks.append([i, j, a, b])
        print("The blocks")
        print(self.blocks)

    def dp_contains_tsir(self, a, b, *args):
        #ctab = self.symtext.chartable
        ctab = self.symtext.character_table
        #a = ctab.characters[a]
        #b = ctab.characters[b]
        print(f"irrep indices a and b {a, b}")
        a = ctab[a]
        b = ctab[b]
        print(f"a b {a, b}")
        chars = a * b
        for arg in args:
            #chars *= ctab.characters[arg]
            chars *= ctab[arg]
        #s = sum(chars * ctab.class_orders * ctab.characters[0])
        #s = sum(chars * ctab.class_orders * ctab[0])
        s = sum(chars * self.symtext.class_orders * ctab[0])
        #n = s / sum(ctab.class_orders)
        n = s / sum(self.symtext.class_orders)
        if np.isclose(n, 0, atol = 1e-4):
            return False
        return True
