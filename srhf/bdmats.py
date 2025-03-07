import numpy as np
from scipy.linalg import fractional_matrix_power

class BDMatrix():
    """
    Block diagonal matrix object.
    Defines functions to speed up computation by ignoring zero blocks.
    Can be tempermental so be careful when using these bad boys.
    Also, probably not optimized, but I'm not optimizing things in Python...
    """
    def __init__(self, blocks):
        #print("did we init?")
        self.blocks = blocks

    def __add__(self, A):
        if self.check_size(A):
            B = []
            for i, block in enumerate(self.blocks):
                B.append(block + A.blocks[i])
        return BDMatrix(B)

    #def __sub__(self, A):
    #    return self+(-1*A)
    def __sub__(self, A):
        if self.check_size(A):
            B = []
            for i, block in enumerate(self.blocks):
                B.append(block - A.blocks[i])
        return BDMatrix(B)

    def __mul__(self, n):
        if type(n) is int or float:
            B = []
            for i, block in enumerate(self.blocks):
                B.append(n * block)
            return BDMatrix(B)
            #raise ValueError(f":{type(n)} multiplication with bdmat is not yet defined")
            
        B = []
        for i, block in enumerate(self.blocks):
            B.append(np.multiply(block,n.blocks[i]))
        return BDMatrix(B)

    def __rmul__(self, n):
        return self.__mul__(n)

    def sum(self):
        suum = 0
        for i, block in enumerate(self.blocks):
            if np.size(block) == 0:
                continue
            else:
                suum += sum(sum(block))
        return suum
    def qr(self):
        Q = []
        R = []
        for i, block in enumerate(self.blocks):
            if block.size == 0:
                Q.append(block)
            else:
                q, r = np.linalg.qr(block)
                Q.append(q)
                R.append(r)
        return BDMatrix(Q), BDMatrix(R)

    def dot(self, A):
        if self.check_size(A):
            B = []
            for i, block in enumerate(self.blocks):
                if block.size == 0:
                    B.append(block)
                elif block.size == 1:
                    B.append(np.array([block[0]*A.blocks[i][0]]))
                else:
                    B.append(np.dot(block, A.blocks[i]))
        return BDMatrix(B)

    def transpose(self, args = None):
        B = []
        for i, block in enumerate(self.blocks):
            if block.size == 0:
                B.append(block)
            else:
                if args == None:
                    B.append(block.transpose())
                else:
                    B.append(block.transpose(*args))
        return BDMatrix(B)
    
    def eigh(self):
        eigval = []
        eigvec = []
        for i, block in enumerate(self.blocks):
            if np.size(block) < 1:
                eigvec.append(np.array([]))
                eigval.append(np.empty((0,)))
            else:
                e,v = np.linalg.eigh(block)
                eigval.append(e)
                eigvec.append(v)
        return eigval, BDMatrix(eigvec)

    def __pow__(self, n):
        B = []
        for i, block in enumerate(self.blocks):
            if block.size == 0:
                B.append(block)
            elif block.size == 1:
                B.append(np.array([block[0]**n]))
            else:
                B.append(fractional_matrix_power(block, n))
        return BDMatrix(B)
    def frob_norm(self):
        Sum = 0
        for i, block in enumerate(self.blocks):
            if len(block) != 0:
                Sum += np.linalg.norm(x=block,ord='fro')
        return Sum
    def T(self):
        B = [] 
        for i, block in enumerate(self.blocks):
            if block.size == 0:
                B.append(block)
            else:
                B.append(block.T)
        return BDMatrix(B)
    def check_size(self, A):
        for i,block in enumerate(self.blocks):
            if np.shape(A.blocks[i])[0] != np.shape(block)[0]:
                raise ValueError(": Arrays do not have same shape")
        return True

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.blocks)

    #def sblock(self, n):
    #    return self.blocks[n]

    def full_mat(self):
        fshape = 0
        for i, block in enumerate(self.blocks):
            fshape += np.shape(block)[0]
        fullmat = np.zeros((fshape,fshape))
        c = 0
        for i, block in enumerate(self.blocks):
            s = np.shape(block)[0]
            fullmat[c:c+s, c:c+s] = block
            c += s
        return fullmat

    def full_to_bd(self, irreplength):
        B = []
        offset = 0
        for i, il in enumerate(irreplength):
            if il == 0:
                B.append(np.array([]))
            else:
                B.append(self[offset:offset + il, offset:offset + il])
            offset += il
        return BDMatrix(B)
    
    def furtherv2(self, s, Orbs_h):
        stuff = []
        for s_i in s:
            if s_i == "":
                stuff.append(None)
            else:
                stuff.append(getattr(Orbs_h, s_i))
        stuff.append(None)
        return stuff

    def further(self, s, Orbs):
        stuff = []
        for s_i in s:
            if s_i == "":
                stuff.append(None)
            else:
                stuff.append(getattr(Orbs[0], s_i))
        stuff.append(None)
        return stuff

    def process_string(self, string_slice, Orbs):
        trials = []
        #loop over each index k 
        for i, string in enumerate(string_slice):
            #split the string so we know the beginning and end point of each slice
            s = string.split(":")
            trials.append(slice(*self.further(s, Orbs)))
        return tuple(trials)
    
    def process_stringv2(self, string_slice, Orbs_h):
        trials = []
        #loop over each index k 
        for i, string in enumerate(string_slice):
            #split the string so we know the beginning and end point of each slice
            s = string.split(":")
            trials.append(slice(*self.furtherv2(s, Orbs_h)))
        return tuple(trials)

    def inv(self):
        B = []
        for i, block in enumerate(self.blocks):
            if block.size == 0:
                B.append(block)
            else:
                B.append(np.linalg.inv(block))
        return BDMatrix(B)

    def slicev2(self, in_slice, Orbs, mat = None):
        #p_string = self.process_string(in_slice, Orbs)
        B = []
        #    raise ValueError("This function only works for one irrep, c1 symmetry")
        for h, block in enumerate(self.blocks):
            if len(self.blocks[h]) == 0:
                B.append(np.array([]))
            else:
                p_string = self.process_stringv2(in_slice, Orbs[h])
                if mat != None:
                    self.blocks[h][p_string] = mat.blocks[h]
                    B.append(self.blocks[h][p_string])
                else:
                    B.append(self.blocks[h][p_string])
        return BDMatrix(B)

    def slice(self, in_slice, Orbs, mat = None):
        p_string = self.process_string(in_slice, Orbs)
        B = []
        if len(self.blocks) > 1:
            raise ValueError("This function only works for one irrep, c1 symmetry")
        for h, block in enumerate(self.blocks):
            if len(self.blocks[h]) == 0:
                B.append(np.array([]))
            else:
                if mat != None:
                    self.blocks[0][p_string] = mat.blocks[h]
                    B.append(self.blocks[0][idk_wut])
                else:
                    B.append(self.blocks[0][idk_wut])
        return BDMatrix(B)

    def einsum(self, string, *stuff):
        #check if *stuff are BDMatrix objects
        if any(isinstance(st, BDMatrix) == False for st in stuff):
            raise ValueError("BDMatrix.einsum() only works with BDMatrix *args objects")
        else:
            B = []
            for h, block in enumerate(self.blocks):
                if len(self.blocks[h]) == 0:
                    B.append(np.array([]))
                else:
                    idk = [stuff[i].blocks[h] for i in range(len(stuff))]
                    B.append(np.einsum(string, *[stuff[i].blocks[h] for i in range(len(stuff))]))
        return BDMatrix(B)

    def reshape(self, *args):
        B = []
        for h, block in enumerate(self.blocks):
            if len(self.blocks[h]) == 0:
                B.append(np.array([]))
            else:
                B.append(block.reshape(*args[h]))
        return BDMatrix(B)
     
    def swapaxes(self, *args):
        B = []
        for h, block in enumerate(self.blocks):
            if len(self.blocks[h]) == 0:
                B.append(np.array([]))
            else:
                B.append(block.swapaxes(*args))
        return BDMatrix(B)

if __name__ == "__main__":
    A = np.array([[1,2],[3,4]])
    B = np.array([[1,2,3],[4,5,6],[7,8,9]])
    C = np.array([1])
    D = np.array([])
    bdmat = BDMatrix((A,B,C))
    bdmat2 = BDMatrix((B,D,C,D,C,A,B,C))
    #print(bdmat2.full_mat())

