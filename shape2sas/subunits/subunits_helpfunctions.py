from ..helpfunctions import InputError

def check_dimension(name,dimensions,n):
    """check if the number of input dimensions for the subunit is correct, else raise an InputError"""
    len_dim = len(dimensions)
    if len_dim != n:
            dim = ' dimension ' if n == 1 else ' dimensions '
            were = ' was ' if len_dim == 1 else ' were '
            raise InputError("subunit " + name + " needs " + str(n) + dim + "(provided after --dimensions or -d), but " + str(len_dim) + were + "given: " + str(dimensions))

