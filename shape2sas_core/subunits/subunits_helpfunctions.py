def printt(s): 
    """ print and write to log file"""
    print(s)
    with open('shape2sas.log','a') as f:
        f.write('%s\n' %s)

def check_dimension(name,dimensions,n):
    """check if the number of input dimensions for the subunit is correct, else return error message"""
    len_dim = len(dimensions)
    if len_dim != n:
            dim = ' dimension ' if n == 1 else ' dimensions '
            were = ' was ' if len_dim == 1 else ' were '
            printt("\nERROR: subunit " + name + " needs " + str(n) + dim + "(provided after --dimensions or -d), but " + str(len_dim) + were + "given: " + str(dimensions) + "\n")
            exit()

