#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# build obj gradient and constraint Jacobian
# from a gjh file written by the ASL gjh 'solver'
# gjh solver may be called using pyomo, use keepfiles = True to write the file

# Use symbolic_solver_labels=True option in pyomo allong with keepfiles = True
# to write the .row file and .col file to get variable mappings


def readgjh(fname):
    f = open(fname,"r")

    data = "dummy_str"
    while data != "param g :=\n":
        data = f.readline()

    data = f.readline()
    g = []
    while data[0] != ';':
        # gradient entry (sparse)
        entry = [int(data.split()[0]) - 1, float(data.split()[1])] # subtract 1 to index from 0
        g.append(entry) # build obj gradient in sparse format
        data = f.readline()

    while data != "param J :=\n":
        data = f.readline()

    data = f.readline()
    J = []
    while data[0] != ';':
        if data[0] == '[':
            # Jacobian row index
            #
            # The following replaces int(filter(str.isdigit,data)),
            # which only works in 2.x
            data_as_int = int(''.join(filter(str.isdigit, data)))
            row = data_as_int - 1  # subtract 1 to index from 0
            data = f.readline()

        entry = [row, int(data.split()[0]) - 1, float(data.split()[1])]  # subtract 1 to index from 0
        J.append(entry) # Jacobian entries, sparse format
        data = f.readline()

    while data != "param H :=\n":
        data = f.readline()

    data = f.readline()
    H = []
    while data[0] != ';':
        if data[0] == '[':
            # Hessian row index
            data_as_int = int(''.join(filter(str.isdigit, data)))
            row = data_as_int - 1  # subtract 1 to index from 0
            data = f.readline()

        entry = [row, int(data.split()[0]) - 1, float(data.split()[1])]  # subtract 1 to index from 0
        H.append(entry) # Hessian entries
        data = f.readline()

    f.close()

    with open(fname[:-3] + 'col', 'r') as f:
        data = f.read()
        varlist = data.split()

    with open(fname[:-3] + 'row', 'r') as f:
        data = f.read()
        conlist = data.split()

    return g,J,H,varlist,conlist
