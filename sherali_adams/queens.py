import sherali_adams as sa
from itertools import chain
import numpy as np

from cvxopt import matrix, solvers, glpk
from cvxopt.glpk import lp, ilp

def linearize(n,a,b):
    return n*a + b

def construct(n):
    acc =[]
    b = []
    #queens
    avec = np.ones(n*n)
    b.append(n)
    acc.append(avec)
    avec = -np.ones(n*n)
    b.append(-n)
    acc.append(avec)

    #rows
    for i in range(n):
        avec = np.zeros(n*n)
        for j in range(n):
            avec[linearize(n,i,j)] = 1
        b.append(1)
        acc.append(avec)

    #cols
    for j in range(n):
        avec = np.zeros(n*n)
        for i in range(n):
            avec[linearize(n,i,j)] = 1
        b.append(1)
        acc.append(avec)

    #neg diags
    for k in range(1, 2*n-2):
        avec = np.zeros(n*n)
        for r in range(n):
            for s in range(n):
                if r + s == k:
                    avec[linearize(n,r,s)] = 1
        b.append(1)
        acc.append(avec)

    #pos diags
    for k in range(2-n, n-1):
        avec = np.zeros(n*n)
        for r in range(n):
            for s in range(n):
                if r - s == k:
                    avec[linearize(n,r,s)] = 1
        b.append(1)
        acc.append(avec)

    # x <= 1
    for i in range(n*n):
        avec = np.zeros(n*n)
        avec[i] = 1
        acc.append(avec)
        b.append(1)

    # x >= 0
    for i in range(n*n):
        avec = np.zeros(n*n)
        avec[i] = -1
        acc.append(avec)
        b.append(0)

    A = np.asarray(list(chain(*acc)))
    return (np.reshape(A,(len(A)//(n*n), n*n)), b)

def count_non_zero ( v ) :

    t = 0

    for x in v:
        if x != 0:
            t += 1

    return t

def fmt ( sol, n ) :
    v = sol[1][:n]
    return v, sum(v), count_non_zero(v)

def solve(A,b):
    glpk.options['meth'] = 'GLP_PRIMAL'
    (row,col) = A.shape
    n = col
    c = np.ones(col)
    opt = lp(matrix(c),matrix(A),matrix(b,tc='d'))
    print(fmt(opt, n))
    #print(lp[1], sum(lp[1]))
    #(st,ip) = ilp(matrix(c),matrix(A),matrix(b,tc='d'), I = set(range(n)))
    #print(ip, sum(ip))
    (SA,sb) = sa.run_SA(1,n,A,b)
    optsa = lp(np.zeros(SA.shape[1]),matrix(SA),matrix(sb,tc='d'))
    return fmt(optsa, n)

def run_queens(n):
    (A,b) = construct(n)
    print(solve(A,b))

if __name__ == '__main__':
    import sys
    n = int(sys.argv[1])
    run_queens(n)
