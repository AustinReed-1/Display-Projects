import random
import numpy as np

u=-3
p=0.5
num = 100
tests = 10000

def D_Q8(i,j,S,T, D_matrix, row_max, col_max):
    if i==0 and j==0:
        D_matrix[i][j] = 0
        row_max[0] = [0,0]
        col_max[0] = [0,0]
    elif i==0 and j!=0:
        D_matrix[i][j] = u
        col_max[j] = [max(col_max[j][0], u), i]
    elif j==0 and i!=0:
        D_matrix[i][j] = u
        row_max[i] = [max(row_max[j][0], u), j]
    else:
        S_i = S[i-1]
        T_j = T[j-1]
        if S_i==T_j:
            delta=1
        else:
            delta=-1
        p = row_max[i][0] + u
        q = col_max[j][0] + u
        r = D_matrix[i-1][j-1] + delta
        answer = max(p, q, r)
        new_row = max(row_max[i][0],answer)
        new_col = max(col_max[j][0],answer)
        if row_max[i][0] < answer:
            row_max[i] = [new_row, j]
        if col_max[j][0] < answer:
            col_max[j] = [new_col, i]
        D_matrix[i][j] = answer
    return

def edit_score_Q8(m,n,S,T):
    b=0
    D_matrix = np.zeros([n + 1, m + 1], dtype=int)
    row_max = np.full([m + 1, 2], -1000, dtype=int)
    col_max = np.full([n + 1, 2], -1000, dtype=int)
    while b<=m:
        a=0
        while a<=n:
            D_Q8(a,b,S,T,D_matrix, row_max, col_max)
            a=a+1
        b=b+1
    return D_matrix[m][n]

def estimate(num, tests):
    total_est = 0
    i=0
    while i < tests:
        j=0
        S=''
        T=''
        while j<num:
            #a=0,b=1
            amino_acid1 = random.randint(0, 1)
            amino_acid2 = random.randint(0, 1)
            S = S + str(amino_acid1)
            T = T + str(amino_acid2)
            j=j+1
        est0 = edit_score_Q8(num,num,S,T)
        est=est0/num
        total_est = total_est + est
        i=i+1
    avg_est = total_est / tests
    return avg_est