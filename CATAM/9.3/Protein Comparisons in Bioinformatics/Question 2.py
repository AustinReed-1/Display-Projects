import numpy as np

S = 'shesells'
T = 'seashells'
m = len(S)
n = len(T)

#very inefficient version
def D(i,j, S, T):
    if i>len(S) or i<0 or j>len(T) or j<0:
        return 'Index out of range.'
    if i==0:
        return j
    elif j==0:
        return i
    else:
        S_i = S[i-1]
        T_j = T[j-1]
        if S_i==T_j:
            delta = 1
        else:
            delta = 0
        answer = min(D(i,j-1,S,T)+1,D(i-1,j,S,T)+1,D(i-1,j-1,S,T)+1-delta)
        return answer

#much better
def D(i,j,S,T):
    if i==0:
        answer = j
    elif j==0:
        answer = i
    else:
        S_i = S[i-1]
        T_j = T[j-1]
        if S_i==T_j:
            delta = 1
        else:
            delta = 0
        a = D_matrix[i][j-1]+1
        b = D_matrix[i-1][j]+1
        c = D_matrix[i-1][j-1]+1-delta
        answer = min(a, b, c)
    return answer

def edit_dist(m,n):
    if m>len(S) or m<0 or n>len(T) or n<0:
        return 'Index out of range.'
    D_matrix = np.zeros([m + 1, n + 1], dtype=int)
    b=0
    while b<=n:
        a=0
        while a<=m:
            D_matrix[a][b] = D(a,b,S,T)
            a=a+1
        b=b+1
    return D_matrix[m][n]