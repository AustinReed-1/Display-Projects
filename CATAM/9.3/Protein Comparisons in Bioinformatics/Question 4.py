from Bio.SubsMat import MatrixInfo as matrixFile
blosum = matrixFile.blosum62
import numpy as np

Protein_A='MGLSDGEWQLVLKVWGKVEGDLPGHGQEVLIRLFKTHPETLEKFDKFKGLKTEDEMKASADLKKHGGTVLTALGNILKKKGQHEAELKPLAQSHATKHKISIKFLEYISEAIIHVLQSKHSADFGADAQAAMGKALELFRNDMAAKYKEFGFQG'
Protein_B='MADFDAVLKCWGPVEADYTTMGGLVLTRLFKEHPETQKLFPKFAGIAQADIAGNAAISAHGATVLKKLGELLKAKGSHAAILKPLANSHATKHKIPINNFKLISEVLVKVMHEKAGLDAGGQTALRNVMGIIIADLEANYKELGFSG'
S=Protein_A
T=Protein_B
m=len(S)
n=len(T)
D_matrix = np.zeros([m+1,n+1],dtype=int)
D2_matrix = np.zeros([m+1,n+1],dtype=str)

def D_Q4(i,j,S,T):
    if i==0 and j==0:
        D_matrix[i][j] = 0
        D2_matrix[i][j] = ''
    elif i==0 and j!=0:
        D_matrix[i][j] = -j*8
        D2_matrix[i][j] = 'I'
    elif j==0 and i!=0:
        D_matrix[i][j] = -i*8
        D2_matrix[i][j] = 'D'
    else:
        S_i = S[i-1]
        T_j = T[j-1]
        if blosum.get((S_i,T_j)) == None:
            s = blosum.get((T_j,S_i))
        else:
            s = blosum.get((S_i,T_j))
        a = D_matrix[i][j-1]-8
        b = D_matrix[i-1][j]-8
        c = D_matrix[i-1][j-1] + int(s)
        answer = max(a, b, c)
        next_edit = ''
        if answer == a:
            next_edit = 'I'
        if answer == b:
            next_edit += 'D'
        if answer == c and S_i==T_j:
            next_edit += 'M'
        elif answer == c and S_i!=T_j:
            next_edit += 'R'
        D_matrix[i][j] = answer
        D2_matrix[i][j] = next_edit
    return

def edit_score_Q4(m,n):
    b=0
    while b<=n:
        a=0
        while a<=m:
            D_Q4(a,b,S,T)
            a=a+1
        b=b+1
    return D_matrix[m][n]

def transcript_Q4(D2_matrix):
    edit_transcript = ''
    i=m
    j=n
    while i!=0 or j!=0:
        edit = D2_matrix[i][j][0]
        edit_transcript += edit
        if edit == 'I':
            j=j-1
        elif edit == 'D':
            i=i-1
        else:
            i=i-1
            j=j-1
    print(len(edit_transcript))
    edit_transcript = edit_transcript[::-1]
    return edit_transcript[0:50]