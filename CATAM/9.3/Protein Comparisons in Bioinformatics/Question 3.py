import numpy as np

Protein_A='MGLSDGEWQLVLKVWGKVEGDLPGHGQEVLIRLFKTHPETLEKFDKFKGLKTEDEMKASADLKKHGGTVLTALGNILKKKGQHEAELKPLAQSHATKHKISIKFLEYISEAIIHVLQSKHSADFGADAQAAMGKALELFRNDMAAKYKEFGFQG'
Protein_B='MADFDAVLKCWGPVEADYTTMGGLVLTRLFKEHPETQKLFPKFAGIAQADIAGNAAISAHGATVLKKLGELLKAKGSHAAILKPLANSHATKHKIPINNFKLISEVLVKVMHEKAGLDAGGQTALRNVMGIIIADLEANYKELGFSG'
S=Protein_A
T=Protein_B
m=len(S)
n=len(T)
D_matrix = np.zeros([m+1,n+1],dtype=int)
D2_matrix = np.zeros([m+1,n+1],dtype=str)

def D2(i,j,S,T):
    if i==0 and j==0:
        D_matrix[i][j] = 0
        D2_matrix[i][j] = ''
    elif i==0 and j!=0:
        D_matrix[i][j] = j
        D2_matrix[i][j] = 'I'
    elif j==0 and i!=0:
        D_matrix[i][j] = i
        D2_matrix[i][j] = 'D'
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
        next_edit = ''
        #Separate 'if' loops to add all of the possible edit choices.
        if answer == a:
            next_edit = 'I'
        if answer == b:
            next_edit += 'D'
        if answer == c and delta == 1:
            next_edit += 'M'
        elif answer == c and delta == 0:
            next_edit += 'R'
        D_matrix[i][j] = answer
        D2_matrix[i][j] = next_edit
    return

def edit_dist2(m,n):
    b=0
    while b<=n:
        a=0
        while a<=m:
            D2(a,b,S,T)
            a=a+1
        b=b+1
    return D_matrix[m][n]

def transcript(D2_matrix):
    edit_transcript = ''
    i=m
    j=n
    while i!=0 or j!=0:
        #if more than one edit possible choose the first.
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
    #reverse order
    edit_transcript = edit_transcript[::-1]
    return edit_transcript[0:50]