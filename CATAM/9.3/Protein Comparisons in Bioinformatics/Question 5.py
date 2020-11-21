from Bio.SubsMat import MatrixInfo as matrixFile
blosum = matrixFile.blosum62
import numpy as np

Protein_A='MGLSDGEWQLVLKVWGKVEGDLPGHGQEVLIRLFKTHPETLEKFDKFKGLKTEDEMKASADLKKHGGTVLTALGNILKKKGQHEAELKPLAQSHATKHKISIKFLEYISEAIIHVLQSKHSADFGADAQAAMGKALELFRNDMAAKYKEFGFQG'
Protein_B='MADFDAVLKCWGPVEADYTTMGGLVLTRLFKEHPETQKLFPKFAGIAQADIAGNAAISAHGATVLKKLGELLKAKGSHAAILKPLANSHATKHKIPINNFKLISEVLVKVMHEKAGLDAGGQTALRNVMGIIIADLEANYKELGFSG'
Protein_C='MTSDCSSTHCSPESCGTASGCAPASSCSVETACLPGTCATSRCQTPSFLSRSRGLTGCLLPCYFTGSCNSPCLVGNCAWCEDGVFTSNEKETMQFLNDRLASYLEKVRSLEETNAELESRIQEQCEQDIPMVCPDYQRYFNTIEDLQQKILCTKAENSRLAVQLDNCKLATDDFKSKYESELSLRQLLEADISSLHGILEELTLCKSDLEAHVESLKEDLLCLKKNHEEEVNLLREQLGDRLSVELDTAPTLDLNRVLDEMRCQCETVLANNRREAEEWLAVQTEELNQQQLSSAEQLQGCQMEILELKRTASALEIELQAQQSLTESLECTVAETEAQYSSQLAQIQCLIDNLENQLAEIRCDLERQNQEYQVLLDVKARLEGEINTYWGLLDSEDSRLSCSPCSTTCTSSNTCEPCSAYVICTVENCCL'
Protein_D='MPYNFCLPSLSCRTSCSSRPCVPPSCHSCTLPGACNIPANVSNCNWFCEGSFNGSEKETMQFLNDRLASYLEKVRQLERDNAELENLIRERSQQQEPLLCPSYQSYFKTIEELQQKILCTKSENARLVVQIDNAKLAADDFRTKYQTELSLRQLVESDINGLRRILDELTLCKSDLEAQVESLKEELLCLKSNHEQEVNTLRCQLGDRLNVEVDAAPTVDLNRVLNETRSQYEALVETNRREVEQWFTTQTEELNKQVVSSSEQLQSYQAEIIELRRTVNALEIELQAQHNLRDSLENTLTESEARYSSQLSQVQSLITNVESQLAEIRSDLERQNQEYQVLLDVRARLECEINTYRSLLESEDCNLPSNPCATTNACSKPIGPCLSNPCTSCVPPAPCTPCAPRPRCGPCNSFVR'
S=Protein_C
T=Protein_D
m=len(S)
n=len(T)
D_matrix = np.zeros([m+1,n+1],dtype=int)
D2_matrix = np.zeros([m+1,n+1],dtype='U4')
row_max=np.full([m+1,2], -1000, dtype=int)
col_max=np.full([n+1,2], -1000, dtype=int)

def D_Q5(i,j,S,T):
    if i==0 and j==0:
        D_matrix[i][j] = 0
        D2_matrix[i][j] = ''
        row_max[0] = [0,0]
        col_max[0] = [0,0]
    elif i==0 and j!=0:
        D_matrix[i][j] = -12
        D2_matrix[i][j] = 'I0'
        col_max[j] = [max(col_max[j][0], -12), i]
    elif j==0 and i!=0:
        D_matrix[i][j] = -12
        D2_matrix[i][j] = 'D0'
        row_max[i] = [max(row_max[j][0], -12), j]
    else:
        S_i = S[i-1]
        T_j = T[j-1]
        if blosum.get((S_i,T_j)) == None:
            s = blosum.get((T_j,S_i))
        else:
            s = blosum.get((S_i,T_j))
        p = row_max[i][0] - 12
        q = col_max[j][0] - 12
        r = D_matrix[i-1][j-1] + int(s)
        answer = max(p, q, r)
        new_row = max(row_max[i][0],answer)
        new_col = max(col_max[j][0],answer)
        if row_max[i][0] < answer:
            row_max[i] = [new_row, j]
        if col_max[j][0] < answer:
            col_max[j] = [new_col, i]
        next_edit = ''
        if answer == p:
            next_edit = 'I' + str(row_max[i][1])
        elif answer == q:
            next_edit = 'D' + str(col_max[j][1])
        elif answer == r and S_i==T_j:
            next_edit = 'M'
        elif answer == r and S_i!=T_j:
            next_edit = 'R'
        D_matrix[i][j] = answer
        D2_matrix[i][j] = next_edit
    return

def edit_score_Q5(m,n):
    b=0
    while b<=n:
        a=0
        while a<=m:
            D_Q5(a,b,S,T)
            a=a+1
        b=b+1
    return D_matrix[m][n]

def transcript_Q5(D2_matrix):
    edit_transcript = ''
    i=m
    j=n
    while i!=0 or j!=0:
        edit = D2_matrix[i][j][0]
        if edit == 'I':
            gap = j - int(D2_matrix[i][j][1:])
            edit = edit * gap
            j=j-gap
        elif edit == 'D':
            gap = i - int(D2_matrix[i][j][1:])
            edit = edit * gap
            i=i-gap
        else:
            i=i-1
            j=j-1
        edit_transcript += edit
    print(len(edit_transcript))
    edit_transcript = edit_transcript[::-1]
    return edit_transcript[0:50]