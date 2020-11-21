from Bio.SubsMat import MatrixInfo as matrixFile
blosum = matrixFile.blosum62
import numpy as np

def form_matrix():
    Protein_C = 'MTSDCSSTHCSPESCGTASGCAPASSCSVETACLPGTCATSRCQTPSFLSRSRGLTGCLLPCYFTGSCNSPCLVGNCAWCEDGVFTSNEKETMQFLNDRLASYLEKVRSLEETNAELESRIQEQCEQDIPMVCPDYQRYFNTIEDLQQKILCTKAENSRLAVQLDNCKLATDDFKSKYESELSLRQLLEADISSLHGILEELTLCKSDLEAHVESLKEDLLCLKKNHEEEVNLLREQLGDRLSVELDTAPTLDLNRVLDEMRCQCETVLANNRREAEEWLAVQTEELNQQQLSSAEQLQGCQMEILELKRTASALEIELQAQQSLTESLECTVAETEAQYSSQLAQIQCLIDNLENQLAEIRCDLERQNQEYQVLLDVKARLEGEINTYWGLLDSEDSRLSCSPCSTTCTSSNTCEPCSAYVICTVENCCL'
    Protein_D = 'MPYNFCLPSLSCRTSCSSRPCVPPSCHSCTLPGACNIPANVSNCNWFCEGSFNGSEKETMQFLNDRLASYLEKVRQLERDNAELENLIRERSQQQEPLLCPSYQSYFKTIEELQQKILCTKSENARLVVQIDNAKLAADDFRTKYQTELSLRQLVESDINGLRRILDELTLCKSDLEAQVESLKEELLCLKSNHEQEVNTLRCQLGDRLNVEVDAAPTVDLNRVLNETRSQYEALVETNRREVEQWFTTQTEELNKQVVSSSEQLQSYQAEIIELRRTVNALEIELQAQHNLRDSLENTLTESEARYSSQLSQVQSLITNVESQLAEIRSDLERQNQEYQVLLDVRARLECEINTYRSLLESEDCNLPSNPCATTNACSKPIGPCLSNPCTSCVPPAPCTPCAPRPRCGPCNSFVR'
    S = Protein_C
    T = Protein_D
    m = len(S)
    n = len(T)
    V_sfx_matrix = np.zeros([m + 1, n + 1], dtype=int)
    V_sfx_matrix2 = np.zeros([m + 1, n + 1], dtype='U4')
    b=0
    while b<n+1:
        a=0
        while a<m+1:
            if a == 0 or b == 0:
                V_sfx_matrix[a][b] = 0
                V_sfx_matrix2[a][b] = ''
            else:
                S_a = S[a-1]
                T_b = T[b-1]
                if blosum.get((S_a, T_b)) == None:
                    s = blosum.get((T_b, S_a))
                else:
                    s = blosum.get((S_a, T_b))
                Rep_match = V_sfx_matrix[a-1][b-1] + s
                Del = V_sfx_matrix[a-1][b]-2
                Ins = V_sfx_matrix[a][b-1]-2

                V_a_b = max(0, Rep_match, Del, Ins)
                V_sfx_matrix[a,b] = V_a_b

                if V_a_b == 0:
                    V_sfx_matrix2[a][b] = 'B'
                elif V_a_b == Rep_match and S_a == T_b:
                    V_sfx_matrix2[a][b] = 'M'
                elif V_a_b == Rep_match and S_a != T_b:
                    V_sfx_matrix2[a][b] = 'R'
                elif V_a_b == Del:
                    V_sfx_matrix2[a][b] = 'D'
                elif V_a_b == Ins:
                    V_sfx_matrix2[a][b] = 'I'
            a=a+1
        b=b+1
    return [V_sfx_matrix, V_sfx_matrix2]

def v_sub():
    matricies = form_matrix()
    max = np.max(matricies[0])
    idx = np.where(matricies[0] == max)
    start_i = idx[0][0]
    start_j = idx[1][0]

    i=start_i
    j=start_j
    edit = ''
    edit_transcript = ''
    while edit != "B":
        edit_transcript += edit
        edit = matricies[1][i][j]
        if edit == 'I':
            j=j-1
        elif edit == 'D':
            i=i-1
        elif edit == 'M' or edit == 'R':
            i=i-1
            j=j-1
    if edit_transcript[-1] == 'I':
        end_i=i
        end_j=j+1
    elif edit_transcript[-1] == 'D':
        end_i=i+1
        end_j=j
    else:
        end_i=i+1
        end_j=j+1

    output = 'v_sub = ' +str(max) + ', substrings: S[' + str(end_i) + ',' + str(start_i) + '], T[' + str(end_j) + ',' + str(start_j) +']'
    edit_transcript = edit_transcript[::-1]

    print(output)
    print(len(edit_transcript))
    print(edit_transcript[0:50])
