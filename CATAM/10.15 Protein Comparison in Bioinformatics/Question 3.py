import numpy as np
from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def form_matrix(N_tr, p):
    #forms (N_tr by p) matrix training set like Q2
    vector_x = np.random.normal(0,1,p)
    matrix_x = np.array([vector_x])
    i=2
    while i<=N_tr:
        vector_x = np.random.normal(0, 1, p)
        matrix_x = np.append(matrix_x, [vector_x], axis=0)
        i=i+1
    return matrix_x

def bestsubset(matrix):
    p = len(matrix[0])
    true_beta = np.array([-0.5,0.45,-0.4,0.35,-0.3,0.25,-0.2,0.15,-0.1,0.05])
    N_tr = len(matrix)
    N_te = 1000
    M_full = []
    for i in range(1,p+1):
        M_full = M_full + [i]
    M_powerset = list(powerset(M_full))
    print(M_full)
    print(M_powerset)

    #Complete training
    vector_y = np.array([])
    for i in range(N_tr):
        vector_y = np.append(vector_y, np.dot(matrix[i],true_beta) + np.random.normal(0,1,1))
    #Find estimate for beta by LS
    product1 = np.linalg.inv(np.dot(matrix.T, matrix))
    product2 = np.dot(product1, matrix.T)
    est_beta = np.dot(product2, vector_y)
    # print(true_beta)
    print(est_beta)

    argmin_array =  [0]*p
    #RSS_test = np.full([p,2],100,dtype=float)
    #RSS_training = np.full([p,2],100,dtype=float)
    RSS_test = []
    RSS_training = []
    for i in range(p):
        RSS_test = RSS_test + [[100,'', [0]*10]]
        RSS_training = RSS_training + [[100,'', [0]*10]]

    for M_j in M_powerset:
        jay = len(M_j)
        M_j_array = []
        reduced_beta = [0] * p
        for a in M_j:
            M_j_array = M_j_array + [a]
            reduced_beta[a-1] = est_beta[a-1]

        #Complete testing
        vector_x_test = np.random.normal(0, 1, p)
        rand = np.random.normal(0, 1, 1)
        scalar_y_true = np.dot(vector_x_test, true_beta) + rand
        scalar_x_beta_test = np.dot(vector_x_test, reduced_beta)
        scalar_y_test = scalar_x_beta_test + rand
        vector_y_true = np.array([scalar_y_true])
        vector_x_beta_test = np.array([scalar_x_beta_test])
        vector_y_test = np.array([scalar_y_test])
        vector_y_error = np.array([abs(scalar_y_true - scalar_y_test)])
        k = 2
        while k <= N_te:
            vector_x_test = np.random.normal(0, 1, p)
            rand = np.random.normal(0, 1, 1)
            scalar_y_true = np.dot(vector_x_test, true_beta) + rand
            scalar_x_beta_test = np.dot(vector_x_test, reduced_beta)
            scalar_y_test = scalar_x_beta_test + rand
            # matrix_x_test = np.append(matrix_x_test, [vector_x_test], axis=0)
            vector_y_true = np.append(vector_y_true, scalar_y_true)
            vector_x_beta_test = np.append(vector_x_beta_test, scalar_x_beta_test)
            vector_y_test = np.append(vector_y_test, scalar_y_test)
            vector_y_error = np.append(vector_y_error, abs(scalar_y_true - scalar_y_test))
            k = k + 1

        #Calculate and update RSS_test vectors
        RSS_vector = np.subtract(vector_y_true, vector_x_beta_test)
        RSS_scalar = (np.dot(RSS_vector, RSS_vector)) / N_te
        if RSS_scalar < RSS_test[jay-1][0]:
            RSS_test[jay-1][0] = RSS_scalar
            RSS_test[jay-1][1] = M_j_array
            RSS_test[jay - 1][2] = reduced_beta

        # Complete Training tests
        vector_x_training = matrix[0]
        scalar_y_true_training = vector_y[0]
        scalar_x_beta_training = np.dot(vector_x_training, reduced_beta)
        scalar_y_training = scalar_x_beta_training + (scalar_y_true_training - np.dot(vector_x_training, true_beta))
        vector_y_true_training = np.array([scalar_y_true_training])
        vector_x_beta_training = np.array([scalar_x_beta_training])
        vector_y_training = np.array([scalar_y_training])
        vector_y_error_training = np.array([abs(scalar_y_true_training - scalar_y_training)])
        k = 2
        while k <= N_tr:
            vector_x_training = matrix[k - 1]
            scalar_y_true_training = vector_y[k - 1]
            scalar_x_beta_training = np.dot(vector_x_training, reduced_beta)
            scalar_y_training = scalar_x_beta_training + (scalar_y_true_training - np.dot(vector_x_training, true_beta))
            vector_y_true_training = np.append(vector_y_true_training, scalar_y_true_training)
            vector_x_beta_training = np.append(vector_x_beta_training, scalar_x_beta_training)
            vector_y_training = np.append(vector_y_training, scalar_y_training)
            vector_y_error_training = np.append(vector_y_error_training, abs(scalar_y_true_training - scalar_y_training))
            k = k + 1

        # Calculate and update RSS_training vectors
        RSS_vector = np.subtract(vector_y_true_training, vector_x_beta_training)
        RSS_scalar = (np.dot(RSS_vector, RSS_vector)) / N_te

        if RSS_scalar < RSS_test[jay - 1][0]:
            RSS_training[jay - 1][0] = RSS_scalar
            RSS_training[jay - 1][1] = M_j_array
            RSS_training[jay - 1][2] = reduced_beta

    print(RSS_test)
    #print(RSS_training)

    #Create matrix B
    B=np.zeros([p,p],dtype=float)
    for i in range(p):
        B[i] = RSS_test[i][2]

    #print(B)
    print(B.T)
    return