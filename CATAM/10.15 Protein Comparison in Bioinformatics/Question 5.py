import numpy as np
import scipy.stats

def form_list(N_tr, p):
    #forms (N_tr by p) matrix training set like Q2
    vector_x = np.random.normal(0,1,p)
    matrix_x = np.array([vector_x])
    true_beta = np.array([-0.5,0.45,-0.4,0.35,-0.3,0.25,-0.2,0.15,-0.1,0.05])
    i=2
    while i<=N_tr:
        vector_x = np.random.normal(0, 1, p)
        matrix_x = np.append(matrix_x, [vector_x], axis=0)
        i=i+1

    # Complete training
    vector_y = np.array([])
    for i in range(N_tr):
        vector_y = np.append(vector_y, np.dot(matrix_x[i], true_beta) + np.random.normal(0, 1, 1))

    training_dataset = [vector_y, matrix_x]
    return training_dataset

def greedysubset2(training_dataset):
    matrix = training_dataset[1]
    vector_y = training_dataset[0]

    p = len(matrix[0])
    true_beta = np.array([-0.5,0.45,-0.4,0.35,-0.3,0.25,-0.2,0.15,-0.1,0.05])
    N_tr = len(matrix)
    N_te = 1000
    M_full = []
    M_not = []
    for i in range(p):
        M_not = M_not + [i+1]

    # set up RSS arrays
    RSS_test = []
    for i in range(p):
        RSS_test = RSS_test + [[100, 0, [0] * 10]]

    #start subset testing
    m=0
    tf = True
    while m < p and tf:
        for j in M_not:
            M_try = M_full + [j]

            jay = len(M_try)
            M_j_array = []
            reduced_beta = [0] * p
            for a in M_try:
                M_j_array = M_j_array + [a]

            # Calculate reduced_beta LS
            left_out = [0] * p
            reduced_matrix_x = np.zeros([N_tr, jay])
            for repeat1 in range(N_tr):
                for repeat2 in range(jay):
                    index = M_j_array[repeat2]
                    reduced_matrix_x[repeat1][repeat2] = matrix[repeat1][index-1]
                    left_out[index - 1] = 1
            product1 = np.linalg.inv(np.dot(reduced_matrix_x.T, reduced_matrix_x))
            product2 = np.dot(product1, reduced_matrix_x.T)
            reduced_beta_initial = np.dot(product2, vector_y)
            # fix size
            count = 0
            for i in range(p):
                if left_out[i] == 0:
                    reduced_beta[i] = 0
                else:
                    reduced_beta[i] = reduced_beta_initial[count]
                    count = count + 1
            # print(reduced_beta)

            # Complete testing
            vector_x_test = np.random.normal(0, 1, p)
            rand = np.random.normal(0, 1, 1)
            scalar_y_true = np.dot(vector_x_test, true_beta) + rand
            scalar_x_beta_test = np.dot(vector_x_test, reduced_beta)
            vector_y_true = np.array([scalar_y_true])
            vector_x_beta_test = np.array([scalar_x_beta_test])
            k = 2
            while k <= N_te:
                vector_x_test = np.random.normal(0, 1, p)
                rand = np.random.normal(0, 1, 1)
                scalar_y_true = np.dot(vector_x_test, true_beta) + rand
                scalar_x_beta_test = np.dot(vector_x_test, reduced_beta)
                vector_y_true = np.append(vector_y_true, scalar_y_true)
                vector_x_beta_test = np.append(vector_x_beta_test, scalar_x_beta_test)
                k = k + 1

            # Calculate and update RSS_test vectors
            RSS_vector = np.subtract(vector_y_true, vector_x_beta_test)
            RSS_scalar = (np.dot(RSS_vector, RSS_vector)) / N_te
            if RSS_scalar < RSS_test[jay - 1][0]:
                RSS_test[jay - 1][0] = RSS_scalar
                RSS_test[jay - 1][1] = j
                RSS_test[jay - 1][2] = reduced_beta
        length = len(M_not)
        for k in range(length):
            if M_not[k] == RSS_test[jay-1][1]:
                M_full.append(M_not[k])
                M_not.remove(M_not[k])
                break

        #F-testing and whether to repeat again
        if jay > 1:
            print(RSS_test[jay-2][0])
            print(RSS_test[jay-1][0])
            F = (RSS_test[jay-2][0] - RSS_test[jay-1][0]) / (RSS_test[jay-1][0] / (N_tr*p - jay))
            p_value = 1 - scipy.stats.f.cdf(F, 1, N_tr*p - jay)
            print('p' +str(p_value))
            if p_value > 0.05 or F < 0:
                tf = False
                print('end' + str(jay))

        m=m+1

    print(RSS_test)

    # Create matrix B
    B = np.zeros([p, p], dtype=float)
    for i in range(p):
        B[i] = RSS_test[i][2]

    return(B.T)
