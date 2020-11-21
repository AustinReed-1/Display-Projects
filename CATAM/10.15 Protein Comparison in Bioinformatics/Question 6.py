import numpy as np
import scipy.stats
from itertools import chain, combinations

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

def crossval(training_dataset, function):
    N_tr = len(training_dataset[1])
    p = len(training_dataset[1][0])
    matrix = training_dataset[1]
    vector_y = training_dataset[0]

    # form the estimator matrices
    B_k = [0]*10
    test_sets = [0]*10
    for k in range(1,11):
        #calculate random permuation for each k
        pi = np.random.permutation(30)

        lower_bound = N_tr * (k-1) / 10
        upper_bound = N_tr * k / 10
        i=1
        n=[]
        not_n=[]
        while i <= N_tr:
            if lower_bound < i <= upper_bound:
                n = n + [i-1]
            else:
                not_n = not_n +[i-1]
            i = i+1
        print(n)
        print(not_n)
        pi_n = pi[n]
        not_pi_n = pi[not_n]
        print(pi_n)
        print(not_pi_n)

        vector_y_n = vector_y[pi_n]
        vector_y_not_n = vector_y[not_pi_n]
        matrix_x_n = matrix[pi_n]
        matrix_x_not_n = matrix[not_pi_n]


        training_k = [vector_y_n, matrix_x_n]
        training_not_k = [vector_y_not_n, matrix_x_not_n]

        #function must take training and testing datasets as the input
        single_matrix = function(training_not_k)
        B_k[k-1] = single_matrix
        test_sets[k-1] = training_k

    RSS_min = 0
    for j in range(p):
        RSS_total = 0
        for k in range (1,11):
            N_te = len(test_sets[k-1][0])
            vector_y_true = test_sets[k-1][0]
            matrix_x_test = test_sets[k-1][1]
            B = B_k[k-1]
            B = B.T
            vector_xbeta_test = []
            for i in range(N_te):
                scalar_xbeta_test = np.dot(B[j], matrix_x_test[i])
                vector_xbeta_test = np.append(vector_xbeta_test, scalar_xbeta_test)

            RSS_vector = vector_y_true - vector_xbeta_test
            RSS = np.dot(RSS_vector, RSS_vector) / N_te
            RSS_total = RSS_total + RSS

        RSS_total = RSS_total / 10

        if RSS_min == 0:
            RSS_min = RSS_total
            count = 0
        elif RSS_total < RSS_min:
            RSS_min = RSS_total
            count = j

    B_needed = function(training_dataset)

    B_needed = B_needed.T
    return(B_needed[count])