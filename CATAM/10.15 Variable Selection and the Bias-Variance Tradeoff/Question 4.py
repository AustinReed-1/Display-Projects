import numpy as np

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

def greedysubset(training_dataset):
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
    RSS_training = []
    for i in range(p):
        RSS_test = RSS_test + [[100, '', [0] * p]]
        RSS_training = RSS_training + [[100, '', [0] * p]]

    #start subset testing
    for i in range(p):
        for j in M_not:
            M_try = M_full + [j]
            M_try.sort()

            jay = len(M_try)
            reduced_beta = [0] * p

            # Calculate reduced_beta LS
            left_out = [0] * p
            reduced_matrix_x = np.zeros([N_tr, jay])
            for repeat1 in range(N_tr):
                for repeat2 in range(jay):
                    index = M_try[repeat2]
                    reduced_matrix_x[repeat1][repeat2] = matrix[repeat1][index - 1]
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

            # Complete testing
            np.random.seed(10)
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


            # RSS values
            vector_x_beta_tr = np.dot(matrix, reduced_beta)
            RSS_vector = np.subtract(vector_y, vector_x_beta_tr)
            RSS_scalar = (np.dot(RSS_vector, RSS_vector)) / N_tr
            RSS_vector2 = np.subtract(vector_y_true, vector_x_beta_test)
            RSS_scalar2 = (np.dot(RSS_vector2, RSS_vector2)) / N_te
            if RSS_scalar < RSS_training[jay - 1][0]:
                RSS_training[jay - 1][0] = RSS_scalar
                RSS_training[jay - 1][1] = j
                RSS_training[jay - 1][2] = reduced_beta
                RSS_test[jay - 1][0] = RSS_scalar2
                RSS_test[jay - 1][1] = M_try
                RSS_test[jay - 1][2] = reduced_beta


        length = len(M_not)
        k=0
        while k < length:
            if M_not[k] == RSS_training[jay-1][1]:
                M_full.append(M_not[k])
                M_not.remove(M_not[k])
                k=length
            k=k+1

    #print(RSS_test)
    #print(RSS_training)

    # Create matrix B
    B = np.zeros([p, p], dtype=float)
    for i in range(p):
        B[i] = RSS_training[i][2]

    # print(B)
    return(B.T)
