import sklearn
import numpy as np
import pandas
import random

def monotonic_lars(training_data):
    matrix_x = training_data[1]
    vector_y = training_data[0]
    N_tr = len(matrix_x)
    p = len(matrix_x[0])
    alphas, active, coef_path =  sklearn.linear_model.lars_path(matrix_x,vector_y)
    coef_path = coef_path.T
    output = np.zeros([p,p])
    for i in range(p):
        output[i] = coef_path[i+1]
    output = output.T
    return output

def setup_data():
    #READ data
    dat_file = r"C:\Austin\II Maths\CATAM\10.15 Variable Selection and the Bias-Variance Tradeoff\test\prostatedata.dat"
    data = pandas.read_csv(dat_file, sep=" ", header=None)

    #Add 4 extra covariate columns
    np.random.seed(1)
    col1 = np.random.normal(0,1,97)
    col2 = np.random.normal(0,1,97)
    col3 = np.random.normal(0,1,97)
    col4 = np.random.normal(0,1,97)
    data[9],data[10],data[11],data[12] = col1, col2, col3, col4

    #Now get in training_data format
    y_vector = np.array(data[0])
    x_matrix = data.T[1:]
    x_matrix = np.array(x_matrix.T)
    full_data = [y_vector, x_matrix]

    #Subsection into training and testing data
    np.random.seed(11)
    y_vector_training = [0]*70
    y_vector_testing = [0]*27
    x_matrix_training = [0]*70
    x_matrix_testing = [0]*27
    rows = [0]*97
    for i in range(97):
        rows[i] = i
    rand_rows = np.random.permutation(rows)
    for i in range(70):
        index = rand_rows[i]
        y_vector_training[i] = y_vector[index]
        x_matrix_training[i] = x_matrix[index]
    for i in range(27):
        index = rand_rows[70+i]
        y_vector_testing[i] = y_vector[index]
        x_matrix_testing[i] = x_matrix[index]

    training_data = [np.array(y_vector_training), np.array(x_matrix_training)]
    testing_data = [np.array(y_vector_testing), np.array(x_matrix_testing)]
    return training_data, testing_data

def bestsubset_Q8(training_dataset):
    matrix = training_dataset[1]
    vector_y = training_dataset[0]

    p = len(matrix[0])
    true_beta = np.array([-0.5,0.45,-0.4,0.35,-0.3,0.25,-0.2,0.15,-0.1,0.05])
    N_tr = len(matrix)
    N_te = 1000
    M_full = []
    for i in range(1,p+1):
        M_full = M_full + [i]
    M_powerset = list(powerset(M_full))
    #print(M_full)
    #print(M_powerset)

    #set up RSS arrays
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

        #Calculate reduced_beta LS
        left_out = [0] * p
        reduced_matrix_x = np.zeros([N_tr, jay])
        for repeat1 in range(N_tr):
            for repeat2 in range(jay):
                index = M_j_array[repeat2]
                reduced_matrix_x[repeat1][repeat2] = matrix[repeat1][index-1]
                left_out[index-1] = 1
        product1 = np.linalg.inv(np.dot(reduced_matrix_x.T, reduced_matrix_x))
        product2 = np.dot(product1, reduced_matrix_x.T)
        reduced_beta_initial = np.dot(product2, vector_y)
        #fix size
        count=0
        for i in range(p):
            if left_out[i]==0:
                reduced_beta[i]=0
            else:
                reduced_beta[i]=reduced_beta_initial[count]
                count=count+1
        #print('rb = ' + str(reduced_beta))

        #RSS values
        vector_x_beta_tr = np.dot(matrix, reduced_beta)
        RSS_vector = np.subtract(vector_y, vector_x_beta_tr)
        RSS_scalar = (np.dot(RSS_vector, RSS_vector)) / N_tr
        if RSS_scalar < RSS_training[jay-1][0]:
            RSS_training[jay-1][0] = RSS_scalar
            RSS_training[jay-1][1] = M_j_array
            RSS_training[jay-1][2] = reduced_beta

    #print(RSS_test)
    #print(RSS_training)

    #Create matrix B
    B=np.zeros([p,p])
    for i in range(p):
        B[i] = RSS_training[i][2]

    #print(B)
    return(B.T)

def greedysubset_Q8(training_dataset):
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

            # RSS values
            vector_x_beta_tr = np.dot(matrix, reduced_beta)
            RSS_vector = np.subtract(vector_y, vector_x_beta_tr)
            RSS_scalar = (np.dot(RSS_vector, RSS_vector)) / N_tr
            if RSS_scalar < RSS_training[jay - 1][0]:
                RSS_training[jay - 1][0] = RSS_scalar
                RSS_training[jay - 1][1] = j
                RSS_training[jay - 1][2] = reduced_beta

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


