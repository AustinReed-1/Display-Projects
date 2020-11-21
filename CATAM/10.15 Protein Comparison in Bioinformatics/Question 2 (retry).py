import numpy as np
import matplotlib.pyplot as plt

#Global variables
p=10
sigma=1
true_beta = np.array([-0.5,0.45,-0.4,0.35,-0.3,0.25,-0.2,0.15,-0.1,0.05])
N_tr = 30
N_te = 1000

def train_and_test(N_tr, N_te, p, sigma, true_beta):
    #Repeat 100 times
    RSS_test_total = np.array([0] * p)
    RSS_training_total = np.array([0] * p)
    repeat = 1
    while repeat <= 100:
        #training
        vector_x = np.random.normal(0, 1, p)
        scalar_y = np.dot(vector_x, true_beta) + np.random.normal(0,sigma,1)
        matrix_x = np.array([vector_x])
        vector_y = np.array([scalar_y])
        i=2
        while i <= N_tr:
            vector_x = np.random.normal(0,1,p)
            scalar_y = np.dot(vector_x, true_beta) + np.random.normal(0,sigma,1)
            matrix_x = np.append(matrix_x, [vector_x],axis=0)
            vector_y = np.append(vector_y, scalar_y)
            i=i+1
        #print(matrix_x)

        #now complete testing for each of M_j
        RSS_test = [0] * p
        RSS_training = [0] * p
        j=1
        while j <= p:
            #print('j=' +str(j))
            reduced_matrix_x = np.zeros([N_tr,j])
            for repeat1 in range(N_tr):
                for repeat2 in range(j):
                    reduced_matrix_x[repeat1][repeat2] = matrix_x[repeat1][repeat2]
            #print(reduced_matrix_x)
            #print(len(reduced_matrix_x))

            #Calculate reduced_beta by LS
            product1 = np.linalg.inv(np.dot(reduced_matrix_x.T, reduced_matrix_x))
            product2 = np.dot(product1, reduced_matrix_x.T)
            reduced_beta = np.dot(product2, vector_y)
            #fix size
            while len(reduced_beta) < p:
                reduced_beta = np.append(reduced_beta, 0)

            #now do N_te tests
            vector_x_test = np.random.normal(0, sigma, p)
            rand = np.random.normal(0, sigma, 1)
            scalar_y_true = np.dot(vector_x_test, true_beta) + rand
            scalar_x_beta_test = np.dot(vector_x_test, reduced_beta)
            vector_y_true = np.array([scalar_y_true])
            vector_x_beta_test = np.array([scalar_x_beta_test])
            k = 2
            while k <= N_te:
                vector_x_test = np.random.normal(0, 1, p)
                rand = np.random.normal(0, sigma, 1)
                scalar_y_true = np.dot(vector_x_test, true_beta) + rand
                scalar_x_beta_test = np.dot(vector_x_test, reduced_beta)
                vector_y_true = np.append(vector_y_true, scalar_y_true)
                vector_x_beta_test = np.append(vector_x_beta_test, scalar_x_beta_test)
                k = k + 1

            #Calc RSS
            RSS_vector = np.subtract(vector_y_true, vector_x_beta_test)
            RSS_test[j-1] = (np.dot(RSS_vector, RSS_vector))/ N_te
            #print(RSS_test[j-1])

            #Now run the N_tr as tests
            vector_x_training = matrix_x[0]
            scalar_y_true_training = vector_y[0]
            scalar_x_beta_training = np.dot(vector_x_training, reduced_beta)
            scalar_y_training = scalar_x_beta_training + (scalar_y_true_training - np.dot(vector_x_training, true_beta))
            vector_y_true_training = np.array([scalar_y_true_training])
            vector_x_beta_training = np.array([scalar_x_beta_training])
            k = 2
            while k <= N_tr:
                vector_x_training = matrix_x[k-1]
                scalar_y_true_training = vector_y[k-1]
                scalar_x_beta_training = np.dot(vector_x_training, reduced_beta)
                vector_y_true_training = np.append(vector_y_true_training, scalar_y_true_training)
                vector_x_beta_training = np.append(vector_x_beta_training, scalar_x_beta_training)
                k=k+1
            RSS_vector = np.subtract(vector_y_true_training, vector_x_beta_training)
            RSS_training[j - 1] = (np.dot(RSS_vector, RSS_vector)) / N_tr
            #print(RSS_training[j - 1])

            j=j+1
        RSS_test_total = np.add(RSS_test_total, RSS_test)
        RSS_training_total = np.add(RSS_training_total, RSS_training)
        repeat = repeat +1
    #Produce RSS vectors
    RSS_test_total = np.divide(RSS_test_total,100)
    RSS_training_total = np.divide(RSS_training_total,100)
    jay = [1,2,3,4,5,6,7,8,9,10]

    print(RSS_test_total)
    print(RSS_training_total)

    #Create Plots
    plt.plot(jay, RSS_training_total, 'ro',  label='RSS Training')
    plt.plot(jay, RSS_test_total, 'bo', label='RSS Testing')
    plt.xlabel('M_j')
    plt.ylabel('RSS')
    plt.legend()
    #plt.title('Average RSS in training and test data sets for LS estimator')
    plt.show
    return
