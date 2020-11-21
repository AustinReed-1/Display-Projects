def MLMC_grad_descent(k, l, x0):
    import math
    h = 0.1 * 2 ** (-l)
    theta = (k * math.pi) / (2 **7)
    T = 10
    number_steps = T * (h ** (-1))

    x1 = x0
    x2 = x0
    i1=1
    while i1 < number_steps + 1:
        df = 4*(x1 ** 3)-3*x1-math.cos(theta)
        x2 = x1 -h * df
        # change = abs(x1-x2)
        x1 = x2
        i1 = i1 + 1
    X = x2
    # print(X)
    return(X)

def MLMC():
    import random
    import sys
    import matplotlib.pyplot as plt
    import math

    L =10
    x_points = []
    points = []
    true_mu = []

    K=1
    while K <= 2 **6:
        sum2=0

        i=0
        while i <= L:
            sum1=0
            # N_i = 5
            N_i = 2 **(L-i)
            if i == 0:
                j = 1
                while j <= N_i:
                    x0 = random.uniform(-1, 1)
                    X2 = MLMC_grad_descent(K, i, x0)
                    Y = X2
                    sum1 = sum1 + Y
                    j = j + 1
                    # print(Y)
            else:
                j=1
                while j <= N_i:
                    x0 = random.uniform(-1, 1)
                    X1=MLMC_grad_descent(K, i-1, x0)
                    X2=MLMC_grad_descent(K, i, x0)
                    Y = X2-X1
                    sum1 = sum1 + Y
                    j = j+1
                    # print(Y)

            sum2 = sum2 + (sum1 / N_i)
            i = i+1
        theta = (K * math.pi) / (2 **7)
        tp1 = math.cos(theta/3)
        tp2 = 0.5*(-tp1+(3*(1- tp1 ** 2)) ** 0.5)
        tp3 = 0.5*(-tp1-(3*(1- tp1 ** 2)) ** 0.5)
        mu = (abs(-1-tp2) * tp3)/2 + (abs(1-tp2)*tp1)/2
        true_mu.append(mu)
        x_points.append(K)
        points.append(sum2)
        K=K+1

    plt.plot(x_points, points, 'ro', label='MLMC Esimations')
    plt.plot(x_points, true_mu, 'bo', label='True values')
    plt.ylabel('Estimator')
    plt.xlabel('k')
    plt.legend(loc="upper right")
    plt.show()