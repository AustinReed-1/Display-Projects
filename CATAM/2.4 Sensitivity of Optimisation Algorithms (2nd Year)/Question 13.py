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

def MLMC2():
    import sys
    import random
    import matplotlib.pyplot as plt
    import math

    L = 7
    x_points = []
    points = []
    true_mu = []

    K = 1
    while K <= 2 ** 6:
        sum2 = 0

        i = 0
        while i <= L:
            sum1 = 0
            if i==0:
                N_i = math.floor(16000 /(1+(3**0.5)*((1-2**(-L/2))/(2-2**(1/2)))))
            else:
                N_i = math.floor(16000 /(1+(3**0.5)*((1-2**(-L/2))/(2-2**(1/2)))*(2**i)*(2**i +2**(i-1))**0.5))
            print('i='+str(i))
            print('N='+str(N_i))
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
                j = 1
                while j <= N_i:
                    x0 = random.uniform(-1, 1)
                    X1 = MLMC_grad_descent(K, i - 1, x0)
                    X2 = MLMC_grad_descent(K, i, x0)
                    Y = X2 - X1
                    sum1 = sum1 + Y
                    j = j + 1
                    # print(Y)

            sum2 = sum2 + (sum1 / N_i)
            i = i + 1
        theta = (K * math.pi) / (2 ** 7)
        tp1 = math.cos(theta / 3)
        tp2 = 0.5 * (-tp1 + (3 * (1 - tp1 ** 2)) ** 0.5)
        tp3 = 0.5 * (-tp1 - (3 * (1 - tp1 ** 2)) ** 0.5)
        mu = (abs(-1 - tp2) * tp3) / 2 + (abs(1 - tp2) * tp1) / 2
        true_mu.append(mu)
        x_points.append(K)
        points.append(sum2)
        K = K + 1

    plt.plot(x_points, points, 'ro', label='MLMC Estimations')
    plt.plot(x_points, true_mu, 'bo', label='True values')
    plt.ylabel('Estimator')
    plt.xlabel('k')
    plt.legend(loc="upper right")
    plt.show()

def MLMC3():
    import sys
    import random
    import matplotlib.pyplot as plt
    import math

    L = 7
    x_points = []
    p1_estimate_points = []
    p2_estimate_points = []
    true_p1 = []
    true_p2 = []

    K = 1
    while K <= 2 ** 7:
        sum2 = 0

        i = 0
        while i <= L:
            sum1 = 0
            if i==0:
                N_i = math.floor(16000 /(1+(3**0.5)*((1-2**(-L/2))/(2-2**(1/2)))))
            else:
                N_i = math.floor(16000 /(1+(3**0.5)*((1-2**(-L/2))/(2-2**(1/2)))*(2**i)*(2**i +2**(i-1))**0.5))
            print('i='+str(i))
            print('N='+str(N_i))
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
                j = 1
                while j <= N_i:
                    x0 = random.uniform(-1, 1)
                    X1 = MLMC_grad_descent(K, i - 1, x0)
                    X2 = MLMC_grad_descent(K, i, x0)
                    Y = X2 - X1
                    sum1 = sum1 + Y
                    j = j + 1
                    # print(Y)

            sum2 = sum2 + (sum1 / N_i)
            i = i + 1
        theta = (K * math.pi) / (2 ** 7)
        tp1 = math.cos(theta / 3)
        tp2 = 0.5 * (-tp1 + (3 * (1 - tp1 ** 2)) ** 0.5)
        tp3 = 0.5 * (-tp1 - (3 * (1 - tp1 ** 2)) ** 0.5)
        mu = (abs(-1 - tp2) * tp3) / 2 + (abs(1 - tp2) * tp1) / 2
        p1 = (mu-tp1)/(tp3-tp1)
        p2 = (tp3-mu)/(tp3-tp1)
        p1_estimate = (sum2-tp1)/(tp3-tp1)
        p2_estimate = (tp3-sum2)/(tp3-tp1)
        true_p1.append(p1)
        true_p2.append(p2)
        x_points.append(K)
        p1_estimate_points.append(p1_estimate)
        p2_estimate_points.append(p2_estimate)
        K = K + 1

    plt.plot(x_points, p1_estimate_points, 'ro', label='MLMC Estimator p_1')
    plt.plot(x_points, true_p1, 'bo', label='True value p_1')
    plt.plot(x_points, p2_estimate_points, 'go', label='MLMC Estimator p_2')
    plt.plot(x_points, true_p2, 'yo', label='True value p_2')
    plt.ylabel('Estimator')
    plt.xlabel('k')
    plt.legend(loc="upper right")
    plt.show()

def Tester():
    #works for T=35.
    import math
    k1=1

    while k1<=2**6:
        k2=1
        theta = (k1 * math.pi) / (2 **7)
        print('theta' + str(theta))
        tp1 = math.cos(theta / 3)
        # tp2 = 0.5 * (-tp1 + (3 * (1 - tp1 ** 2)) ** 0.5)
        tp3 = 0.5 * (-tp1 - (3 * (1 - tp1 ** 2)) ** 0.5)
        while k2 <= 2**4:
            x=-1+(k2/(2**3))
            print('x' + str(x))
            a=MLMC_grad_descent(k1,10,x)
            e1=abs(tp3-a)
            e2=abs(tp1-a)
            print(e1)
            print(e2)
            if min(e1,e2) > 2**-10:
                return('failed')
            k2 = k2+1
        k1=k1+1

def func(x):
    return(10*(3**0.5)*((1+(3**0.5)/(2-2**0.5))*(2**((3*x-1)/2))-(3**0.5)*(2**(x-0.5)))-160000)