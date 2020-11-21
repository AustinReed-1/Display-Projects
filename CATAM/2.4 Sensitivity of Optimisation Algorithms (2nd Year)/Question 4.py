def rand_grad_descent(k):
    import math
    import random

    h=0.1*2 ** (-k)
    N = 2 ** (20-k)
    theta = math.pi/4
    T = 10
    number_steps = T*h**(-1)
    k=1
    sigma = 0
    sigma2 = 0
    while k<N+1:
        x0 = random.uniform(-1,1)
        x1 = x0
        x2 = x0
        i=1
        while i< number_steps + 1:
            df = 4*(x1 ** 3)-3*x1-math.cos(theta)
            x2 = x1 -h * df
            # change = abs(x1-x2)
            x1 = x2
            i = i + 1
        sigma = sigma + x2
        # sigma2 = sigma2 + (0.346-x2) ** 2
        # print(k)
        # print(x2)
        # print('e' + str(change))

        k = k+1
    # estimator = sigma / N
    print(sigma/N)