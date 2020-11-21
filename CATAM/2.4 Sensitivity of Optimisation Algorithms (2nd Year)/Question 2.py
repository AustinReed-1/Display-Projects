def grad_descent(h, number_steps):
    import math
    k=-50
    theta = math.pi/6
    while k<51:
        x0 = k / 50
        x1 = x0
        x2 = x0
        i=1
        while i< number_steps + 1:
            df = 4*(x1 ** 3)-3*x1-math.cos(theta)
            x2 = x1 -h * df
            change = abs(x1-x2)
            x1 = x2
            i = i + 1
        # print(k)
        print('x_final = '+ str(x2))
        print('Difference = ' + str(change))
        k = k+1