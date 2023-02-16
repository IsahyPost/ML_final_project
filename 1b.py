import numpy as np
from sympy import symbols, lambdify, diff

# 1.b

def gradient_decent(lr, num_iter, x_data, y_data):
    a = 1
    b = 1

    #errors = [sum(squared_error_func(x_data,y_data,a,b))]
    a_list = [a]
    b_list = [b]

    for i in range(num_iter):
            grad_dir_a = sum(grad_a_func(x_data,y_data, a, b))
            grad_dir_b = sum(grad_b_func(x_data,y_data, a, b))
            a = a - lr * grad_dir_a
            b = b - lr * grad_dir_b
            #errors.append(sum(squared_error_func(x_data, y_data, a, b)))
            a_list.append(a)
            b_list.append(b)
            
    return a_list[-1], b_list[-1]

x_data = np.array([-3.0, -2.0, 0.0, 1.0, 3.0, 4.0])
y_data = np.array([-1.5, 2.0, 0.7, 5.0, 3.5, 7.5])

x,y,a,b = symbols('x,y,a,b')
f = a * x + b
e = (f - y)**2

# the symbolic derivative
grad_a = diff(e,a)
grad_b = diff(e,b)

squared_error_func = lambdify([x,y,a,b], e,'numpy') 
grad_a_func = lambdify([x,y,a,b], grad_a,'numpy') 
grad_b_func = lambdify([x,y,a,b], grad_b,'numpy') 

num_iter = 100
lr = 0.01
        
res_a, res_b = gradient_decent(lr, num_iter, x_data, y_data)
print ("Parameter's values are: ", res_a, res_b)
