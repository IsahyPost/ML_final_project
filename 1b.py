import numpy as np
from sympy import symbols, lambdify, diff
#REMEMBER TO CHECK: do the lines errors = [sum(squared_error)....] need to be deleted?
# 1.b

def gradient_decent(lr, num_iter, x_data, y_data):
    '''
    Performs gradient descent on the input data to optimize the parameters 'a'
    and 'b' of a linear model. It updates the parameters using the gradients and
    learning rate, and returns the optimized values of 'a' and 'b'.

    Parameters
    ----------
    lr : Learning rate used in gradient descent optimization
    num_iter : The number of iterations to be performed in gradient descent optimization
    x_data : The input values of the data (numpy.ndarray)
    y_data : The target values of the data (numpy.ndarray)

    Returns
    -------
    res_a: optimized value of parameter 'a'
    res_b: optimized value of parameter 'b'

    '''
    # Initialize the parameters
    a = 1
    b = 1

    #errors = [sum(squared_error_func(x_data,y_data,a,b))]
    
    # Initialize lists to store the parameter values at each iteration
    a_list = [a]
    b_list = [b]

    # Loop through the specified number of iterations
    for i in range(num_iter):
            grad_dir_a = sum(grad_a_func(x_data,y_data, a, b))
            grad_dir_b = sum(grad_b_func(x_data,y_data, a, b))
            
            # Update the parameters using the gradients and learning rate
            a = a - lr * grad_dir_a
            b = b - lr * grad_dir_b
            
            #errors.append(sum(squared_error_func(x_data, y_data, a, b)))
            
            # Store the parameter values at each iteration
            a_list.append(a)
            b_list.append(b)
            
    return a_list[-1], b_list[-1]

# Define the data
x_data = np.array([-3.0, -2.0, 0.0, 1.0, 3.0, 4.0])
y_data = np.array([-1.5, 2.0, 0.7, 5.0, 3.5, 7.5])

# Define the symbolic variables and functions for the model and loss
x,y,a,b = symbols('x,y,a,b')
f = a * x + b
e = (f - y)**2

# the symbolic derivative
grad_a = diff(e,a)
grad_b = diff(e,b)

# Convert the symbolic functions to numpy functions for efficient computation
squared_error_func = lambdify([x,y,a,b], e,'numpy') 
grad_a_func = lambdify([x,y,a,b], grad_a,'numpy') 
grad_b_func = lambdify([x,y,a,b], grad_b,'numpy') 

# Set the hyperparameters for the gradient descent algorithm
num_iter = 100
lr = 0.01
        
# Run the gradient descent algorithm on the data
res_a, res_b = gradient_decent(lr, num_iter, x_data, y_data)
print ("Parameter's values are: ", res_a, res_b)
