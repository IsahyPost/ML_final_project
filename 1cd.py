import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, lambdify, diff, sin
from scipy.optimize import curve_fit
from matplotlib import cm


# 1.c

def plot_2D_error(losses):
    '''
    Plots the error values as a function of the iteration number
    
    Parameters
    ----------
    losses : The error values at each iteration.
    
    Returns
    -------
    None.
    '''
    fig = plt.figure()
    fig.suptitle("Error vs Iteration", fontsize=14)
    plt.plot(losses, color = "blue")
    plt.xlabel('Iteration #')
    plt.ylabel('Error Value')
    plt.show() 
    
    
def plot_2D_linear(a, b, x, y):
    '''
    Plots a 2D linear regression model using the data provided

    Parameters
    ----------
    a : Slope of the linear function to be plotted
    b : The y-intercept of the linear function to be plotted
    x : The x-axis data points for the scatter plot and the x-axis range for the line plot
    y : The y-axis data points for the scatter plot

    Returns
    -------
    None.

    '''
    # Plot the data and model
    x1 = np.linspace(x[0], x[-1], 500)
    y1 = F(x1, a, b)

    plt.plot(x1, y1)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('f(x) = a * sin(b * x)')
    plt.scatter(x, y, color="red")
    #plt.plot(x, f(x, a, b), color="blue")
    plt.show()


def gradient_decent(lr, num_iter, x_data, y_data):
    '''
    Performs gradient descent on given input data and returns the list of updated
    values of the parameters and the error values after each iteration

    Parameters
    ----------
    lr : Learning rate for the gradient descent algorithm
    num_iter : Number of iterations for the gradient descent algorithm
    x_data : Array of x values representing the independent variable (numpy array)
    y_data : Array of y values representing the dependent variable (numpy array)

    Returns
    -------
    a_list : List of parameter 'a' values at each iteration during the gradient descent algorithm.
    b_list : List of parameter 'b' values at each iteration during the gradient descent algorithm.
    errors : List of loss values at each iteration during the gradient descent algorithm.
    '''
    a = 1
    b = 1

    errors = [sum(squared_error_func(x_data,y_data,a,b))]
    a_list = [a]
    b_list = [b]

    for i in range(num_iter):
        # Calculate the gradient of the error function with respect to a and b
            grad_dir_a = sum(grad_a_func(x_data,y_data, a, b))
            grad_dir_b = sum(grad_b_func(x_data,y_data, a, b))
            
        # Update the values of a and b using gradient descent
            a = a - lr * grad_dir_a
            b = b - lr * grad_dir_b
            
        # Calculate the error for the updated values of a and b
            errors.append(sum(squared_error_func(x_data, y_data, a, b)))
            
        # Append the updated values of a and b to their respective lists
            a_list.append(a)
            b_list.append(b)
            
    return a_list, b_list, errors

x_data = np.array([-5., -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1., -0.5, 0.,
                   0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5. ])

y_data = np.array([-2.16498306, -1.53726731, 1.67075645, 2.47647932, 4.49579917,
                   1.14600963, 0.15938811, -3.09848048, -3.67902427, -1.84892687,
                   -0.11705947, 3.14778203, 4.26365256, 2.49120585, 0.55300516,
                   -2.105836 , -2.68898773, -2.39982575, -0.50261972, 1.40235643,
                   2.15371399])


def plot_3D_error_vs_iterations(a_list, b_list, losses):
    '''
    Plots a 3D error surface using the given lists of parameters and losses and
    highlights the parameter values found during all iterations.

    Parameters
    ----------
    a_list : List of parameter a values found during gradient descent
    b_list : List of parameter b values found during gradient descent
    losses : List of error values corresponding to each set of a and b values 
             during gradient descent.

    Returns
    -------
    None.

    '''
    # Converts the lists to numpy arrays
    
    a_array = np.linspace(0.5, 4 , 100)
    b_array = np.linspace(0.5, 4 , 100)
    
    # Creates a meshgrid of the parameter values
    A, B = np.meshgrid(a_array, b_array)
    
    # Calculates the loss surface over the parameters
    Z = np.zeros_like(A)
    for i in range(A.shape[0]):
      for j in range(A.shape[1]):
        Z[i, j] = np.sum(squared_error_func(x_data, y_data, A[i, j], B[i, j]))
    
    # Plots the error surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(A, B, Z, rstride=1, cstride=1, cmap=cm.viridis, alpha=0.5, edgecolor='none')
    
    # Calculates the z values of the parameter values found during all iterations
    ax.scatter(a_list, b_list, losses, color='red', marker='o')
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_zlabel("Loss")
    plt.show()


def squared_error_func2(x,y,a,b):
    '''
    Calculates the squared error between a 2D linear regression model and the actual data.

    Parameters
    ----------
    x : Array of x values representing the independent variable
    y : Array of y values representing the dependent variable
    a : The slope of the linear function to be plotted
    b : The y-intercept of the linear function to be plotted

    Returns
    -------
    e : The squared error between the predicted and actual y values for a given a and b.

    '''
    f = a * sin(b*x)
    e = (f - y)**2
    return e


# Define the symbolic equation

x,y,a,b = symbols('x,y,a,b')
f = a * sin(b*x)
e = (f - y)**2

# the symbolic derivative
grad_a = diff(e,a)
grad_b = diff(e,b)

# Convert the symbolic functions to lambda functions
F = lambdify([x,a,b],f,'numpy') 
squared_error_func = lambdify([x,y,a,b], e,'numpy') 
grad_a_func = lambdify([x,y,a,b], grad_a,'numpy') 
grad_b_func = lambdify([x,y,a,b], grad_b,'numpy') 

num_iter = 250
lr = 0.001
        
res_a, res_b, errors = gradient_decent(lr, num_iter, x_data, y_data)

plot_2D_error(errors)
plot_2D_linear(res_a[-1], res_b[-1], x_data, y_data)
plot_3D_error_vs_iterations(res_a, res_b, errors)

print(res_a[-1], res_b[-1])

# 1d
# Fit a curve to the data using the curve_fit function and store the optimal
# parameters and covariance matrix in popt and pcov
popt, pcov = curve_fit(F, x_data, y_data)

# Print the optimal parameters
print("The optimal parameters found using the function curve_fit are: ",popt)


