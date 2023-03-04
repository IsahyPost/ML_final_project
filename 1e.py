import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, lambdify, diff, cos
from scipy.optimize import curve_fit
from matplotlib import cm
import time


# 1e

def plot_3d_error_vs_iterations(a_list, b_list, c_list, losses):
    '''
    Plots a 3D surface plot of the error surface over two parameters, with the
    third parameter visualized as the color of the data points, and the values
    found during the iterations plotted on top

    Parameters
    ----------
    a_list : List of updated values of parameter a after each iteration.
    b_list : List of updated values of parameter b after each iteration.
    c_list : List of updated values of parameter c after each iteration.
    losses : The error values after each iteration.

    Returns
    -------
    None.

    '''
    # Convert the lists to numpy arrays
    a_array = np.array(a_list)
    b_array = np.array(b_list)
    c_array = np.array(c_list)

    # Create a meshgrid of the parameter values
    a_array = np.linspace(np.min(a_array), np.max(a_array), 100)
    b_array = np.linspace(np.min(b_array), np.max(b_array), 100)
    a_mash, b_mash = np.meshgrid(a_array, b_array)

    # Calculate the loss surface over the parameters
    z = np.zeros_like(a_mash)
    for i in range(a_mash.shape[0]):
        for j in range(a_mash.shape[1]):
            z[i, j] = np.sum(squared_error_func(x_data, y_data, a_mash[i, j], b_mash[i, j], res_c[-1]))

    # Plot the error surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(a_mash, b_mash, z, rstride=1, cstride=1, cmap=cm.viridis, alpha=0.5, edgecolor='none')

    # Calculate the z values of the parameter values found during all iterations
    ax.scatter(a_list, b_list, losses, c=c_array, cmap='coolwarm', marker='o')
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_zlabel("Loss")
    plt.show()


def plot_2d_error(losses):
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
    plt.plot(losses, color="blue")
    plt.xlabel('Iteration #')
    plt.ylabel('Error Value')
    plt.show()


def plot_2d_linear(a2plot, b2plot, c2plot, x2plot, y):
    '''
    Plots a 2D linear regression model using the input data and model parameters.

    Parameters
    ----------
    a2plot, b2plot, c2plot : coefficients of the sine function to be plotted
    x2plot : x-axis data points for the scatter plot and the x-axis range for the line plot.
    y : y-axis data points for the scatter plot

    Returns
    -------
    None.

    '''
    # Plot the data and model
    x1 = np.linspace(x2plot[0], x2plot[-1], 500)
    y1 = F(x1, a2plot, b2plot, c2plot)

    plt.plot(x1, y1)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('f(x) = a * sin(b * x)')
    plt.scatter(x2plot, y, color="red")
    # plt.plot(x, f(x, a, b), color="blue")
    plt.show()


def gradient_decent(grad_lr, iter_num, x_data, y_data):
    '''
    Performs gradient descent on given input data and returns the list of updated
    values of the parameters and the error values after each iteration

    Parameters
    ----------
    grad_lr : Learning rate
    iter_num : Number of iterations
    x_data : Input data for the model
    y_data : Output data for the model

    Returns
    -------
    a_list : List of parameter 'a' found during all iterations
    b_list : List of parameter 'b' found during all iterations
    c_list : List of parameter 'c' found during all iterations
    errors : List of errors found during all iterations

    '''
    # Initialize parameters
    a = 1
    b = 1
    c = 19

    # Calculate the error for the initial parameter values
    errors = [sum(squared_error_func(x_data, y_data, a, b, c))]
    
    # Create lists to store the parameter values found during all iterations
    a_list = [a]
    b_list = [b]
    c_list = [c]

    # Perform gradient descent for the given number of iterations
    for i in range(iter_num):
        # Calculate the gradient direction for each parameter
        grad_dir_a = sum(grad_a_func(x_data, y_data, a, b, c))
        grad_dir_b = sum(grad_b_func(x_data, y_data, a, b, c))
        grad_dir_c = sum(grad_c_func(x_data, y_data, a, b, c))
        
        # Update the parameter values based on the gradient direction and learning rate
        a = a - grad_lr * grad_dir_a
        b = b - grad_lr * grad_dir_b
        c = c - grad_lr * grad_dir_c
        
        # Calculate the error for the updated parameter values and store the parameter values
        errors.append(sum(squared_error_func(x_data, y_data, a, b, c)))
        a_list.append(a)
        b_list.append(b)
        c_list.append(c)

    return a_list, b_list, c_list, errors

# Create x_data and y_data with added noise
x_data = np.linspace(-5, 5, 21)
true_a, true_b, true_c = 3, 0.7, 20
y_data = true_a ** 2 * np.cos(true_b * x_data + true_c)
y_data = [y + np.random.normal(0, 1.0) for y in y_data]

# Create symbolic variables and functions for gradient descent and curve fitting
x, y, a, b, c = symbols('x,y,a,b,c')
f = a ** 2 * cos(b * x + c)
e = (f - y) ** 2

# the symbolic derivative
grad_a = diff(e, a)
grad_b = diff(e, b)
grad_c = diff(e, c)

# Create numpy functions from symbolic functions
F = lambdify([x, a, b, c], f, 'numpy')
squared_error_func = lambdify([x, y, a, b, c], e, 'numpy')
grad_a_func = lambdify([x, y, a, b, c], grad_a, 'numpy')
grad_b_func = lambdify([x, y, a, b, c], grad_b, 'numpy')
grad_c_func = lambdify([x, y, a, b, c], grad_c, 'numpy')

# Set the number of iterations and learning rate for gradient descent
num_iter = 4000
lr = 0.00001

# Run gradient descent and print results
start_time = time.time()
res_a, res_b, res_c, errors = gradient_decent(lr, num_iter, x_data, y_data)
print("--- Gradient Descent ---")
print("Time taken: {:.3f}s".format(time.time() - start_time))
print("Final parameters: a = {:.2f}, b = {:.2f}, c = {:.2f}".format(res_a[-1], res_b[-1], res_c[-1]))

# Find the parameters using optimize.curve_fit and print results
start_time = time.time()
popt, pcov = curve_fit(F, x_data, y_data)
print("The optimal parameters found using the function curve_fit are: ", popt)
print("Time taken: {:.3f}s".format(time.time() - start_time))
print("Final parameters: a = {:.4f}, b = {:.4f}, c = {:.4f}".format(*popt))

# Plot the errors and linear regression results
plot_2d_error(errors)
plot_2d_linear(res_a[-1], res_b[-1], res_c[-1], x_data, y_data)

# Plot the 3D error surface over parameter iterations
plot_3d_error_vs_iterations(res_a, res_b, res_c, errors)

