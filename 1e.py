import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, lambdify, diff, sin, cos
from scipy.optimize import curve_fit
from matplotlib import cm
import math as m
import time


# 1.c

def plot_3D_error_vs_iterations(a_list, b_list, c_list, losses):
    # Convert the lists to numpy arrays
    a_array = np.array(a_list)
    b_array = np.array(b_list)
    c_array = np.array(c_list)

    # Create a meshgrid of the parameter values
    a_array = np.linspace(np.min(a_array), np.max(a_array), 100)
    b_array = np.linspace(np.min(b_array), np.max(b_array), 100)
    A, B = np.meshgrid(a_array, b_array)

    # Calculate the loss surface over the parameters
    Z = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            Z[i, j] = np.sum(squared_error_func(x_data, y_data, A[i, j], B[i, j], res_c[-1]))

    # Plot the error surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(A, B, Z, rstride=1, cstride=1, cmap=cm.viridis, alpha=0.5, edgecolor='none')

    # Calculate the z values of the parameter values found during all iterations
    ax.scatter(a_list, b_list, losses,c=c_array,cmap = 'coolwarm', marker='o')
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_zlabel("Loss")
    plt.show()


def plot_2D_error(losses):
    fig = plt.figure()
    fig.suptitle("Error vs Iteration", fontsize=14)
    plt.plot(losses, color="blue")
    plt.xlabel('Iteration #')
    plt.ylabel('Error Value')
    plt.show()


def plot_2D_linear(a, b, c, x, y):
    # Plot the data and model
    x1 = np.linspace(x[0], x[-1], 500)
    y1 = F(x1, a, b, c)

    plt.plot(x1, y1)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('f(x) = a * sin(b * x)')
    plt.scatter(x, y, color="red")
    # plt.plot(x, f(x, a, b), color="blue")
    plt.show()


def gradient_decent(lr, num_iter, x_data, y_data):
    a = 1
    b = 1
    c = 19

    errors = [sum(squared_error_func(x_data, y_data, a, b, c))]
    a_list = [a]
    b_list = [b]
    c_list = [c]

    for i in range(num_iter):
        grad_dir_a = sum(grad_a_func(x_data, y_data, a, b, c))
        grad_dir_b = sum(grad_b_func(x_data, y_data, a, b, c))
        grad_dir_c = sum(grad_c_func(x_data, y_data, a, b, c))
        a = a - lr * grad_dir_a
        b = b - lr * grad_dir_b
        c = c - lr * grad_dir_c
        errors.append(sum(squared_error_func(x_data, y_data, a, b, c)))
        a_list.append(a)
        b_list.append(b)
        c_list.append(c)

    return a_list, b_list, c_list, errors


x_data = np.linspace(-5, 5, 21)
true_a, true_b, true_c = 3, 0.7, 20
y_data = true_a ** 2 * np.cos(true_b * x_data + true_c)
y_data = [y + np.random.normal(0, 1.0) for y in y_data]

# def plot_3D_error_vs_iterations(a_list, b_list, losses):
#     # Convert the lists to numpy arrays

#     a_array = np.linspace(0.5, 4 , 100)
#     b_array = np.linspace(0.5, 4 , 100)

#     # Create a meshgrid of the parameter values
#     A, B = np.meshgrid(a_array, b_array)

#     # Calculate the loss surface over the parameters
#     Z = np.zeros_like(A)
#     for i in range(A.shape[0]):
#       for j in range(A.shape[1]):
#         Z[i, j] = np.sum(squared_error_func(x_data, y_data, A[i, j], B[i, j]))

#     # Plot the error surface
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     ax.plot_surface(A, B, Z, rstride=1, cstride=1, cmap=cm.viridis, alpha=0.5, edgecolor='none')

#     # Calculate the z values of the parameter values found during all iterations
#     ax.scatter(a_list, b_list, losses, color='red', marker='o')
#     ax.set_xlabel("a")
#     ax.set_ylabel("b")
#     ax.set_zlabel("Loss")
#     plt.show()


x, y, a, b, c = symbols('x,y,a,b,c')
f = a ** 2 * cos(b * x + c)
e = (f - y) ** 2

# the symbolic derivative
grad_a = diff(e, a)
grad_b = diff(e, b)
grad_c = diff(e, c)

F = lambdify([x, a, b, c], f, 'numpy')

squared_error_func = lambdify([x, y, a, b, c], e, 'numpy')
grad_a_func = lambdify([x, y, a, b, c], grad_a, 'numpy')
grad_b_func = lambdify([x, y, a, b, c], grad_b, 'numpy')
grad_c_func = lambdify([x, y, a, b, c], grad_c, 'numpy')

num_iter = 4000
lr = 0.00001

start_time = time.time()
res_a, res_b, res_c, errors = gradient_decent(lr, num_iter, x_data, y_data)
print("--- Gradient Descent ---")
print("Time taken: {:.3f}s".format(time.time() - start_time))
print("Final parameters: a = {:.2f}, b = {:.2f}, c = {:.2f}".format(res_a[-1], res_b[-1], res_c[-1]))

# Find the parameters using optimize.curve_fit
start_time = time.time()
popt, pcov = curve_fit(F, x_data, y_data)
print("The optimal parameters found using the function curve_fit are: ", popt)
print("Time taken: {:.3f}s".format(time.time() - start_time))
print("Final parameters: a = {:.4f}, b = {:.4f}, c = {:.4f}".format(*popt))

plot_2D_error(errors)
plot_2D_linear(res_a[-1], res_b[-1], res_c[-1], x_data, y_data)

plot_3D_error_vs_iterations(res_a, res_b, res_c, errors)

# plot_3D_error_vs_iterations(res_a, res_b, errors)

# print(true_a, res_a[-1], true_b, res_b[-1], true_c, res_c[-1])
# #1,d
# popt, pcov = curve_fit(F, x_data, y_data)
# print("The optimal parameters found using the function curve_fit are: ",popt)


