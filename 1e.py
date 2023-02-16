import numpy as np
import math
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit


# Define the model and its derivative with respect to the parameters
def model(x, a, b, c):
    return a**2 * np.cos(b * x + c)

def grad_a(x, a, b, c):
    return 2 * a * np.cos(b * x + c)

def grad_b(x, a, b, c):
    return -a**2 * np.sin(b * x + c) * x

def grad_c(x, a, b, c):
    return -a**2 * np.sin(b * x + c)

# Define the gradient descent function
def gradient_descent(x_data, y_data, num_iter, lr):
    a, b, c = np.random.randn(3)
    errors = []
    a_list, b_list, c_list = [a], [b], [c]
    for i in range(num_iter):
        grad_a_sum = np.sum(grad_a(x_data, a, b, c))
        grad_b_sum = np.sum(grad_b(x_data, a, b, c))
        grad_c_sum = np.sum(grad_c(x_data, a, b, c))
        a -= lr * grad_a_sum
        b -= lr * grad_b_sum
        c -= lr * grad_c_sum
        error = np.sum((model(x_data, a, b, c) - y_data)**2)
        errors.append(error)
        a_list.append(a)
        b_list.append(b)
        c_list.append(c)
    return a_list, b_list, c_list, errors


# Generate synthetic data for the model
np.random.seed(0)
x_data = np.linspace(-5, 5, 21)
a, b, c = 3, 0.7, 20
y_data = a**2 * np.cos(b * x_data + c)
noise = 0.5 * np.random.normal(size=x_data.shape)
y_data_noisy = y_data + noise

# Find the parameters using gradient descent
num_iter = 500
lr = 0.1
start_time = time.time()
a_gd, b_gd, c_gd, errors_gd = gradient_descent(x_data, y_data_noisy, num_iter, lr)
print("--- Gradient Descent ---")
print("Time taken: {:.2f}s".format(time.time() - start_time))
print("Final parameters: a = {:.2f}, b = {:.2f}, c = {:.2f}".format(a_gd[-1], b_gd[-1], c_gd[-1]))

x_fit = np.linspace(x_data.min(), x_data.max(), 1000)
y_fit = a_gd[-1]**2 * np.cos(b_gd[-1] * x_fit + c_gd[-1])
plt.scatter(x_data, y_data_noisy, label="Data (with noise)")
plt.plot(x_fit, y_fit, label="Fit (Gradient Descent)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()                 

plt.plot(errors_gd)
plt.xlabel("Iteration #")
plt.ylabel("Error")
plt.show()

# Find the parameters using optimize.curve_fit
start_time = time.time()
popt, pcov = curve_fit(model, x_data, y_data_noisy)
print("--- optimize.curve_fit ---")
print("Time taken: {:.2f}s".format(time.time() - start_time))
print("Final parameters: a = {:.2f}, b = {:.2f}, c = {:.2f}".format(*popt))

x_fit = np.linspace(x_data.min(), x_data.max(), 1000)
y_fit = model(x_fit, *popt)
plt.scatter(x_data, y_data_noisy, label="Data (with noise)")
plt.plot(x_fit, y_fit, label="Fit (optimize.curve_fit)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Compare the parameters found by both methods
print("--- Comparison ---")
print("Gradient Descent: a = {:.2f}, b = {:.2f}, c = {:.2f}".format(a_gd[-1], b_gd[-1], c_gd[-1]))
print("curve_fit: a = {:.2f}, b = {:.2f}, c = {:.2f}".format(popt[0], popt[1], popt[2]))


'''
The final parameters obtained from gradient descent and from scipy.optimize.curve_fit
can be different. The reason for this is that these two methods might converge to different
local minima or have different convergence rates. The final parameters obtained by curve_fit
are the parameters that minimize the sum of squared differences between the model and the data,
while the final parameters obtained by gradient descent are the parameters that minimize
the sum of squared differences between the model and the data as estimated by the gradient descent algorithm.

It is possible that the parameters obtained by curve_fit are more accurate because curve_fit
uses more sophisticated optimization algorithms compared to gradient descent. However, gradient descent can
be used to solve more complex optimization problems and can be easier to implement in some cases.
'''



