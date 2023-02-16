import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, lambdify, diff, sin
from scipy.optimize import curve_fit
from matplotlib import cm


# 1.c

def plot_2D_error(losses):
    fig = plt.figure()
    fig.suptitle("Error vs Iteration", fontsize=14)
    plt.plot(losses, color = "blue")
    plt.xlabel('Iteration #')
    plt.ylabel('Error Value')
    plt.show() 
    
    
def plot_2D_linear(a, b, x, y):
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
    a = 1
    b = 1

    errors = [sum(squared_error_func(x_data,y_data,a,b))]
    a_list = [a]
    b_list = [b]

    for i in range(num_iter):
            grad_dir_a = sum(grad_a_func(x_data,y_data, a, b))
            grad_dir_b = sum(grad_b_func(x_data,y_data, a, b))
            a = a - lr * grad_dir_a
            b = b - lr * grad_dir_b
            errors.append(sum(squared_error_func(x_data, y_data, a, b)))
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

# def loss(x, y, a, b):
#   return np.sum((y - f(x, a, b))**2)


def plot_3D_error_vs_iterations(a_list, b_list, losses):
    # Convert the lists to numpy arrays
    
    a_array = np.linspace(0.5, 4 , 100)
    b_array = np.linspace(0.5, 4 , 100)
    
    # Create a meshgrid of the parameter values
    A, B = np.meshgrid(a_array, b_array)
    
    # Calculate the loss surface over the parameters
    Z = np.zeros_like(A)
    for i in range(A.shape[0]):
      for j in range(A.shape[1]):
        Z[i, j] = np.sum(squared_error_func(x_data, y_data, A[i, j], B[i, j]))
    
    # Plot the error surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(A, B, Z, rstride=1, cstride=1, cmap=cm.viridis, alpha=0.5, edgecolor='none')
    
    # Calculate the z values of the parameter values found during all iterations
    ax.scatter(a_list, b_list, losses, color='red', marker='o')
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_zlabel("Loss")
    plt.show()


x,y,a,b = symbols('x,y,a,b')
f = a * sin(b*x)
e = (f - y)**2

def squared_error_func2(x,y,a,b):
    f = a * sin(b*x)
    e = (f - y)**2
    return e


# the symbolic derivative
grad_a = diff(e,a)
grad_b = diff(e,b)

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
#1,d
popt, pcov = curve_fit(F, x_data, y_data)
print("The optimal parameters found using the function curve_fit are: ",popt)


