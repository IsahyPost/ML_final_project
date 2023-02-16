import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
##REMEMBER TO CHANGE THE gradient_decent RETURNING VALUES TO NOT HAVE a,b

# Define the model and loss function
def f(x, a, b):
  return a * x + b

def loss(x, y, a, b):
  return np.sum((y - f(x, a, b))**2)


def gradient_decent(lr, num_iter, x, y):
    
    # Initialize the model parameters
    a = 1
    b = 1
    
    # Initialize the list to store the parameters at each iteration
    a_list = []
    b_list = []
    a_list.append(a)
    b_list.append(b)
    
    losses = []
    losses.append(loss(x, y, a, b))
    
    # Perform gradient descent
    for i in range(num_iter):
      # Calculate the gradient of the loss with respect to the parameters
      grad_a = -2 * np.sum(x * (y - f(x, a, b)))
      grad_b = -2 * np.sum(y - f(x, a, b))
      
      # Update the parameters
      a -= lr * grad_a
      b -= lr * grad_b
      
      # Store the parameters at each iteration
      a_list.append(a)
      b_list.append(b)
    
      # Calculate the loss and appends it to the losses list
      losses.append(loss(x, y, a, b))
      
    
    return a, b, losses, a_list, b_list
  
 
def plot_2D_error(losses):
    fig = plt.figure()
    fig.suptitle("Error vs Iteration", fontsize=14)
    plt.plot(losses, color = "blue")
    plt.xlabel('Iteration #')
    plt.ylabel('Error Value')
    plt.show()   
    
    
def plot_2D_linear(a, b, x, y):
    # Plot the data and model
    plt.scatter(x, y, color="red")
    plt.plot(x, f(x, a, b), color="blue")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
 
    
def plot_3D_error_vs_iterations(a_list, b_list, losses):
    # Convert the lists to numpy arrays
    a_array = np.array(a_list)
    b_array = np.array(b_list)
    
    # Create a meshgrid of the parameter values
    A, B = np.meshgrid(a_array, b_array)
    
    # Calculate the loss surface over the parameters
    Z = np.zeros_like(A)
    for i in range(A.shape[0]):
      for j in range(A.shape[1]):
        Z[i, j] = loss(x, y, A[i, j], B[i, j])
    
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
    
    
    
# Define the data
x = np.array([-3, -2, 0, 1, 3, 4])
y = np.array([-1.5, 2, 0.7, 5, 3.5, 7.5])

# Set the learning rate and number of iterations
lr = 0.01
num_iter = 100

a, b, losses, a_list, b_list = gradient_decent(lr, num_iter, x, y)

# Print the final parameters
print("a:", a)
print("b:", b)
print(*[losses[i] for i in range(len(losses)) if i % 10 == 0], sep="\n")
plot_2D_error(losses)
plot_2D_linear(a, b, x, y)
plot_3D_error_vs_iterations(a_list, b_list, losses)




