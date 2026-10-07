# Libraries
import numpy as np
import time
from scipy.special import softmax
from scipy.special import logsumexp
np.random.seed(42)


# Homework generated data
m = 1000
d = 1000
K = 50

A = np.random.randn(m, d)
X = np.random.randn(d, K)
E = np.random.randn(m, K)
Y = A @ X + E
b = np.argmax(Y, axis=1) + 1

I = np.zeros((m, K))
I[np.arange(m), b - 1] = 1

A_train = A
A_test = A
b_train = b
b_test = b
I_train = I
I_test = I

# Functions (Gradient, Loss and Armijo)

# Full gradient
def grad(Z, A_t, I):
 P = softmax(Z, axis = 1)
 return A_t @(P-I)

# Loss function
def fx(Z, I):
  T_1 = -np.sum(I * Z, axis=1)
  # we used logsumexp to avoid numerical overflow 
  T_2 = logsumexp(Z, axis=1)
  return np.sum(T_1 + T_2)

# Armijo Rule
def armijo(X, A, I, grad, dir, cur_loss, alpha_in=0.001, delta=0.5, gamma=1e-4):
  alpha = alpha_in

  while True:
    X_new = X + alpha * dir
    Z = A @ X_new
    new_loss = fx(Z, I)
    if new_loss <= cur_loss + gamma * alpha * np.sum(grad*dir):
      break
    alpha = alpha * delta 
    
  return alpha


# Gradient Descent
def gradient_descent(A, I, h = None, n_iter_gd = 100, tol = 500, use_armijo=False):
  np.random.seed(42)
  d = A.shape[1]
  K = I.shape[1]
  A_train_t = A.T
  X = np.random.randn(d, K) * 0.01
  Z = A @ X

  if use_armijo == False:
    L = np.linalg.norm(A_train, ord=2) ** 2
    if h == None:
      stepsize_gd = 1/L
    else:
      stepsize_gd = h/L

  for k in range(n_iter_gd):
    gradient_gd = grad(Z, A_train_t, I)

    # Breaking condition
    # we used the frobenius norm to compute the gradient in all the functions as it is a useful and fast approximation of the spectral norm
    grad_norm = np.linalg.norm(gradient_gd)
    if grad_norm < tol:
      val_loss = fx(Z, I) 
      print(f"Iter {k}, f(x) = {round(val_loss,3):.4f}, Stepsize: {'None' if k == 0 else round(alpha_k, 7)}")
      print(f"Convergence at iter {k}")
      break
    
    # Stepsize determination
    if use_armijo: 
      val_loss = fx(Z, I)
      direction = -gradient_gd
      alpha_k = armijo(X, A, I, gradient_gd, direction, val_loss, alpha_in=1.0, delta=0.5, gamma=1e-4)
    else:
      alpha_k = stepsize_gd
    
    X -= alpha_k * gradient_gd
    Z = A @ X

    if k == n_iter_gd - 1 and grad_norm >= tol:
        val_loss = fx(Z, I)
        print(f"Iter {k}, f(x) = {round(val_loss,3):.4f}, Stepsize: {round(alpha_k,7)}")
        print(f"Max iteration reached: {n_iter_gd}")
  
  return X

# Test GD
tollist = [100,500]    
step_multipliers = [50] 

n_iteration = 1000
use_armijo = False 
# we arranged a dictionary to store the results in every testing section of the code
results = {}

for tol in tollist:
  if not use_armijo: 
    for mult in step_multipliers:
      print("Gradient Descent (GD)")
      print(f"stepsize: {mult}/L, tol: {tol}")
      
      # we saved only the CPU time needed to minimize the objective function
      start_time = time.time()
      X_f = gradient_descent(A_train, I_train, h=mult, n_iter_gd=n_iteration, tol=tol, use_armijo=False) 
      cpu_time = time.time() - start_time
      
      z_pred = A_test @ X_f 
      prediction = np.argmax(z_pred, axis=1) +1
      accuracy = np.mean(prediction == b_test)
      
      results[(mult, tol)] = (round(accuracy, 3), round(cpu_time, 3))
      
      print(f"accuracy: {round(accuracy * 100, 2)}%")
      print(f"CPU time: {round(cpu_time, 2)} s\n")

  else: 
    print("Gradient Descent (GD)")
    print(f"stepsize: Armijo, tol: {tol}")

    start_time = time.time()
    X_f = gradient_descent(A_train, I_train, h=None, n_iter_gd=n_iteration, tol=tol, use_armijo=True) 
    cpu_time = time.time() - start_time

    Z_f = A_test @ X_f 
    prediction = np.argmax(Z_f, axis=1) + 1
    accuracy = np.mean(prediction == b_test)
    
    results[("armijo", tol)] = (round(accuracy, 3), round(cpu_time, 3))
    
    print(f"accuracy: {round(accuracy * 100, 2)}%")
    print(f"CPU time: {round(cpu_time, 2)}")
    print(" ")

# Randomized BCGD 
def randomized_BCGD(A,
                    I,
                    n_iter,
                    h = None,
                    tol=500,
                    e=500): 
  np.random.seed(42)
  d = A.shape[1]
  K = I.shape[1]
  X = np.random.randn(d, K) * 0.01
  Z = A @ X 
  L_partials = np.sum(A**2, axis=0)
  A_t = A.T
  
  for i in range(n_iter): 
    
    # checking the gradient norm every "e" iteration to avoid slowing down the algorithm
    if i % e == 0:
      
      full_gradient = grad(Z, A_t, I)
      grad_norm = np.linalg.norm(full_gradient)
      
      # convergence check
      if grad_norm < tol:
        val_loss = fx(Z, I)
        print(f"Iter {i}, f(x) = {round(val_loss,3):.4f}, Stepsize: {'None' if i == 0 else round(alpha, 7)}")
        print(f"Convergence at iter {i}")
        break

    j = np.random.randint(0,d) 
    column_Aj = A[:,j] 
    
    if h is None:
      alpha = 1 / L_partials[j]
    else:
      alpha = h / L_partials[j] 
    
    P = softmax(Z, axis = 1)
    gradient_j = column_Aj.T @ (P-I)  

    delta_j =  - alpha * gradient_j
    X[j,: ] += delta_j 

    for u in range(K):
      var = delta_j[u]
      Z[:,u] += column_Aj * var 

    if i == n_iter-1:
      val_loss = fx(Z, I)
      print(f"Iter {i}, f(x) = {round(val_loss,3):.4f}, Stepsize: {round(alpha,7)}")
      print(f"Max iteration reached: {n_iter}")
       
  return X

# Test Randomized BCGD
check = 500
step_multipliers = [5,8]
tollist = [100,500]
results = {}

for mult in step_multipliers:
  for tol in tollist:
    print("BCGD Randomized")
    print(f"stepsize: {mult}/L, tol: {tol}")

    start_time = time.time()
    x_pred= randomized_BCGD(A_train, I_train, 15000, h=mult, tol=tol, e = check) 
    cpu_time = time.time() - start_time 
    
    z_pred = A_test @ x_pred
    prediction = np.argmax(z_pred, axis=1) +1
    accuracy = np.mean(prediction == b_test)
  
    results[(mult, tol)] = (round(accuracy, 3), round(cpu_time, 3))

    print(f"accuracy: {round(accuracy * 100, 2)}%")
    print(f"CPU time: {round(cpu_time, 2)}")
    print(" ")



# Gauss-Southwell BCGD 
def gauss_southwell_BCGD(A,
                         I, 
                         n_iter, 
                         h = None,
                         tol=500):  
  np.random.seed(42)
  d = A.shape[1] 
  K = I.shape[1] 
  X = np.random.randn(d, K) * 0.01
  Z = A @ X  
  L_partials = np.sum(A**2, axis=0) 
  A_t = A.T

  for i in range(n_iter):
    grad_gs = grad(Z, A_t, I)
    grad_norm = np.linalg.norm(grad_gs)

    # convergence check
    if grad_norm < tol:
      val_loss = fx(Z, I)
      print(f"Iter {i}, f(x) = {round(val_loss,3):.4f}, Stepsize: {'None' if i == 0 else round(alpha, 7)}")
      print(f"Convergence at iter {i}")
      break

    row_norms = np.linalg.norm(grad_gs, axis=1)
    j = np.argmax(row_norms)

    if h is None:
      alpha = 1 / L_partials[j]
    else:
      alpha = h / L_partials[j]
    
    column_Aj = A[:,j]
    gradient_j = grad_gs[j, :]

    delta_j =  - alpha * gradient_j
    X[j,: ] += delta_j

    for u in range(K):
      var = delta_j[u]
      Z[:,u] += column_Aj * var

    if i == n_iter -1:
      val_loss = fx(Z, I)
      print(f"Iter {i}, f(x) = {round(val_loss,3):.4f}, Stepsize: {round(alpha,7)}")
      print(f"Max iterations reached:{n_iter}")
  return X

# Test Gauss-Southwell BCGD
step_multipliers = [9, 17]
tollist = [100,500]
results = {}

for mult in step_multipliers:
  for tol in tollist:
    print("BCGD Gauss-Southwell")
    print(f"stepsize: {mult}/L, tol: {tol}")

    start_time = time.time()
    x_pred = gauss_southwell_BCGD(A_train, I_train, 15000, h=mult, tol=tol) 
    cpu_time = time.time() - start_time 

    z_pred = A_test @ x_pred
    prediction = np.argmax(z_pred, axis=1) + 1
    accuracy = np.mean(prediction == b_test)

    results[(mult, tol)] = (round(accuracy, 3), round(cpu_time, 3))

    print(f"accuracy: {round(accuracy * 100, 2)}%")
    print(f"CPU time: {round(cpu_time, 2)}")
    print(" ")
