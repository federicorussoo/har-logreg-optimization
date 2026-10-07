# Libraries

'''
In this file we changed the main functions in order to save the CPU time at each iteration and the corresponding accuracy
'''
import pandas as pd
import numpy as np
import time
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from scipy.special import logsumexp
import matplotlib.pyplot as plt
np.random.seed(42)

# Smartphone data 
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

A_train_1 = data_train.drop(columns=["subject", "Activity"]).values
A_test_1 = data_test.drop(columns=["subject", "Activity"]).values

Y_train= data_train["Activity"].values
Y_test = data_test["Activity"].values

scaler = StandardScaler()
A_train = scaler.fit_transform(A_train_1)
A_test = scaler.transform(A_test_1)

I_train = pd.get_dummies(Y_train).values
I_test = pd.get_dummies(Y_test).values

b_train = np.argmax(I_train, axis=1)
b_test = np.argmax(I_test, axis=1)

print(f"Train: A_train {A_train.shape}, I_train {I_train.shape}")
print(f"Test:  A_test {A_test.shape}, I_test {I_test.shape}")


# Functions 

# Full gradient
def grad(Z, A_t, I):
 P = softmax(Z, axis = 1)
 return A_t @(P-I)

# Loss function
def fx(Z, I):
  T_1 = -np.sum(I * Z, axis=1)
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
def gradient_descent(A, I, h = None, n_iter_gd = 100, tol = 500, use_armijo=False, time_up=10):
  np.random.seed(42)
  d = A.shape[1]
  K = I.shape[1]
  A_train_t = A.T
  X = np.random.randn(d, K) * 0.01
  Z = A @ X

  history = {"time": [], "acc": []}
  tot_time = 0.0

  if use_armijo == False:
    L = np.linalg.norm(A_train, ord=2) ** 2
    if h == None:
      stepsize_gd = 1/L
    else:
      stepsize_gd = h/L

  start_time = time.time()

  for k in range(n_iter_gd + 1):

    # save time history
    # we save the time at each iteration and the accuracy to use them in the plot
    if k % time_up == 0 or k == n_iter_gd:
        iter_time = time.time() - start_time
        tot_time += iter_time
        
        Z_pred = A_test @ X 
        acc = np.mean(np.argmax(Z_pred, axis=1) == b_test) * 100
        
        history["time"].append(tot_time)
        history["acc"].append(acc)
        
        start_time = time.time()
        
        if k == n_iter_gd:
            break
        
    gradient_gd = grad(Z, A_train_t, I)

    # Breaking condition
    grad_norm = np.linalg.norm(gradient_gd)
    if grad_norm < tol:
      val_loss = fx(Z, I) 
    
    if use_armijo: 
      val_loss = fx(Z, I)
      direction = -gradient_gd
      alpha_k = armijo(X, A, I, gradient_gd, direction, val_loss, alpha_in=1.0, delta=0.5, gamma=1e-4)
    else:
      alpha_k = stepsize_gd
    
    X -= alpha_k * gradient_gd
    Z = A @ X
  
  return X, history

# Randomized BCGD 
def randomized_BCGD(A,
                    I,
                    n_iter,
                    h = None,
                    tol=500,
                    e=500,
                    time_up=10): 
  np.random.seed(42)
  d = A.shape[1]
  K = I.shape[1]
  X = np.random.randn(d, K) * 0.01
  Z = A @ X 
  L_partials = np.sum(A**2, axis=0)
  A_t = A.T

  history = {"time": [], "acc": []}
  tot_time = 0.0

  start_time = time.time()
  
  for i in range(n_iter + 1): 

    if i % time_up == 0 or i == n_iter:
        iter_time = time.time() - start_time
        tot_time += iter_time
        
        Z_pred = A_test @ X
        acc = np.mean(np.argmax(Z_pred, axis=1) == b_test) * 100
        
        history["time"].append(tot_time)
        history["acc"].append(acc)
        
        start_time = time.time()
        
        if i == n_iter:
            break

    if i % e == 0:
      full_gradient = grad(Z, A_t, I)
      grad_norm = np.linalg.norm(full_gradient)
      
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

  return X, history

# Gauss-Southwell BCGD 
def gauss_southwell_BCGD(A,
                         I, 
                         n_iter, 
                         h = None,
                         tol=500,
                         time_up=10):  
  np.random.seed(42)
  d = A.shape[1] 
  K = I.shape[1] 
  X = np.random.randn(d, K) * 0.01
  Z = A @ X  
  L_partials = np.sum(A**2, axis=0) 
  A_t = A.T

  history = {"time": [], "acc": []}
  tot_time = 0.0

  start_time = time.time()

  for i in range(n_iter):

    if i % time_up == 0 or i == n_iter:
        iter_time = time.time() - start_time
        tot_time += iter_time
        
        Z_pred = A_test @ X
        acc = np.mean(np.argmax(Z_pred, axis=1) == b_test) * 100
        
        history["time"].append(tot_time)
        history["acc"].append(acc)
        
        start_time = time.time() 
      
        if i == n_iter:
            break
            
    grad_gs = grad(Z, A_t, I)
    grad_norm = np.linalg.norm(grad_gs)

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

  return X, history

_, hist_gd = gradient_descent(A_train,
                              I_train,
                              h = 50,
                              n_iter_gd = 1000)

_, hist_arm = gradient_descent(A_train, I_train,
                               h=None,
                               n_iter_gd = 1000,
                               use_armijo = True)

_, hist_rand = randomized_BCGD(A_train,
                               I_train,
                               n_iter = 15000,
                               h = 5)

_, hist_gs = gauss_southwell_BCGD(A_train,
                                  I_train,
                                  n_iter = 15000,
                                  h = 9)


# Plot of all models: accuracy vs CPU time
plt.figure()
plt.plot(hist_gd["time"], hist_gd["acc"], label="GD", color="blue")
plt.plot(hist_rand["time"], hist_rand["acc"], label="Randomized BCGD", color="red")
plt.plot(hist_gs["time"], hist_gs["acc"], label="Gauss-Southwll BCGD", color="green")

plt.xlabel("CPU Time (s)")
plt.ylabel("Test Accuracy (%)")
plt.xlim(-0.2, 2)
plt.legend()
plt.grid(True)
plt.show()

# Plot GD fixed vs GD Armijo 
plt.figure()
plt.plot(hist_gd["time"], hist_gd["acc"], color = "blue", label="GD fixed stepsize")
plt.plot(hist_arm["time"], hist_arm["acc"], color = "orange", label="GD Armijo")

# accuracy GD with tol = 500
plt.axvline(x=0.20, color="blue", linestyle="--", alpha=0.5)
plt.scatter(0.20, 94.06, color="blue", zorder=5)
plt.text(0.35, 95.2, "94.06% at 0.20s", color="blue", verticalalignment="center")

# accuracy GD with Armijo with tol = 500
plt.axvline(x=3.84, color="orange", linestyle="--", alpha=0.5)
plt.scatter(3.84, 94.3, color="orange", zorder=5)
plt.text(4.05, 92.3, "94.3% at 3.84s", color="orange", verticalalignment="center")

plt.xlabel("CPU Time (s)")
plt.ylabel("Test Accuracy (%)")
plt.xlim(-0.2, 5)
plt.legend()
plt.grid(True)
plt.show()
