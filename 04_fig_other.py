'''
In this file, the logic behind the function is the same as in "smartphone_data", we just added a counter to track variables for the plots
Moreover, we added a parameter called "no_break": if it is triggered, the function records the iteration at which the break would happen but does not stop 
until it reaches the maximum number of iterations (to store data useful for plots). We did this using simple if clauses, because in these plots the CPU time does not matter
'''

import pandas as pd
import numpy as np
from scipy.special import softmax, logsumexp
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
def gradient_descent(A,
                    I,
                    h = None,
                    n_iter_gd = 100,
                    tol = 500,
                    use_armijo=False,
                    no_brake = False):
  np.random.seed(42)
  d = A.shape[1]
  K = I.shape[1]
  A_train_t = A.T
  X = np.random.randn(d, K) * 0.01
  Z = A @ X

  acc_hist = []
  iter_hist = []
  stop_iter = None
  stop_acc = None

  if use_armijo == False:
    L = np.linalg.norm(A_train, ord=2) ** 2
    if h == None:
      stepsize_gd = 1/L
    else:
      stepsize_gd = h/L

  for k in range(n_iter_gd):
    gradient_gd = grad(Z, A_train_t, I)

    grad_norm = np.linalg.norm(gradient_gd)
    
    if no_brake == True:
        if grad_norm < tol and stop_iter is None:
            val_loss = fx(Z, I) 
            stop_iter = k
            pred_stop = np.argmax(A_test @ X, axis=1)
            stop_acc = np.mean(pred_stop == b_test) * 100
            print(f"Convergence at iter {k}, f(x) = {round(val_loss,3):.4f}")
    else:
        if grad_norm < tol:
            val_loss = fx(Z, I) 
            stop_iter = k
            pred_stop = np.argmax(A_test @ X, axis=1)
            stop_acc = np.mean(pred_stop == b_test) * 100
            print(f"Iter {k}, f(x) = {round(val_loss,3):.4f}")
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

    if k % 10 == 0:  
        pred_test = np.argmax(A_test @ X, axis=1)
        acc = np.mean(pred_test == b_test) * 100
        acc_hist.append(acc)
        iter_hist.append(k)

    if k == n_iter_gd - 1 and grad_norm >= tol:
        val_loss = fx(Z, I)
        print(f"Iter {k}, f(x) = {round(val_loss,3):.4f}, Stepsize: {round(alpha_k,7)}")
        print(f"Max iteration reached: {n_iter_gd}")
  
  return X, iter_hist, acc_hist, stop_iter, stop_acc # returning all the variables and lists for the plot below

# Randomized BCGD 
def randomized_BCGD(A,
                    I,
                    n_iter,
                    h = None,
                    tol=500,
                    e=500,
                    no_brake = False): 
  np.random.seed(42)
  d = A.shape[1]
  K = I.shape[1]
  X = np.random.randn(d, K) * 0.01
  Z = A @ X 
  L_partials = np.sum(A**2, axis=0)
  A_t = A.T

  acc_hist = []
  iter_hist = []
  stop_iter = None
  stop_acc = None

  for i in range(n_iter): 

    if i % e == 0:
      
      full_gradient = grad(Z, A_t, I)
      grad_norm = np.linalg.norm(full_gradient)
      
      if no_brake == True:
          if grad_norm < tol and stop_iter is None:
              val_loss = fx(Z, I)
              stop_iter = i
              pred_stop = np.argmax(A_test @ X, axis=1)
              stop_acc = np.mean(pred_stop == b_test) * 100
              print(f"Convergence at iter {i}, f(x) = {round(val_loss,3):.4f}")
      else:
          if grad_norm < tol:
              val_loss = fx(Z, I)
              stop_iter = i
              pred_stop = np.argmax(A_test @ X, axis=1)
              stop_acc = np.mean(pred_stop == b_test) * 100
              print(f"Iter {i}, f(x) = {round(val_loss,3):.4f}")
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

    if i % 100 == 0: 
        pred_test = np.argmax(A_test @ X, axis=1)
        acc = np.mean(pred_test == b_test) * 100
        acc_hist.append(acc)
        iter_hist.append(i)

    for u in range(K):
      var = delta_j[u]
      Z[:,u] += column_Aj * var 

    if i == n_iter-1:
      val_loss = fx(Z, I)
      print(f"Iter {i}, f(x) = {round(val_loss,3):.4f}, Stepsize: {round(alpha,7)}")
      print(f"Max iteration reached: {n_iter}")
       
  return X, iter_hist, acc_hist, stop_iter, stop_acc


# Gauss-Southwell BCGD 
def gauss_southwell_BCGD(A,
                        I,
                        n_iter,
                        h = None,
                        tol=500,
                        return_counts=False,
                        no_brake = False): 
    np.random.seed(42)
    d = A.shape[1] 
    K = I.shape[1] 
    X = np.random.randn(d, K) * 0.01 
    Z = A @ X  
    L_partials = np.sum(A**2, axis=0) 
    A_t = A.T 
  
    counts = np.zeros(d)
    acc_hist = []
    iter_hist = []
    stop_iter = None
    stop_acc = None

    for i in range(n_iter):
        grad_gs = grad(Z, A_t, I)
        grad_norm = np.linalg.norm(grad_gs)

        if no_brake == True:
            if grad_norm < tol and stop_iter is None:
                val_loss = fx(Z, I)
                stop_iter = i
                pred_stop = np.argmax(A_test @ X, axis=1)
                stop_acc = np.mean(pred_stop == b_test) * 100
                print(f"Convergence at iter {i}, f(x) = {round(val_loss,3):.4f}")
        else:
            if grad_norm < tol:
                val_loss = fx(Z, I)
                stop_iter = i
                pred_stop = np.argmax(A_test @ X, axis=1)
                stop_acc = np.mean(pred_stop == b_test) * 100
                print(f"Iter {i}, f(x) = {round(val_loss,3):.4f}")
                print(f"Convergence at iter {i}")
                break

        row_norms = np.linalg.norm(grad_gs, axis=1)
        j = np.argmax(row_norms)
        
        counts[j] += 1

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

        if i % 10 == 0:  
            pred_test = np.argmax(A_test @ X, axis=1)
            acc = np.mean(pred_test == b_test) * 100
            acc_hist.append(acc)
            iter_hist.append(i)

        if i == n_iter -1:
            val_loss = fx(Z, I)
            print(f"Iter {i}, f(x) = {round(val_loss,3):.4f}, Stepsize: {round(alpha,7)}")
            print(f"Max iterations reached:{n_iter}")
            
    if return_counts == True:
            return X, iter_hist, acc_hist, stop_iter, stop_acc, counts
    return X, iter_hist, acc_hist, stop_iter, stop_acc


# Gauss-Southwell Features plot
_, _, _, _, _, feature_count = gauss_southwell_BCGD(A_train, I_train, n_iter = 15000, h = 9, tol = 500, return_counts=True)

# taking the most updated 15 features
n_plotted = 15 
top_idx = np.argsort(feature_count)[-n_plotted:][::-1] 
top_values = feature_count[top_idx]
labels  =  [str(i) for i in top_idx]

# plotting the histogram
plt.figure(figsize = (10, 5))

plt.bar(labels, top_values)
plt.xlabel("Feature")
plt.ylabel("Frequency")
plt.show()



# PCA analysis to see the data in 2 dimensions
pca  =  PCA(n_components = 2)
A_train_2 = pca.fit_transform(A_train)


plt.figure()

classes = np.unique(b_train)
names = ["Walking", "Walking Upstairs", "Walking Downstairs", "Sitting", "Standing", "Laying"]

colors = ["yellow", "skyblue", "green", "purple", "orange", "red"]
for c in classes:
    idx = (b_train == c)
    plt.scatter(A_train_2[idx, 0],
                A_train_2[idx, 1],
                edgecolors ="black",
                label = names[c],
                color = colors[c])

plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(title = "Class")
plt.show()



# Error visualization in 2 dimension (PCA)

X_gd, _, _, _, _ = gradient_descent(A_train, I_train,
                                    h = 50,
                                    n_iter_gd = 1000,
                                    tol = 500,
                                    use_armijo = False)

X_bcgd, _, _, _, _ = randomized_BCGD(A_train,
                                     I_train,
                                     n_iter = 5000,
                                     h = 5,
                                     tol = 500,
                                     e = 500)

X_gs, _, _, _, _ = gauss_southwell_BCGD(A_train,
                                        I_train,
                                        n_iter = 1000,
                                        h = 9, 
                                        tol = 500)

prev_gd = np.argmax(A_train @ X_gd, axis = 1)
prev_bcgd = np.argmax(A_train @ X_bcgd, axis = 1)
prev_gs = np.argmax(A_train @ X_gs, axis = 1)

# correct is True if the label is matching, False if it's not matching, one for every algorithm
correct_gd = (prev_gd == b_train)
correct_rand = (prev_bcgd == b_train)
correct_gs = (prev_gs == b_train)

# to make the misclassified points visible we split the points to 
# print first the correct examples and then the wrong examples 

# GD 
# assigning the right color to each point
plt.figure()
plt.scatter(A_train_2[correct_gd, 0],
            A_train_2[correct_gd, 1],
            color = "lightgray",
            edgecolors = "black")
plt.scatter(A_train_2[~correct_gd, 0],
            A_train_2[~correct_gd, 1],
            color = "red",
            edgecolors = "black")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

# Randomized BCGD
plt.figure()
plt.scatter(A_train_2[correct_rand, 0],
            A_train_2[correct_rand, 1],
            color = "lightgray",
            edgecolors = "black")
plt.scatter(A_train_2[~correct_rand, 0],
            A_train_2[~correct_rand, 1],
            color = "red",
            edgecolors = "black")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

# Gauss-Southwell BCGD
plt.figure()
plt.scatter(A_train_2[correct_gs, 0],
            A_train_2[correct_gs, 1],
            color = "lightgray",
            edgecolors = "black")
plt.scatter(A_train_2[~correct_gs, 0],
            A_train_2[~correct_gs, 1],
            color = "red",
            edgecolors = "black")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()




# Plotting the accuracy vs iterations and the stopping condition to see if the algorithms naturally stop at the accuracy peak without wasting iterations
X_gd, iter_gd, acc_gd, stop_i_gd, stop_a_gd = gradient_descent(A_train,
                                                               I_train,
                                                               h = 50,
                                                               n_iter_gd = 300,
                                                               tol=500,
                                                               use_armijo = False,
                                                               no_brake = True)

X_rand, iter_rand, acc_rand, stop_i_rand, stop_a_rand = randomized_BCGD(A_train,
                                                                        I_train,
                                                                        n_iter = 3000,
                                                                        h = 5,
                                                                        tol = 500,
                                                                        e = 500,
                                                                        no_brake = True)

X_gs, iter_gs, acc_gs, stop_i_gs, stop_a_gs = gauss_southwell_BCGD(A_train,
                                                                   I_train,
                                                                   n_iter = 1000,
                                                                   h = 9,
                                                                   tol = 500,
                                                                   no_brake = True)


# Plot GD
plt.figure(figsize=(8, 5))
plt.plot(iter_gd, acc_gd, color = "blue", label = "Test Accuracy")

# if the algorithm stopped after reaching the assigned tolerance level we draw the stopping line and the accuracy 
if stop_i_gd is not None:
    plt.axvline(x=stop_i_gd, color = "black", linestyle = "--", label = f"Stop (Tol={500})")
    plt.scatter([stop_i_gd], [stop_a_gd], color="black", zorder=5)
    plt.text(stop_i_gd + 10, stop_a_gd, f"{stop_a_gd:.2f}%", fontweight = "bold")

plt.title("Gradient Descent: Accuracy vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()


# Plot Randomized BCGD
plt.figure(figsize = (8, 5))
plt.plot(iter_rand, acc_rand, color = "red", label = "Test Accuracy")

if stop_i_rand is not None:
    plt.axvline(x = stop_i_rand, color = "black", linestyle = "--", label = f"Stop (Tol={500})")
    plt.scatter([stop_i_rand], [stop_a_rand], color = "black", zorder = 5)
    plt.text(stop_i_rand + 50, stop_a_rand, f"{stop_a_rand:.2f}%", fontweight = "bold")

plt.title("Randomized BCGD: Accuracy vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()


# Plot Gauss-Southwell BCGD
plt.figure(figsize = (8, 5))
plt.plot(iter_gs, acc_gs, color = "green", label = "Test Accuracy")

if stop_i_gs is not None:
    plt.axvline(x = stop_i_gs, color = "black", linestyle = "--", label = f"Stop (Tol={500})")
    plt.scatter([stop_i_gs], [stop_a_gs], color="black", zorder=5)
    plt.text(stop_i_gs + 30, stop_a_gs, f"{stop_a_gs:.2f}%", fontweight = "bold")

plt.title("Gauss-Southwell BCGD: Accuracy vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()




# Plot accuracy vs tolerance
tollist = [100, 200, 300, 400, 500, 600]

# list to store the accuracy reached for the plots
acc_gd, acc_bcgd, acc_gs = [], [], []

# trying every tolerance in the list to store the data for the plots (for every algorithm)
for tolerance in tollist:
    _, _, _, _, stop_gd= gradient_descent(A_train,
                                                I_train,
                                                h = 50,
                                                n_iter_gd = 1000,
                                                tol = tolerance)

    _, _, _, _, stop_rand = randomized_BCGD(A_train,
                                                I_train,
                                                n_iter = 15000,
                                                h = 5,
                                                tol = tolerance)

    _, _, _, _, stop_gs = gauss_southwell_BCGD(A_train,
                                                    I_train,
                                                    n_iter = 15000,
                                                    h = 9,
                                                    tol = tolerance)

    acc_gd.append(stop_gd)
    acc_bcgd.append(stop_rand)
    acc_gs.append(stop_gs)

# drawing the 3 lines overlapped to compare them
plt.figure()

plt.plot(tollist,
        acc_gd,
        label = "GD",
        marker = "s",
        color = "blue")

plt.plot(tollist,
         acc_bcgd,
         label = "Randomized BCGD",
         marker = "^",
         color = "red")

plt.plot(tollist, acc_gs,
         label = "Gauss-Southwell BCGD",
         marker = "o",
         color = "green")

plt.title("accuracy vs tolerance")
plt.xlabel("epsilon (Ɛ)")
plt.ylabel("Accuracy (%)")

plt.gca().invert_xaxis()

plt.grid(True)
plt.legend()

# printing the accuracy at epsilon = 500 
plt.text(500, acc_gd[4] + 0.1,
        f"{acc_gd[4]:.2f}%",
        color = "blue")

plt.text(500, acc_bcgd[4] - 0.2,
        f"{acc_bcgd[4]:.2f}%",
        color = "red")

plt.text(500, acc_gs[4] - 0.2,
        f"{acc_gs[4]:.2f}%",
        color = ("green"))

plt.show()