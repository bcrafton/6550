import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

A = np.array([[3.0, 2.0], [-1.0, 0.0]])
#A = np.array([[-1.0/3.0, -8.0/3.0], [5.0/3.0, -5.0/3.0]])
val, vec = np.linalg.eig(A)
val = val * np.eye(2)
invvec = np.linalg.inv(vec)

print (val)
print (vec)

##########################################

T = 1.0
dt = 0.001
steps = int(T / dt)
Ts = np.linspace(0, T, steps)

NUM_POINTS = 100

X0s = []
Xs = []       
for ii in range(NUM_POINTS):
    X0s.append(np.random.uniform(low=-0.5, high=0.5, size=(2)))
    Xs.append([])

for step in range(steps):
    t = Ts[step]
    
    elt = np.dot(invvec, np.dot(la.expm(A * t), vec))
    
    for ii in range(NUM_POINTS):
        X = np.dot(elt, X0s[ii])
        Xs[ii].append(X)

##########################################
        
vec0 = vec[:, 0]
slope0 = vec0[0] / vec0[1]
basis0 = np.linspace(-5, 5, 100) * slope0

vec1 = vec[:, 1]
slope1 = vec1[0] / vec1[1]
basis1 = np.linspace(-5, 5, 100) * slope1

##########################################3
# imaginary warning happens while plotting.

for ii in range(NUM_POINTS):
    pts = np.transpose(Xs[ii])
    x = pts[0]
    y = pts[1]
    plt.plot(x, y, '.')

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()

##########################################
