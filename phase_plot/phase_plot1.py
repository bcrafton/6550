import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def fix(mat):
    u = 0.5 * (vec[:, 0] + vec[:, 1]).real
    w = 0.5 * (vec[:, 0] - vec[:, 1]).imag
    return np.array([u, w]).T

def pretty(mat):
    shape = np.shape(mat)
    m, n = shape
    
    for ii in range(n):
        if (np.min(np.abs(mat[:, ii])) > 0):
            mat[:, ii] = mat[:, ii] * (1.0 / np.min(np.abs(mat[:, ii])))
        else:
            mat[:, ii] = mat[:, ii] * (1.0 / np.max(np.abs(mat[:, ii])))
            
        if (np.sign(mat[0, ii]) < 0):
            mat[:, ii] = mat[:, ii] * np.sign(mat[:, ii])
    
    return mat

##########################################

# A = np.array([[1.0, -4.0], [1.0, 1.0]])
# A = np.array([[-1.0/3.0, -8.0/3.0], [5.0/3.0, -5.0/3.0]])
# A = np.array([[-5.0, -10.0], [4.0, -1.0]])
# A = np.array([[2.0, 0.0], [3.0, -1.0]])
A = np.array([[-5.0/3.0, -2.0/3.0], [-1.0/3.0, -4.0/3.0]])

##########################################

val, vec = np.linalg.eig(A)

vec = pretty(vec)

invvec = pretty(np.linalg.inv(vec))

val = val * np.eye(2)
# val = np.dot(invvec, np.dot(A, vec))

##########################################

vec = np.array([[2.0, 1.0], [1.0, -1.0]])
val = np.array([[-2.0, 0.0], [0.0, -1.0]])
invvec = pretty(np.linalg.inv(vec))

##########################################

print (val)
print (vec)
print (invvec)

##########################################

T = 1.
dt = 0.001
steps = int(T / dt)
Ts = np.linspace(0, T, steps)

NUM_POINTS = 100

X0s = []
Xs = []       
for ii in range(NUM_POINTS):
    X0 = np.random.uniform(low=-1, high=1, size=(2))
    # X0 = np.dot(vec, np.random.uniform(low=-1, high=1, size=(2)))
    X0s.append(X0)
    Xs.append([])

for step in range(steps):
    t = Ts[step]
    
    # eat = np.dot(invvec, np.dot(la.expm(val * t), vec))
    # elt = np.dot(vec, np.dot(eat, invvec))
    
    elt = la.expm(val * t)
    
    for ii in range(NUM_POINTS):
        X = np.dot(elt, X0s[ii])
        Xs[ii].append(X)

##########################################

points = np.linspace(-100, 100, 100)

print ("basis 0: ", vec[0][0], vec[1][0])
basis0_x = points * vec[0][0]
basis0_y = points * vec[1][0]

print ("basis 1: ", vec[0][1], vec[1][1])
basis1_x = points * vec[0][1]
basis1_y = points * vec[1][1]

##########################################3
# imaginary warning happens while plotting.

for ii in range(NUM_POINTS):
    pts = np.transpose(Xs[ii])
    x = pts[0]
    y = pts[1]
    plt.plot(x, y, '.')

plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()

##########################################
