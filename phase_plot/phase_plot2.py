import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def fix(mat):
    u = 0.5 * (vec[:, 0] + vec[:, 1]).real
    w = 0.5 * (vec[:, 0] - vec[:, 1]).imag
    return np.array([u, w]).T

def pretty(mat):
    scale = np.max(np.abs(mat))
    mat = (1.0) / scale * mat

    if np.sign(mat[0][0]) < 0:
        mat[:, 0] = mat[:, 0] * np.sign(mat[0][0])

    if np.sign(mat[0][1]) < 0:
        mat[:, 1] = mat[:, 1] * np.sign(mat[0][1])
    
    return mat
    
##########################################

# A = np.array([[1.0, -4.0], [1.0, 1.0]])
A = np.array([[-1.0/3.0, -8.0/3.0], [5.0/3.0, -5.0/3.0]])
# A = np.array([[-5.0, -10.0], [4.0, -1.0]])

val, vec = np.linalg.eig(A)

vec = pretty(fix(vec))
invvec = np.linalg.inv(vec)

# val = val * np.eye(2)
val = np.dot(invvec, np.dot(A, vec))

from_notes = 0
if from_notes:
    vec = np.array([[1.0, 0.0], [-0.2, 0.6]])
    invvec = np.linalg.inv(vec)
    val = np.array([[-3.0, -6.0], [6.0, -3.0]])
    # val = np.dot(invvec, np.dot(A, vec))

print (val)
print (vec)
print (invvec)

##########################################

T = 1.0
dt = 0.001
steps = int(T / dt)
Ts = np.linspace(0, T, steps)

NUM_POINTS = 100

X0s = []
Xs = []       
for ii in range(NUM_POINTS):
    X0s.append(np.random.uniform(low=-1.0, high=1.0, size=(2)))
    Xs.append([])

for step in range(steps):
    t = Ts[step]
    
    eat = np.dot(vec, np.dot(la.expm(val * t), invvec))
    
    for ii in range(NUM_POINTS):
        X = np.dot(eat, X0s[ii])
        Xs[ii].append(X)

##########################################
        
points = np.linspace(-100, 100, 100)

basis0_x = points * vec[0][0]
basis0_y = points * vec[0][1]

basis1_x = points * vec[1][0]
basis1_y = points * vec[1][1]

##########################################3
# imaginary warning happens while plotting.
plt.plot(basis0_x, basis0_y)
plt.plot(basis1_x, basis1_y)

for ii in range(NUM_POINTS):
    pts = np.transpose(Xs[ii])
    x = pts[0]
    y = pts[1]
    plt.plot(x, y, '.')

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()

##########################################
