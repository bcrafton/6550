import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

A = np.array([[-2., 2., -4.], [2., -1., -10.], [-0.5, 0.0, 0.0]])
B = np.array([2., 1., 0.5]).T

# A = np.array([[1., 1., -2.], [-1., -1., 1.], [-2., -1., 1.]])
# B = np.array([2., -1., -2.]).T

##########################################

T = 1.
dt = 0.001
steps = int(T / dt)
Ts = np.linspace(0, T, steps)

X0 = np.array([0., 0., 0.])
Xs = [X0]

for step in range(steps):
    t = Ts[step]
    p1 = np.dot(A, Xs[-1])
    p2 = B * np.random.uniform(-1., 1.)
    dxdt = p1 + p2
    dx = dxdt * dt
    X = Xs[-1] + dx
    Xs.append(X)
##########################################

Xs = np.array(Xs)
print ('rank', np.linalg.matrix_rank(Xs))

# plt.plot(Xs.T[0], Xs.T[1], Xs.T[2])
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xs.T[0], Xs.T[1], Xs.T[2])
plt.show()

##########################################
