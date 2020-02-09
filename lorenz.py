import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

ro = 28.0
sigma = 10.0
beta = 8.0 / 3.0

import numpy as np
import matplotlib.pyplot as plt


def lorenz(x, y, z, s=10, r=28, b=2.667):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


dt = 0.01
num_steps = 15000

# Need one more for the initial values
xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)

# Valores iniciales de Lorenz
xs[0], ys[0], zs[0] = (0., 1., 0.)

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)


# Plot
fig = plt.figure()
ax = fig.gca(projection='3d')

# En 3d (1400 a 1900)
ax.plot(xs[1400:1900], ys[1400:1900], zs[1400:1900], lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")
plt.show()

# En 3d (0 a 15000)
ax.plot(xs, ys, zs, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")
plt.show()


################################
# Inestabilidad
# Y EN 1000 iteraciones
t = np.arange(0.0, 1000.0, 1.0)
s = ys[0:1000]
fig, ax = plt.subplots()
ax.plot(t, s)
ax.set(xlabel='iterations', ylabel='Y',
       title='Y en 1000 iteraciones')
ax.grid()
plt.show()

# Y EN 1000/2000 iteraciones
t = np.arange(1000.0, 2000.0, 1.0)
s = ys[1000:2000]
fig, ax = plt.subplots()
ax.plot(t, s)
ax.set(xlabel='iterations', ylabel='Y',
       title='Y en 1000/2000 iteraciones')
ax.grid()
plt.show()

# Y EN 2000/3000 iteraciones
t = np.arange(2000.0, 3000.0, 1.0)
s = ys[2000:3000]
fig, ax = plt.subplots()
ax.plot(t, s)
ax.set(xlabel='iterations', ylabel='Y',
       title='Y en 2000/3000 iteraciones')
ax.grid()
plt.show()

# Y EN 3000/15000 iteraciones
t = np.arange(3000.0, 15000.0, 1.0)
s = ys[3000:15000]
fig, ax = plt.subplots()
ax.plot(t, s)
ax.set(xlabel='iterations', ylabel='Y',
       title='Y en 3000/15000 iteraciones')
ax.grid()
plt.show()


# Y EN 0 a 15000 iteraciones
t = np.arange(0.0, 15000.0, 1.0)
s = ys[0:15000]
fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='iterations', ylabel='Y',
       title='Y en 0 a 15000 iteraciones')
ax.grid()
plt.show()

Idas y vueltas en 3d
En 3d (3000 a 15000)
ax.plot(xs[3000:15000], ys[3000:15000], zs[3000:15000], lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")
plt.show()