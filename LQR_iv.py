
import math
import time


import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, eig
import numpy as np

# Model parameters

l_bar = 3.0  # length of bar
M = 3.0  # [kg]
m = 0.7  # [kg]
g = 9.8  # [m/s^2]

nx = 4  # number of state
nu = 1  # number of input
Q = np.diag([0.0, 1.0, 1.0, 0.0])  # state cost matrix
R = np.diag([0.01])  # input cost matrix

delta_t = 0.05  # time tick [s]
sim_time = 5.0  # simulation time [s]

 


def main():
    x0 = np.array([
        [0.8],
        [0.0],
        [1.0],
        [0.0]
    ])

    time_list = []
    x_list = []
    theta_list = []

    x = np.copy(x0)
    time = 0.0

    while sim_time > time:
        time += delta_t
        u = lqr_control(x)
        x = simulation(x, u)

        time_list.append(time)
        x_list.append(float(x[0]))
        theta_list.append(float(x[2]))

        print(u)
        print(x)


    print("Finish")
    print(f"x={float(x[0]):.2f} [m] , theta={math.degrees(x[2]):.2f} [deg]")


    plt.plot(time_list, x_list, label="x")
    plt.plot(time_list, theta_list, label="theta")
    plt.axhline(y = 0, color = 'y', linestyle = '--')
    plt.xlabel("Time [s]")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Inverted Pendulum LQR Control")
    plt.grid(True)
    plt.show()


def get_model_matrix():
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, m * g / M, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g * (M + m) / (l_bar * M), 0.0]
    ])
    A = np.eye(nx) + delta_t * A

    B = np.array([
        [0.0],
        [1.0 / M],
        [0.0],
        [1.0 / (l_bar * M)]
    ])
    B = delta_t * B

    return A, B

def simulation(x, u):
    A, B = get_model_matrix()
    x = A @ x + B @ u

    return x

def get_numpy_array_from_matrix(x):
    return np.array(x).flatten()


def flatten(a):
    return np.array(a).flatten()
    
def solve_DARE(A, B, Q, R, maxiter=150, eps=0.01):
    P = Q
    for i in range(maxiter):
        Pn = A.T @ P @ A - A.T @ P @ B @ \
            inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
        if (abs(Pn - P)).max() < eps:
            break
        P = Pn
    return Pn


def dlqr(A, B, Q, R):
    P = solve_DARE(A, B, Q, R)
    K = inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    eigVals, eigVecs = eig(A - B @ K)
    return K, P, eigVals


def lqr_control(x):
    A, B = get_model_matrix()
    start = time.time()
    K, _, _ = dlqr(A, B, Q, R)
    u = -K @ x
    elapsed_time = time.time() - start
    print(f"calc time:{elapsed_time:.6f} [sec]")
    return u





if __name__ == '__main__':
    main()