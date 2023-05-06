import numpy as np
import control
import matplotlib.pyplot as plt


class DroneController:
    def __init__(self):
        # Define system matrices
        self.m = 0.5   # mass of drone (kg)
        self.g = 9.81  # gravitational acceleration (m/s^2)
        self.l = 0.25  # length of arm (m)
        self.I = 0.1   # moment of inertia (kg.m^2)
        
        self.A = np.array([
            [0, 0, 10, 0],
            [0, 0, 0, 1],
            [0, -self.g, 0, 0],
            [0, 0, 0, 0]
        ])
        self.B = np.array([
            [0],
            [0],
            [0],
            [1/self.m]
        ])
        self.C = np.eye(4)
        self.D = np.zeros((4, 1))
        
        # Define weighting matrices
        self.Q = np.diag([1000, 100, 10, 1])
        self.R = np.array([[1]])
        
        # Compute LQR gain matrix
        self.K, _, _ = control.lqr(self.A, self.B, self.Q, self.R)
        
        # Define initial state and time vector
        self.x0 = np.array([0, 0, 1, 0])
        self.t = np.linspace(0, 10, 1000)
        
    def simulate(self, perturbation=None):
        # # Simulate closed-loop system with LQR controller and perturbation
        # sys = control.StateSpace(self.A - self.B @ self.K, self.B, self.C, self.D)
        # u = np.zeros((1, len(self.t)))
        # if perturbation is not None:
        #     u = perturbation.reshape(1, -1)
        # #_, y, _ = control.forced_response(sys, T=self.t, U=u, X0=self.x0)
        # t, y = control.forced_response(sys, T=self.t, U=u, X0=self.x0)

        alt_ref = np.zeros_like(drone.t)
        #alt_ref[-1] = 1

        
        
        sys = control.StateSpace(self.A - self.B @ self.K, self.B, self.C, self.D)
        

        # Simulate closed-loop system with LQR controller and perturbation
        #sys = control.StateSpace(self.A - self.B @ self.K, self.B, self.C, self.D)
        u = np.zeros((1, len(self.t)))
        if perturbation is not None:
            u = perturbation.reshape(1, -1)
        #_, y, _ = control.forced_response(sys, T=self.t, U=u, X0=self.x0)
        t, y = control.forced_response(sys, T=self.t, U=u, X0=self.x0)
        #t, y = control.forced_response(sys, T=self.t, U=alt_ref.reshape(1, -1), X0=self.x0)
        return y
    
    def plot_results(self, y):
        # Plot results
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        axs[0, 0].plot(self.t, y[0])
        axs[0, 0].set_title("Altitude (m)")
        axs[0, 1].plot(self.t, y[1])
        axs[0, 1].set_title("Pitch Angle (rad)")
        axs[1, 0].plot(self.t, y[2])
        axs[1, 0].set_title("Velocity (m/s)")
        axs[1, 1].plot(self.t, y[3])
        axs[1, 1].set_title("Pitch Rate (rad/s)")
        for ax in axs.flat:
            ax.grid(True, color='r', linestyle='--') 
        #axs.grid(True, color='r', linestyle='--')        
        fig.tight_layout()
        plt.show()


# Create instance of DroneController
drone = DroneController()

# Simulate system without perturbation
y = drone.simulate()
drone.plot_results(y)

# Simulate system with sinusoidal perturbation
# perturbation = np.sin(0.5 * drone.t)  # example sinusoidal perturbation
# y = drone.simulate(perturbation=perturbation)
drone.plot_results(y)
