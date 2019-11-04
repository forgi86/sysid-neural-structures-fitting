import control
import numpy as np

class LinearStateSpaceSystem:
    def __init__(self, A, B, C, D=None, x0=None):
        if x0 is None:
            x0 = np.zeros(A.shape[0])
        self.x = np.copy(x0)
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        self.D = np.array(D)

    def output(self,u=None):
        self.y = self.C @ self.x
        if u is not None and self.D is not None:
            u = np.array(u).ravel()
            self.y += self.D @ u
        return self.y

    def update(self, u):
        u = np.array(u).ravel()
        self.x = self.A @ self.x + self.B @ u

if __name__ == '__main__':

    Ts = 0.1
    nx = 2
    nu = 1
    ny = 1

    Ac = np.eye(2)
    Bc = np.ones((2,1))
    Cd = np.eye(2)
    Dc = np.zeros((2,1))

    Ad = np.eye(nx) + Ac*Ts
    Bd = Bc*Ts

    # Default controller parameters -
    K_NUM = [-2100, -10001, -100]
    K_DEN = [1,   100,     0]

    Ts = 1e-3
    K = control.tf(K_NUM,K_DEN)
    Kd_tf = control.c2d(K, Ts)
    Kd_ss = control.ss(Kd_tf)
    Kd = LinearStateSpaceSystem(A=Kd_ss.A, B=Kd_ss.B, C=Kd_ss.C, D=Kd_ss.D)


    P = -100.01
    I = -1
    D = -20
    N = 100.0

    kP = control.tf(P,1, Ts)
    kI = I*Ts*control.tf([0, 1], [1,-1], Ts)
    kD = D*control.tf([N, -N], [1.0, Ts*N -1], Ts)
    kPID = kP + kD + kI


