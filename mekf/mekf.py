""" Class for Multi Extended Kalman Filter implementation."""

# %% Import libraries

import numpy as np
from scipy.linalg import expm


def discretize(A: list, B: list, ts: float) -> tuple[list, list]:
    """Discretize LTI systems.

    Parameters
    ----------
    A : list
        Dynamic matric of the continuous system dx/dt = Ax + Bu.
    B : list
        Input matric of the continuous system dx/dt = Ax + Bu.
    ts : float, optional
        Sampling time of the system. The default is 1.

    Returns
    -------
    Ad : list
        Dynamic matric of the discret system x+ = Adx + Bdu.
    Bd : list
        Input matric of the discret system x+ = Adx + Bdu.

    """
    (n, m) = B.shape
    Mt = np.zeros((n+m, n+m))
    Mt[0:n, 0:n] = A
    Mt[0:n, n:n+m] = B
    Mtd = expm(Mt*ts)
    Ad = Mtd[0:n, 0:n]
    Bd = Mtd[0:n, n:n+m]
    return Ad, Bd


def derivated_of_f(x: list, bis_param: list) -> list:
    """Compute the derivated of the non-linear function BIS.

    Parameters
    ----------
    x : list
        State vector [xep, xer].
    bis_param : list
        Parameters of the non-linear function BIS_param = [C50p, C50r, gamma, beta, E0, Emax].

    Returns
    -------
    list
        Derivated of the non-linear function BIS.

    """

    C50p = bis_param[0]
    C50r = bis_param[1]
    gamma = bis_param[2]
    beta = bis_param[3]
    Emax = bis_param[5]
    df = np.zeros((1, 8))
    up = x[0] / C50p
    ur = x[1] / C50r
    Phi = up/(up + ur + 1e-6)
    U50 = 1 - beta * (Phi - Phi**2)
    I = (up + ur)/U50
    dPhi_dxep = 1/(C50p*(up + ur + 1e-6)**2)
    dPhi_dxer = 1/(C50r*(up + ur + 1e-6)**2)
    dU50_dxep = beta*(1 - 2*Phi)*dPhi_dxep
    dU50_dxer = beta*(1 - 2*Phi)*dPhi_dxer
    dI_dxep = (U50/C50r-(up + ur) * dU50_dxep)/U50**2
    dI_dxer = (U50/C50p-(up + ur) * dU50_dxer)/U50**2

    dBIS_dxep = -Emax*gamma*I**(gamma-1)*dI_dxep/(1+I**gamma)**2
    dBIS_dxer = -Emax*gamma*I**(gamma-1)*dI_dxer/(1+I**gamma)**2
    df[0, 3] = dBIS_dxep
    df[0, 7] = dBIS_dxer
    return df


def BIS(xep: float, xer: float, Bis_param: list) -> float:
    """
    Compute the non-linear output function.

    Parameters
    ----------
    xep : float
        Propofol concentration in the effect site.
    xer : float
        Remifentanil concentration in the effect site.
    Bis_param : list
        Parameters of the non-linear function BIS_param = [C50p, C50r, gamma, beta, E0, Emax].

    Returns
    -------
    BIS : float
        BIS value associated to the concentrations.

    """
    C50p = Bis_param[0]
    C50r = Bis_param[1]
    gamma = Bis_param[2]
    beta = Bis_param[3]
    E0 = Bis_param[4]
    Emax = Bis_param[5]
    up = xep / C50p
    ur = xer / C50r
    Phi = up/(up + ur + 1e-6)
    U_50 = 1 - beta * (Phi - Phi**2)
    i = (up + ur)/U_50
    BIS = E0 - Emax * i ** gamma / (1 + i ** gamma)
    return BIS


class EKF:
    """Implementation of the Extended Kalman Filter for the Coadministration of drugs in Anesthesia."""

    def __init__(self, A: list, B: list, BIS_param: list, ts: float, x0: list = np.zeros((8, 1)),
                 Q: list = np.eye(8), R: list = np.array([1]), P0: list = np.eye(8)):
        """
        Init the EKF class.

        Parameters
        ----------
        A : list
            Dynamic matric of the continuous system dx/dt = Ax + Bu.
        B : list
            Input matric of the continuous system dx/dt = Ax + Bu.
        BIS_param : list
            Contains parameters of the non-linear function output BIS_param = [C50p, C50r, gamma, beta, E0, Emax]
        ts : float, optional
            Sampling time of the system. The default is 1.
        x0 : list, optional
            Initial state of the system. The default is np.zeros((8, 1)).
        Q : list, optional
            Covariance matrix of the process uncertainties. The default is np.eye(8).
        R : list, optional
            Covariance matrix of the measurement noises. The default is np.array([1]).
        P0 : list, optional
            Initial covariance matrix of the state estimation. The default is np.eye(8).

        Returns
        -------
        None.

        """
        self.Ad, self.Bd = discretize(A, B, ts)
        self.ts = ts
        self.BIS_param = BIS_param

        self.R = R
        self.Q = Q
        self.P = P0

        # init state and output
        self.x = x0
        self.Biso = [BIS(self.x[3], self.x[7], BIS_param)]
        self.error = 0

    def estimate(self, u: list, bis: float) -> tuple[list, float]:
        """
        Estimate the state given past input and current measurement.

        Parameters
        ----------
        u : list
            Last control inputs.
        bis : float
            Last BIS measurement.

        Returns
        -------
        x: list
            State estimation.
        BIS: float
            Filtered BIS.

        """
        # prediction
        self.x = self.Ad @ self.x + self.Bd @ u
        self.P = self.Ad @ self.P @ self.Ad.T + self.Q

        # correction
        self.error = bis - BIS(self.x[3], self.x[7], self.BIS_param)
        H = derivated_of_f(self.x, self.BIS_param)
        self.K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + self.R)
        self.x = self.x + self.K * self.error
        self.P = (np.eye(8) - self.K @ H) @ self.P

        # output
        self.bis = BIS(self.x[3], self.x[7], self.BIS_param)

        return self.x, self.bis


class MEKF:
    """Multi Extended Kalman Filter for estimation of the PD parameters in TIVA anesthesia.

    Parameters
    ----------
    A : list
        Dynamic matrix of the continuous system dx/dt = Ax + Bu.
    B : list
        Input matrix of the continuous system dx/dt = Ax + Bu.
    grid_vector : list
        Contains a drif of parameters of the non-linear function output BIS_param = [C50p, C50r, gamma, beta, E0, Emax].
    ts : float, optional
        Sampling time of the system. The default is 1.
    x0 : list, optional
        Initial state of the system. The default is np.zeros((8, 1)).
    Q : list, optional
        Covariance matrix of the process uncertainties. The default is np.eye(8).
    R : list, optional
        Covariance matrix of the measurement noises. The default is np.array([1]).
    P0 : list, optional
        Initial covariance matrix of the state estimation. The default is np.eye(8).
    eta0 : list, optional
        Initial state of the parameter estimation. The default is np.ones((6, 1)).
    design_param : list, optional
        Design parameters of the system [lambda_1, lambda_2, nu, epsilon]. The default is [1, 1, 0.1, 0.9].
    """

    def __init__(self, A: list, B: list, grid_vector: list, ts: float = 1,
                 x0: list = np.zeros((8, 1)), Q: list = np.eye(8),
                 R: list = np.array([1]), P0: list = np.eye(8),
                 eta0: list = None, design_param: list = [1, 1, 0.1]) -> None:
        """Init the MEKF class."""
        self.ts = ts

        # define the set of EKF
        self.EKF_list = []
        for BIS_param in grid_vector:
            self.EKF_list.append(EKF(A, B, BIS_param, ts, x0, Q, R, P0))

        # Init the criterion
        self.grid_vector = grid_vector
        if eta0 is None:
            self.eta = np.ones((len(self.EKF_list), 1))
        self.eta = eta0
        self.best_index = np.argmin(eta0)

        # define the design parameters
        self.lambda_1 = design_param[0]
        self.lambda_2 = design_param[1]
        self.nu = design_param[2]
        self.epsilon = design_param[3]

    def one_step(self, u: list, measurement: float) -> tuple[list, float]:
        """
        Estimate the state given past input and current measurement.

        Parameters
        ----------
        u : list
            Last control inputs.
        measurement : float
            Last BIS measurement.

        Returns
        -------
        x: list
            State estimation.
        BIS: float
            Filtered BIS.
        best_index: int
            Index of the best EKF.
        """
        # estimate the state for each EKF
        for i, ekf in enumerate(self.EKF_list):
            ekf.estimate(u, measurement)
            error = measurement - ekf.bis
            K_i = np.array(ekf.K)
            self.eta[i] += self.ts*(-self.nu*self.eta[i] + self.lambda_1 * error **
                                    2 + self.lambda_2 * (K_i.T @ K_i) * error**2)

        # compute the criterion
        possible_best_index = np.argmin(self.eta)
        if self.eta[possible_best_index] < self.epsilon * self.eta[self.best_index]:
            self.best_index = possible_best_index

        return self.EKF_list[self.best_index].x, self.EKF_list[self.best_index].bis, self.best_index
