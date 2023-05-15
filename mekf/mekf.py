""" Class for Multi Extended Kalman Filter implementation."""

# %% Import libraries

import numpy as np
import casadi as cas
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

        # declare CASADI variables
        x = cas.MX.sym('x', 8)  # x1p, x2p, x3p, xep, x1r, x2r, x3r, xer [mg/ml]
        y = cas.MX.sym('y')  # BIS [%]
        prop = cas.MX.sym('prop')   # Propofol infusion rate [mg/ml/min]
        rem = cas.MX.sym('rem')   # Remifentanil infusion rate [mg/ml/min]
        u = cas.vertcat(prop, rem)
        P = cas.MX.sym('P', 8, 8)   # P matrix
        Pup = cas.MX.sym('P', 8, 8)   # P matrix

        # declare CASADI functions
        xpred = cas.MX(self.Ad) @ x + cas.MX(self.Bd) @ u
        Ppred = cas.MX(self.Ad) @ P @ cas.MX(self.Ad.T) + cas.MX(self.Q)
        self.Pred = cas.Function('Pred', [x, u, P], [xpred, Ppred], [
                                 'x', 'u', 'P'], ['xpred', 'Ppred'])

        h_fun = BIS(x[3], x[7], self.BIS_param)
        self.output = cas.Function('output', [x], [h_fun], ['x'], ['bis'])

        H = cas.gradient(h_fun, x).T

        S = H @ P @ H.T + cas.MX(self.R)
        K = P @ H.T @ cas.inv(S)

        error = y - h_fun
        xup = x + K @ error
        Pup = (cas.MX(np.identity(8)) - K@H)@P
        self.Update = cas.Function('Update', [x, y, P], [xup, Pup, error, S, K], [
                                   'x', 'y', 'P'], ['xup', 'Pup', 'error', 'S', 'K'])

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
        self.Predk = self.Pred(x=self.x, u=u, P=self.P)
        self.xpr = self.Predk['xpred'].full().flatten()
        self.Ppr = self.Predk['Ppred'].full()

        self.Updatek = self.Update(x=self.xpr, y=bis, P=self.Ppr)
        self.x = self.Updatek['xup'].full().flatten()
        self.P = self.Updatek['Pup']
        self.error = float(self.Updatek['error'])
        self.bis_pred = bis - self.error
        self.S = self.Updatek['S']
        self.K = self.Updatek['K']

        # self.x[3] = max(1e-3, self.x[3])
        # self.x[7] = max(1e-3, self.x[7])
        self.x = np.clip(self.x, a_min=1e-3, a_max=None)
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
            K_i = ekf.K
            self.eta[i] += self.ts*(-self.nu*self.eta[i] + self.lambda_1 * error**2 + self.lambda_2 * (K_i @ error)**2)

        # compute the criterion
        possible_best_index = np.argmin(self.eta)
        print(possible_best_index)
        if self.eta[possible_best_index] < self.epsilon * self.eta[self.best_index]:
            self.best_index = possible_best_index

        return self.EKF_list[self.best_index].x, self.EKF_list[self.best_index].Bis, self.best_index
