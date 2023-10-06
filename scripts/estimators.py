import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import casadi as cas
import control as ctrl


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
    # if len(x) == 8:
    C50p = bis_param[0]
    C50r = bis_param[1]
    gamma = bis_param[2]

    # elif len(x) == 11:
    #     C50p = x[8]
    #     C50r = x[9]
    #     gamma = x[10]
    #     df = np.zeros((1, 11))

    beta = bis_param[3]
    Emax = bis_param[5]

    up = x[3] / C50p
    ur = x[7] / C50r
    Phi = up/(up + ur + 1e-6)
    U50 = 1 - beta * (Phi - Phi**2)
    I = (up + ur)/U50
    dup_dxp = 1/C50p
    dur_dxr = 1/C50r
    dPhi_dxep = dup_dxp * ur/(up + ur + 1e-6)**2
    dPhi_dxer = -dur_dxr * up/((up + ur + 1e-6)**2)
    dU50_dxep = beta*(1 - 2*Phi)*dPhi_dxep
    dU50_dxer = beta*(1 - 2*Phi)*dPhi_dxer
    dI_dxep = (dup_dxp*U50 - dU50_dxep*(up+ur))/U50**2
    dI_dxer = (dur_dxr*U50 - dU50_dxer*(up+ur))/U50**2

    dBIS_dxep = -Emax*gamma*I**(gamma-1)*dI_dxep/(1+I**gamma)**2
    dBIS_dxer = -Emax*gamma*I**(gamma-1)*dI_dxer/(1+I**gamma)**2
    df = cas.hcat([0, 0, 0, dBIS_dxep, 0, 0, 0, dBIS_dxer])

    # if len(x) == 11:
    #     dup_dc50p = -x[3]/C50p**2
    #     dur_dc50r = -x[7]/C50r**2
    #     dPhi_dc50p = (dup_dc50p*ur)/(up + ur + 1e-6)**2
    #     dPhi_dc50r = -(dur_dc50r*up)/(up + ur + 1e-6)**2
    #     dU50_dc50p = beta*(1 - 2*Phi)*dPhi_dc50p
    #     dU50_dc50r = beta*(1 - 2*Phi)*dPhi_dc50r
    #     dI_dc50p = (dup_dc50p*U50 - (up + ur)*dU50_dc50p)/U50**2
    #     dI_dc50r = (dur_dc50r*U50 - (up + ur)*dU50_dc50r)/U50**2
    #     dBIS_dc50p = -Emax*gamma*I**(gamma-1)*dI_dc50p/(1+I**gamma)**2
    #     dBIS_dc50r = -Emax*gamma*I**(gamma-1)*dI_dc50r/(1+I**gamma)**2
    #     dBIS_gamma = -Emax*I**gamma*np.log(I)/(1+I**gamma)**2
    #     df[0, 8] = dBIS_dc50p
    #     df[0, 9] = dBIS_dc50r
    #     df[0, 10] = dBIS_gamma
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

    def __init__(self, A: list, B: list, BIS_param: list, ts: float, x0: list = np.ones((8, 1))*1.e-3,
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
        C50p = BIS_param[0]
        C50r = BIS_param[1]
        gamma = BIS_param[2]
        beta = BIS_param[3]
        E0 = BIS_param[4]
        Emax = BIS_param[5]

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

        up = x[3] / C50p
        ur = x[7] / C50r
        Phi = up/(up + ur + 1e-6)
        U_50 = 1 - beta * (Phi - Phi**2)
        i = (up + ur)/U_50

        h_fun = E0 - Emax * i ** gamma / (1 + i ** gamma)
        self.output = cas.Function('output', [x], [h_fun], ['x'], ['bis'])

        H = derivated_of_f(x, self.BIS_param)
        # cas.gradient(h_fun, x).T

        S = H @ P @ H.T + cas.MX(self.R)
        K = P @ H.T @ cas.inv(S)

        error = y - h_fun
        xup = x + K @ error
        Pup = (cas.MX(np.identity(8)) - K@H)@P
        # S = H @ P0 @ H.T + cas.MX(self.R)
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
        self.P = self.Updatek['Pup'].full()
        self.K = self.Updatek['K'].full()
        self.error = float(self.Updatek['error'])
        self.bis_pred = bis - self.error
        self.S = self.Updatek['S']

        # self.x[3] = max(1e-3, self.x[3])
        # self.x[7] = max(1e-3, self.x[7])
        self.x = np.clip(self.x, a_min=1e-3, a_max=None)
        self.bis = BIS(self.x[3], self.x[7], self.BIS_param)

        return self.x, self.bis

    def predict_from_state(self, x: list, up: list, ur: list) -> list:
        """
        Return the BIS prediction using the given initial state and the control input.

        Parameters
        ----------
        x : list
            Initial state vector of the interval.
        up : list
            Propofol rates over the interval.
        ur : list
            Remifentanil rates over the interval.

        Returns
        -------
        BIS_list: list
            BIS value predicted by the model over the interval.

        """
        bis = self.output(x=x)
        BIS_list = [float(bis['bis'])]
        x = np.expand_dims(x, axis=1)
        for i in range(len(up)):
            u = np.array([[up[i]], [ur[i]]])
            x = self.Ad @ x + self.Bd @ u
            bis = self.output(x=x)
            BIS_list.append(float(bis['bis']))

        return np.array(BIS_list)

    def find_from_state(self, x: list, up: list, ur: list) -> list:
        """
        Return the BIS prediction using the given initial state and the control input.

        Parameters
        ----------
        x : list
            Initial state vector of the interval.
        up : list
            Propofol rates over the interval.
        ur : list
            Remifentanil rates over the interval.

        Returns
        -------
        BIS_list: list
            BIS value predicted by the model over the interval.

        """
        invA = np.linalg.inv(self.Ad)
        bis = self.output(x=x)
        BIS_list = [float(bis['bis'])]
        x = np.expand_dims(x, axis=1)
        for i in range(len(up)):
            u = np.array([[up[len(up)-i-1]], [ur[len(up)-i-1]]])
            x = invA @ (x - self.Bd @ u)
            bis = self.output(x=x)
            BIS_list = [float(bis['bis'])] + BIS_list

        return np.array(BIS_list)


class MEKF_Petri:
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
                 x0: list = np.ones((8, 1))*1.e-3, Q: list = np.eye(8),
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
            # init the criterion again
            # self.eta = np.ones(len(self.EKF_list))
            # for ekf in self.EKF_list:
            #     ekf.x = self.EKF_list[self.best_index].x

        return self.EKF_list[self.best_index].x, self.EKF_list[self.best_index].bis, self.best_index


class MEKF_Narendra:
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
        Design parameters of the system [alpha, beta, lambda_p, epsilon, window_length]. The default is [1, 1, 0.1, 0.9].
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

        # define a control system
        self.syst = ctrl.ss(A, B, np.eye(8), np.zeros((8, 2)), ts)

        # Init the criterion
        self.grid_vector = grid_vector
        if eta0 is None:
            self.eta = np.ones((len(self.EKF_list), 1))
        self.eta = eta0
        self.best_index = np.argmin(eta0)

        # define the design parameters
        self.N_model = len(self.EKF_list)  # total number of model
        self.window_length = design_param[4]
        self.epsilon = design_param[3]
        self.alpha = design_param[0]
        self.beta = design_param[1]
        self.lambda_p = design_param[2]
        self.ts = self.EKF_list[0].ts

        # Init internal variables
        self.error = np.zeros(self.N_model)
        self.X = np.zeros((self.N_model, self.window_length, 8))
        self.Up = np.zeros(self.window_length-1)
        self.Ur = np.zeros(self.window_length-1)
        self.BIS = np.zeros(self.window_length)

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
        plot = False  # for debug

        # estimate the state for each EKF
        if np.sum(self.BIS) == 0:
            self.BIS = np.ones(self.window_length)*measurement
        else:
            self.BIS = np.concatenate((self.BIS[1:], np.array([measurement])), axis=0)

        for idx in range(self.N_model):
            self.EKF_list[idx].estimate(u, measurement)  # update the estimators
            self.X[idx] = np.concatenate((self.X[idx, 1:, :], np.expand_dims(  # save the paste states
                self.EKF_list[idx].x, axis=1).T), axis=0)
            # compute the prediction error
            _, _, x_pred = ctrl.forced_response(self.syst, U=np.vstack((self.Up, self.Ur)), return_x=True)
            bis_pred = BIS(x_pred[3, :], x_pred[7, :], self.grid_vector[idx])
            bis_pred = self.EKF_list[idx].predict_from_state(x=self.X[idx, 0], up=self.Up, ur=self.Ur)
            # bis_pred = self.EKF_list[idx].find_from_state(x=self.EKF_list[idx].x, up=self.Up, ur=self.Ur)

            epsilon = np.square(bis_pred - self.BIS)
            integral = self.ts * np.sum([np.exp(- self.lambda_p * (self.window_length - i)*self.ts) * epsilon[i]
                                         for i in range(self.window_length)])

            self.error[idx] = self.alpha*epsilon[-1] + self.beta*integral

            if plot:
                plt.plot(bis_pred, label='model ' + str(idx))

        if plot:
            # plt.legend()
            plt.plot(self.BIS, 'r--', label='Measured BIS')
            plt.show()
            plt.bar(np.arange(len(self.error)), self.error)
            plt.yscale('log')
            plt.show()

        idx_best_new = np.argmin(self.error)

        if self.error[idx_best_new] < self.epsilon * self.error[self.best_index]:
            self.best_index = idx_best_new

        return self.EKF_list[self.best_index].x, self.EKF_list[self.best_index].bis, self.best_index


class MHE:
    """Implementation of the Moving Horizon Estimator for the Coadministration of propof and remifentanil in Anesthesia.

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
    N_MHE : int, optional
        Number of steps of the horizon. The default is 20.
    theta : list, optional
        Parameters of the Q matrix. The default is np.ones(12).

    Returns
    -------
    None.

    """

    def __init__(self, A: list, B: list, BIS_param: list, ts: float = 1, x0: list = np.zeros((8, 1)),
                 Q: list = np.eye(8), R: list = np.array([1]), N_MHE: int = 20, theta: list = np.ones(12)) -> None:

        self.Ad, self.Bd = discretize(A, B, ts)
        self.ts = ts
        self.nb_states = 11
        self.nb_inputs = 2
        self.BIS_param = BIS_param
        self.theta = theta
        C50p = BIS_param[0]
        C50r = BIS_param[1]
        gamma = BIS_param[2]
        beta = BIS_param[3]
        E0 = BIS_param[4]
        Emax = BIS_param[5]

        self.Q = cas.MX(Q)
        self.R = R
        self.N_mhe = N_MHE

        # declare CASADI variables
        x = cas.MX.sym('x', self.nb_states)  # x1p, x2p, x3p, x4p, x1r, x2r, x3r, x4r, c50p, c50r, gamma
        u = cas.MX.sym('u', self.nb_inputs)   # Propofol and remifentanil infusion rate

        self.Ad = np.block([[self.Ad, np.zeros((8, 3))], [np.zeros((3, 8)), np.eye(3)]])
        self.Bd = np.block([[self.Bd], [np.zeros((3, 2))]])
        # declare CASADI functions
        xpred = cas.MX(self.Ad) @ x + cas.MX(self.Bd) @ u
        self.Pred = cas.Function('Pred', [x, u], [xpred], ['x', 'u'], ['xpred'])

        up = x[3] / x[8]
        ur = x[7] / x[9]
        Phi = up/(up + ur + 1e-6)
        U_50 = 1 - beta * (Phi - Phi**2)
        i = (up + ur)/U_50
        y = E0 - Emax * i ** x[10] / (1 + i ** x[10])
        self.output = cas.Function('output', [x], [y], ['x'], ['bis'])

        # ----- optimization problem -----
        # optimization variables
        x_bar = cas.MX.sym('x0', self.nb_states*N_MHE)
        # parameters
        x_pred = cas.MX.sym('x_pred', self.nb_states*N_MHE)
        u = cas.MX.sym('u', self.nb_inputs*N_MHE)
        y = cas.MX.sym('y', N_MHE)
        time = cas.MX.sym('time', 1)

        # objective function
        Q8 = theta[0] + theta[1]*np.exp(-theta[2]*np.exp(-theta[3]*time))
        Q9 = theta[4] + theta[5]*np.exp(-theta[6]*np.exp(-theta[7]*time))
        Q10 = theta[8] + theta[9]*np.exp(-theta[10]*np.exp(-theta[11]*time))
        Q = cas.blockcat([[self.Q, cas.MX(np.zeros((8, 3)))],
                          [cas.MX(np.zeros((3, 8))), cas.diag(cas.vertcat(Q8, Q9, Q10))]])

        J = (y[0] - self.output(x_bar[:self.nb_states]))**2 * self.R
        g = []
        for i in range(1, N_MHE):
            J += (y[i] - self.output(x_bar[self.nb_states*i:self.nb_states*(i+1)]))**2 * self.R
            x_pred_plus = self.Pred(x=x_pred[self.nb_states*i:self.nb_states*(i+1)],
                                    u=u[self.nb_inputs*(i-1):self.nb_inputs*i])['xpred']
            J += (x_bar[self.nb_states*i:self.nb_states*(i+1)] -
                  x_pred_plus).T @ Q @ (x_bar[self.nb_states*i:self.nb_states*(i+1)] - x_pred_plus)
            # constraints
            x_bar_plus = self.Pred(x=x_bar[self.nb_states*(i-1):self.nb_states*i],
                                   u=u[self.nb_inputs*(i-1):self.nb_inputs*i])['xpred']
            g = cas.vertcat(g, x_bar[self.nb_states*i:self.nb_states*(i+1)] - x_bar_plus)

        # create solver instance
        opts = {'ipopt.print_level': 1, 'print_time': 0, 'ipopt.max_iter': 300}
        prob = {'f': J, 'p': cas.vertcat(*[u, y, x_pred, time]),
                'x': x_bar, 'g': g}  # +gbis
        self.solver = cas.nlpsol('solver', 'ipopt', prob, opts)
        # define bound for variables
        self.lbx = ([1e-6]*8 + [0.5]*3)*N_MHE
        self.ubx = ([20]*8 + [8, 60, 5])*N_MHE
        self.lbg = [0]*self.nb_states*(N_MHE-1)
        self.ubg = [0]*self.nb_states*(N_MHE-1)

        # init state and output
        self.x = np.array([[1e-6]*8+[C50p, C50r, gamma]]).T * np.ones((1, N_MHE))
        self.y = []
        self.u = np.zeros(2*N_MHE)
        self.x_pred = self.x.reshape(11*N_MHE, order='F')
        self.time = 0

    def one_step(self, u, Bis) -> np.array:
        """solve the MHE problem for one step.

        Parameters
        ----------
        u : list
            propofol and remifentanil infusion rate at time t-1
        Bis : float
            BIS value at time t

        Returns
        -------
        np.array
            state estimation at time t

        """
        if len(self.y) == 0:
            self.y = [Bis]*self.N_mhe
        else:
            self.y = self.y[1:] + [Bis]

        self.u = np.hstack((self.u[2:], u))
        self.time += self.ts
        # solve the problem
        x0 = []
        for i in range(self.N_mhe):
            x0 += list(self.Pred(x=self.x_pred[self.nb_states*i:self.nb_states*(i+1)], u=self.u[2*i:2*(i+1)])
                       ['xpred'].full().flatten())
        res = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx,
                          lbg=self.lbg, ubg=self.ubg,
                          p=cas.vertcat(*[self.u, self.y, self.x_pred, self.time]))
        x_bar = res['x']
        # x_bar = x0

        self.x = np.reshape(x_bar, (11, self.N_mhe), order='F')

        if self.time > 400000:
            # compute the prediction of x_pred
            self.x_pred = self.x_pred.reshape((11, self.N_mhe), order='F')
            X_pred_plus = np.zeros((11, self.N_mhe-1))
            for i in range(self.N_mhe-1):
                X_pred_plus[:, i] = self.Pred(x=self.x_pred[:, i], u=self.u[2*(i-1):2*i])['xpred'].full().flatten()
            plt.subplot(11, 1, 1)
            plt.plot(self.x[0, :], 'b', label='x estimated')
            plt.plot(self.x_pred[0, :], 'r', label='x init')
            plt.plot(X_pred_plus[0, :], 'g', label='x pred')
            plt.subplot(11, 1, 2)
            plt.plot(self.x[1, :], 'b')
            plt.plot(self.x_pred[1, :], 'r')
            plt.plot(X_pred_plus[1, :], 'g')
            plt.subplot(11, 1, 3)
            plt.plot(self.x[2, :], 'b')
            plt.plot(self.x_pred[2, :], 'r')
            plt.plot(X_pred_plus[2, :], 'g')
            plt.subplot(11, 1, 4)
            plt.plot(self.x[3, :], 'b')
            plt.plot(self.x_pred[3, :], 'r')
            plt.plot(X_pred_plus[3, :], 'g')
            plt.subplot(11, 1, 5)
            plt.plot(self.x[4, :], 'b')
            plt.plot(self.x_pred[4, :], 'r')
            plt.plot(X_pred_plus[4, :], 'g')
            plt.subplot(11, 1, 6)
            plt.plot(self.x[5, :], 'b')
            plt.plot(self.x_pred[5, :], 'r')
            plt.plot(X_pred_plus[5, :], 'g')
            plt.subplot(11, 1, 7)
            plt.plot(self.x[6, :], 'b')
            plt.plot(self.x_pred[6, :], 'r')
            plt.plot(X_pred_plus[6, :], 'g')
            plt.subplot(11, 1, 8)
            plt.plot(self.x[7, :], 'b')
            plt.plot(self.x_pred[7, :], 'r')
            plt.plot(X_pred_plus[7, :], 'g')
            plt.subplot(11, 1, 9)
            plt.plot(self.x[8, :], 'b')
            plt.plot(self.x_pred[8, :], 'r')
            plt.plot(X_pred_plus[8, :], 'g')
            plt.subplot(11, 1, 10)
            plt.plot(self.x[9, :], 'b')
            plt.plot(self.x_pred[9, :], 'r')
            plt.plot(X_pred_plus[9, :], 'g')
            plt.subplot(11, 1, 11)
            plt.plot(self.x[10, :], 'b')
            plt.plot(self.x_pred[10, :], 'r')
            plt.plot(X_pred_plus[10, :], 'g')
            plt.show()

            plt.figure()
            plt.plot(self.y, 'b', label='true bis')
            bis = [self.output(x=self.x[:, i])['bis'].full().flatten() for i in range(self.N_mhe)]
            bis_x0 = [self.output(x=self.x0[self.nb_states*i:self.nb_states*(i+1)])
                      ['bis'].full().flatten() for i in range(self.N_mhe)]
            plt.plot(bis, 'r', label='estimated bis')
            plt.legend()
            plt.show()
        self.x_pred = np.array(res['x']).reshape(self.nb_states*self.N_mhe)
        bis = float(self.output(x=self.x[:, [-1]])['bis'])
        return self.x[:, [-1]], bis

    def plot_variable_cost(self):
        """Plot the variable cost of the MHE problem."""

        plt.figure()
        t = np.linspace(0, 1000, 100)
        y_1 = self.theta[0] + self.theta[1]*np.exp(-self.theta[2]*np.exp(-self.theta[3]*t))
        y_2 = self.theta[4] + self.theta[5]*np.exp(-self.theta[6]*np.exp(-self.theta[7]*t))
        y_3 = self.theta[8] + self.theta[9]*np.exp(-self.theta[10]*np.exp(-self.theta[11]*t))
        plt.plot(t, y_1, label='Q8')
        plt.plot(t, y_2, label='Q9')
        plt.plot(t, y_3, label='Q10')
        plt.legend()
        plt.grid()
        plt.show()
