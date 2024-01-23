import numpy as np
import python_anesthesia_simulator as pas
import optuna
import scipy


def load_mekf_param():
    # %% MEKF parameters

    mean_c50p = 4.47
    mean_c50r = 19.3
    mean_gamma = 1.13
    cv_c50p = 0.182
    cv_c50r = 0.888
    cv_gamma = 0.304

    BIS_param_nominal = [mean_c50p, mean_c50r, mean_gamma, 0, 97.4, 97.4]
    # estimation of log normal standard deviation
    w_c50p = np.sqrt(np.log(1+cv_c50p**2))
    w_c50r = np.sqrt(np.log(1+cv_c50r**2))
    w_gamma = np.sqrt(np.log(1+cv_gamma**2))

    c50p_normal = scipy.stats.lognorm(scale=mean_c50p, s=w_c50p)
    c50r_normal = scipy.stats.lognorm(scale=mean_c50r, s=w_c50r)
    gamma_normal = scipy.stats.lognorm(scale=mean_gamma, s=w_gamma)

    nb_points = 5
    points = np.linspace(0, 1, nb_points+1)
    points = [np.mean([points[i], points[i+1]]) for i in range(nb_points)]

    c50p_list = c50p_normal.ppf(points)

    nb_points = 6
    points = np.linspace(0, 1, nb_points+1)
    points = [np.mean([points[i], points[i+1]]) for i in range(nb_points)]

    c50r_list = c50r_normal.ppf(points)
    gamma_list = gamma_normal.ppf(points)

    # c50p_list = BIS_param_nominal[0]*np.exp([-2.2*w_c50p, -w_c50p, -0.4*w_c50p, 0, w_c50p])  # , -w_c50p
    # c50r_list = BIS_param_nominal[1]*np.exp([-2.2*w_c50r, -w_c50r, -0.4*w_c50r, 0, 0.6*w_c50r, w_c50r])
    # gamma_list = BIS_param_nominal[2]*np.exp([-2.2*w_gamma, -w_gamma, -0.4*w_gamma, 0, 0.8*w_gamma, 1.5*w_gamma])  #
    # surrender list by Inf value
    c50p_list = np.concatenate(([-np.Inf], c50p_list, [np.Inf]))
    c50r_list = np.concatenate(([-np.Inf], c50r_list, [np.Inf]))
    gamma_list = np.concatenate(([-np.Inf], gamma_list, [np.Inf]))

    def get_probability(c50p_set: list, c50r_set: list, gamma_set: list, method: str) -> float:
        """_summary_

        Parameters
        ----------
        c50p_set : float
            c50p set.
        c50r_set : float
            c50r set.
        gamma_set : float
            gamma set.
        method : str
            method to compute the probability. can be 'proportional' or 'uniform'.

        Returns
        -------
        float
            propability of the parameter set.
        """
        if method == 'proportional':

            proba_c50p = c50p_normal.cdf(c50p_set[1]) - c50p_normal.cdf(c50p_set[0])

            proba_c50r = c50r_normal.cdf(c50r_set[1]) - c50r_normal.cdf(c50r_set[0])

            proba_gamma = gamma_normal.cdf(gamma_set[1]) - gamma_normal.cdf(gamma_set[0])

            proba = proba_c50p * proba_c50r * proba_gamma
        elif method == 'uniform':
            proba = 1/(len(c50p_list))/(len(c50r_list))/(len(gamma_list))
        return proba

    def init_proba(alpha):
        grid_vector = []
        eta0 = []
        for i, c50p in enumerate(c50p_list[1:-1]):
            for j, c50r in enumerate(c50r_list[1:-1]):
                for k, gamma in enumerate(gamma_list[1:-1]):
                    grid_vector.append([c50p, c50r, gamma]+BIS_param_nominal[3:])
                    c50p_set = [np.mean([c50p_list[i], c50p]),
                                np.mean([c50p_list[i+2], c50p])]

                    c50r_set = [np.mean([c50r_list[j], c50r]),
                                np.mean([c50r_list[j+2], c50r])]

                    gamma_set = [np.mean([gamma_list[k], gamma]),
                                 np.mean([gamma_list[k+2], gamma])]

                    eta0.append(alpha*(1-get_probability(c50p_set, c50r_set, gamma_set, 'proportional')))
        i_nom = np.argmin(np.sum(np.abs(np.array(grid_vector)-np.array(BIS_param_nominal))), axis=0)
        eta0[i_nom] = alpha
        return grid_vector, eta0

    study_petri = optuna.load_study(study_name="petri_final", storage="sqlite:///data/mekf.db")

    P0 = 1e-3 * np.eye(9)
    Q = study_petri.best_params['Q']
    Q_mat = Q * np.diag([0.1, 0.1, 0.05, 0.05, 1, 1, 10, 1, 0.001])
    R = study_petri.best_params['R']
    alpha = study_petri.best_params['alpha']
    grid_vector, eta0 = init_proba(alpha)
    lambda_1 = 1
    lambda_2 = study_petri.best_params['lambda_2']
    nu = 1.e-5
    epsilon = study_petri.best_params['epsilon']

    design_param = [lambda_1, lambda_2, nu, epsilon]
    MEKF_param = [R, Q_mat, P0, eta0, grid_vector, lambda_1, lambda_2, nu, epsilon]

    return MEKF_param


def load_mhe_param():
    study_mhe_std = optuna.load_study(study_name="mhe_std_final", storage="sqlite:///data/mhe.db")

    R = study_mhe_std.best_params['R']
    q = study_mhe_std.best_params['q']
    Q_std = np.diag([1, 550, 550, 1, 1, 50, 750, 1]+[1e3]*4)*q
    # R = 1/R_p
    # Q_std = np.linalg.inv(Q_p)
    # p = study_mhe_std.best_params['p']
    eta = study_mhe_std.best_params['eta']
    N_mhe_std = study_mhe_std.best_params['N_mhe']
    P = np.diag([1, 550, 550, 1, 1, 50, 750, 1])
    theta = [eta, 800, 100, 0.005]*4
    theta[4] = eta/100
    theta[0] = eta*0.08
    theta[1] = 300
    theta[3] = 0.05
    MHE_std = [R, Q_std, theta, N_mhe_std, P]

    return MHE_std
