#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:11:17 2023

@author: aubouinb
"""
import numpy as np
import matplotlib.pyplot as plt
import python_anesthesia_simulator as pas


class TCI():
    """Implement the control algorithm coded in the TCI device Orchestra PRIMUS.

    This code has been retro enginering and does not came from an official source.
    """

    def __init__(self, patient_info: list, drug_name: str, model_used: str,
                 drug_concentration: float = 10, maximum_rate: float = 500, sampling_time: float = 1,
                 control_time: float = 10, target_compartement: str = 'effect_site'):
        """Init the class and do pre-computation.

        Parameters
        ----------
        patient_info : list
            Patient information = [age (yr), height (cm), weight (kg), gender( 0= female, 1 = male)].
        drug_name : str
            Can be either 'Propofol' or 'Remifentanil'.
        drug_concentration : float
            drug concentration in the seringue (mg/ml for Propofol and µg/ml for remifentanil).
        model_used : str
            Could be "Minto", "Eleveld" for Remifentanil,
            ""Schnider", "Marsh_initial", "Marsh_modified", "Shuttler" or "Eleveld" for Propofol.
        maximum_rate : float
            Maximum drug rate in ml/hr.
        sampling_time : float, optional
            Sampling time of the model for the calculs. The default is 1s.
        control_time : float, optional
            Sampling time of the controller, must be a multiple of the sampling time. The default is 10s.
        target_compartement : str, optional
            Can be either "plasma" or "effect_site". The default is 'effect_site'.

        Returns
        -------
        None.

        """
        self.sampling_time = sampling_time
        self.control_time = control_time
        self.drug_concentration = drug_concentration
        if target_compartement == 'plasma':
            self.target_id = 0
        elif target_compartement == 'effect_site':
            self.target_id = 3
        self.infusion_max = maximum_rate * drug_concentration / 3600  # in mg/s or µg/s respectively Propo and Remi

        height = patient_info[1]
        weight = patient_info[2]
        gender = patient_info[3]
        if gender == 1:  # homme
            lbm = 1.1 * weight - 128 * (weight / height) ** 2
        elif gender == 0:  # femme
            lbm = 1.07 * weight - 148 * (weight / height) ** 2

        pk_model = pas.CompartmentModel(patient_info, lbm, drug_name, ts=sampling_time, model=model_used)
        self.Ad = pk_model.discretize_sys.A[:4, :4]
        self.Bd = pk_model.discretize_sys.B[:4]

        pk_model = pas.CompartmentModel(patient_info, lbm, drug_name, ts=control_time, model=model_used)
        self.Ad_control = pk_model.discretize_sys.A[:4, :4]
        self.Bd_control = pk_model.discretize_sys.B[:4]
        # find the response to a 10s infusion
        x = np.zeros((4, 1))
        x_p = np.zeros((4, 1))
        self.Ce = []
        t = sampling_time
        self.t_peak = 0
        while self.t_peak == 0:
            if t < control_time+1:
                x = self.Ad @ x + self.Bd * 1  # simulation of an infusion of 1 mg/s
            else:
                x = self.Ad @ x  # simulation of no infusion
            t += sampling_time
            self.Ce.append(float(x[self.target_id]))
            if x[self.target_id] < x_p[self.target_id]:
                self.t_peak = t-sampling_time
            x_p = x
        self.Ce = np.array(self.Ce)
        # variable used for control
        self.infusion_rate = 0  # last control move chosen
        self.x = np.zeros((4, 1))  # state to store the real patient
        self.target = 0
        self.jpeak_0 = 0
        self.jpeak_1 = 0
        self.time = 0

    def one_step(self, target: float = 0) -> float:
        """Implement one_step of the model. must be called each control sampling time of the model.

        Parameters
        ----------
        target : float, optional
            target concentration (µg/ml for propofol, ng/ml for Remifentanil). The default is 0.

        Returns
        -------
        infusion rate: float
            infusion rate in ml/hr.

        """
        if target != self.target:
            self.jpeak_0 = self.t_peak
            self.target = target

        # compute trajectory from where we are without any infusion
        x_temp = self.x
        Ce_temp = np.zeros(int(self.t_peak/self.sampling_time))
        for t in range(int(self.t_peak/self.sampling_time)):
            x_temp = self.Ad @ x_temp
            Ce_temp[t] = x_temp[self.target_id]
        if Ce_temp[0] > 0.95*target and Ce_temp[0] < 1.05 * target:
            self.infusion_rate = (target - (self.Ad_control @ self.x)[0]) / self.Bd_control[0]
            self.infusion_rate = max(0, self.infusion_rate)
        else:
            if Ce_temp[int(self.control_time/self.sampling_time)] > self.target:
                self.infusion_rate = 0
            else:
                # compute a first guess of the infusion
                infusion_rate_temp = ((self.target - Ce_temp[int(self.jpeak_0/self.sampling_time)-1]) /
                                      self.Ce[int(self.jpeak_0/self.sampling_time)-1])
                self.infusion_rate = min(infusion_rate_temp, self.infusion_max)
                self.jpeak_1 = (Ce_temp + infusion_rate_temp*self.Ce).argmax()
                counter = 0
                while self.jpeak_1 != self.jpeak_0 and counter < 500:
                    self.jpeak_0 = self.jpeak_1
                    infusion_rate_temp = ((self.target - Ce_temp[int(self.jpeak_0/self.sampling_time)-1]) /
                                          self.Ce[int(self.jpeak_0/self.sampling_time)-1])
                    self.infusion_rate = max(min(infusion_rate_temp, self.infusion_max), 0)
                    self.jpeak_1 = (Ce_temp + infusion_rate_temp*self.Ce).argmax()
                    counter += 1

                if counter == 500:
                    print('bug')
                    self.infusion_rate = max(min(infusion_rate_temp, self.infusion_max), 0)

                if False:
                    plt.plot(Ce_temp + infusion_rate_temp*self.Ce)
                    plt.grid()
                    plt.show()

        self.time += self.sampling_time
        self.x = self.Ad_control @ self.x + self.Bd_control * self.infusion_rate
        self.infusion_rate = max(min(self.infusion_rate, self.infusion_max), 0)
        return self.infusion_rate / self.drug_concentration * 3600
