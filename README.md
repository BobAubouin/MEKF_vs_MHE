<img src ="https://img.shields.io/github/last-commit/BobAubouin/MEKF_vs_MHE" alt="GitHub last commit"> 

# MEKF_vs_MHE
Use simulated  and clinical data to compare Multi Extended Kalman Filter and Moving Horizon Estimation to identify PD parameters in drug model during total intravenous anesthesia.

## Structure

    .
    ├── data                # Simulations data generated to compare the results
    ├── scripts             # python scripts to launch the tuning and test of the estimators 
    ├── observabilite       # scripts to study the observability of the system
    ├── realistic_simu_PID  # scripts to generate the simulated data 
    ├── LICENSE
    ├── requirements.txt
    ├── README.md
    └── .gitignore     


## Usage

First some folder must be created to store the data and the results. 
```bash
mkdir data figures
mkdir data/mekf_p data/mhe_std data/simulations data/vital
mkdir data/vital/mekf_p data/vital/mhe_std
```
Code have runned using python 3.9.

Simulation data can be generated with the following command:
```bash
python3 realistic_simu_PID.py
```


Then the tuning of the hyper parameters can be launched with the following command:
```bash
python3 scripts/tuning_mekf.py
python3 scripts/tuning_mhe.py
```

The test of the estimators on simulated data can be launched with the following command:
```bash
python3 scripts/test_on_simu.py
python3 scripts/metrics_function.py
```
Results figures will be saved on the figures folder.

The test of the estimators on clinical data can be launched with the following command:
```bash
python3 scripts/download_vital.py
python3 scripts/test_on_vital.py
python3 scripts/metrics_vital.py
```
Results figures will be saved on the figures folder.

## License
_GNU General Public License 3.0_

## Project status
- Code finished, paper in review.

## Author
Bob Aubouin--Paitault
