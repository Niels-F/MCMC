# MCMC Parameter Estimation for Planetary Interior Models using CEPAM

This repository contains a Python script that performs **Markov Chain Monte Carlo (MCMC)** simulations to estimate internal structure parameters of a planetary model. The script uses the **CEPAM** planetary evolution code to simulate physical outputs (e.g., radius, gravitational moments), and optimizes parameters to fit observations.

> ⚠️ **Important**: This script requires the **CEPAM** program, which is **not included** in this repository. You must [download and compile CEPAM](https://www.oca.eu/fr/cepam) separately. The binary path must be specified correctly in the script (`../Testfolder_SVN/bin/cepam` by default).

---

## 🧰 Requirements

- **Python 3.x**
- **CEPAM** binary (external, not included)
- Python packages:
  pip install numpy astropy pexpect

---


## 📁 Repository Contents


.
├── mcmc_script.py        # Main script that runs the MCMC loop
├── config.txt            # Parameter names and ranges (editable by user)
├── jup.ini               # CEPAM input file (initial planetary model)
├── cepam_etat.don        # CEPAM auxiliary input file
├── results.txt           # Log of accepted models and likelihoods
├── results_2.txt         # Log of rejected models
└── jup_js.csv            # CEPAM output (auto-generated at runtime)


---

## 📝 Configuration File (`config.txt`)

Defines the parameters to optimize and their sampling range.

---

## 🚀 How to Use

1. Make sure `cepam` is compiled and available at the path defined in `mcmc_script.py`.
2. Edit `config.txt` to define your parameter space.
3. Prepare valid `jup.ini` and `cepam_etat.don` files.
4. Run the script:

The script will:

* Initialize a model
* Propose new parameter sets using MCMC
* Run CEPAM for each proposal
* Compute the likelihood compared to observational data
* Save accepted and rejected proposals to text files

---

## 📤 Output Files

* `results.txt`: Contains accepted models, with parameter values and likelihood scores.
* `results_2.txt`: Contains rejected proposals and their likelihoods.

Each entry includes:

* Observables: `REQ_CM`, `J2`, `J4`
* Model parameters: `mnoyau`, `zatm_i`, `zatm_r`, `zdeep_i`, `zdeep_r`, `ppt`
* Likelihood value

---

## ⚠️ Notes

* CEPAM runs in **interactive mode** and is controlled via `pexpect`.
* Models that take longer than 7 seconds are discarded.
* If the initial guess is too far from feasible values, it is redrawn until valid.

---

## 📜 License

This code is provided for research purposes. Please cite the author if used. CEPAM is developed and maintained by the Observatoire de la Côte d'Azur, and should be cited as required by its license.

---

## 🙋 Contact

For issues or suggestions, feel free to open a pull request or GitHub issue.

