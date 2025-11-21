-------------------------------------------------------------

This package suggests Hamiltonian parameter combinations to
reproduce experimental Resonant Inelastic X-ray Scattering (RIXS)
spectra from first-principles models based on exact diagonalization.

The method works in three steps:

1. Bayesian Optimization (BO) in the most important parameter space

2. Greedy refinement of the best outcomes of BO

3. Clustering the outcomes for a small number of distinct
best fit candidates.

Usage:

Material/model-specific information is assumed to be stored in a
config python script. Please consult with the examples `NiPS3_config.py`
and `NiCl2_config.py`. The config script is dynamically loaded by the BO,
refinement and clustering scripts.

Usage in a HPC environment is facilitated by an extra index argument
segmenting run batches.

1. Bayesian optimization:

```bash
python BO_run.py --config-py NiPS3_config.py --init-seed 0 --run-ind 1
```

2. Greedy refinement:
(This command runs the refinement for candidate points with indices 0 through 6)

```bash
python BO_refine.py --config-py NiPS3_config.py  0  6
```

3. Clustering:

```bash
python BO_cluster.py --config-py NiPS3_config.py --input NiPS3_L1sum_Final_s0.025.txt --output NiPS3_clusters
```

See `NiPS_pipeline.sh` for further details of the running process.

Data Repository
===============

Source code and data files for the manuscript. 

How to cite
-----------
If this work is used, please cite "Hamiltonian parameter inference from RIXS spectra with active learning", Marton K. Lajer, Xin Dai, Kipton Barros, Matthew R. Carbone, S. Johnston, and M. P. M. Dean, Phys. Rev. B 112, 155167 (2025)

This work is based on data from the papers

- NiPS3: W. He, Y. Shen, K. Wohlfeld, J. Sears, J. Li, J. Pelliciari, M. Walicki, S. Johnston, E. Baldini, V. Bisogni, M. Mitrano, and M. P. M. Dean, Magnetically propagating Hund’s exciton in van der Waals antiferromagnet, NiPS3, Nature Communications 15 (2024).

- NiCl2: C. A. Occhialini, Y. Tseng, H. Elnaggar, Q. Song, M. Blei, S. A. Tongay, V. Bisogni, F. M. F. de Groot, J. Pelliciari, and R. Comin, Nature of excitons and their ligand-mediated delocalization in nickel dihalide charge-transfer insulators, Phys. Rev. X 14, 031007 (2024)

- Fe2O3: J. Li, Y. Gu, Y. Takahashi, K. Higashi, T. Kim, Y. Cheng, F. Yang, J. Kuneˇs, J. Pelliciari, A. Hariki, and
V. Bisogni, Single- and Multimagnon Dynamics in Anti-ferromagnetic α -Fe2O3 Thin Films, Physical Review X
13, 011012 (2023).

- Ca3LiOsO6: A. E. Taylor, S. Calder, R. Morrow, H. L. Feng, M. Upton, M. D. Lumsden, K. Yamaura, P. M. Woodward, A. D. Christianson, and A. D. Christianson, Spin-orbit coupling controlled J = 3/2 electronic ground state in 5d3 oxides., Physical review letters 118 20, 207202 (2016).

Please cite these as appropriate.


Run
===

This code is designed for linux only.

```bash
    conda env create -f environment.yml -n edrixs_BO
    conda activate edrixs_BO
```
Change `tree` to `lab` in the URL for JupyterLab.
