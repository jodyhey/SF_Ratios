# SF_Ratios Documentation
by Jody Hey, 2024

The SF_Ratios archive contains materials associated with the manuscript "Poisson random field ratios in population genetics:  estimating the strength of selection while sidestepping non-selective factors" by Jody Hey and Vitor Pavinato. Included are the main scripts for estimating selection parameters, the site frequency spectra for the *Drosophila melanogaster* data,  and scripts for testing the method on simulated data.

This archive is also a copy of a Visual Studio Code Workspace, including a launch.json file if anyone wants to use it. 

## Main Scripts
* SF_Ratios.py  - selection model fitting for ratios of Site Frequency Spectra, Selected/Neutral
* SF_Ratios_functions.py - various functions called by SF_Ratios.py and other scripts in this archive 

SF_Ratios.py should generally be run using the -g option that turns on basinhopping for optimization.  This is quite slow,  and some runs may take a couple days,  but it is often worth it. 

## Subfolder Contents
### ./performance
Scripts and folders for assessing estimator performance.
* Estimation_on_WrightFisher_SF_simulations.py - runs ROC, Power and Chi^2 comparison analyses on data simulated under PRF.  
* Estimation_on_SFS_with_SLiM.py - does PRF-Ratio model fitting on data previously simulated using SLiM
* Simulate_SFS_with_SLiM.py - runs SLiM simulations using models and functions found in the *slim_work* folder
* Results_WrightFisher_SF_simulations - the default folder for output from Estimation_on_WrightFisher_SF_simulations.py.  Contains the results of ROC,Power and Chi^2 comparison analyses presented in the paper
* Results_SFS_with_SLiM - the default folder for output from Estimation_on_SFS_with_SLiM.py. Contains the results for various demographic models that were presented in the paper. 

### ./data 
Data files for North Caroline (DGRP2) and Zambia (DPGP3) samples. All files have the neutral SFS based on short introns first, followed by the selected SFS.  All SFSs begin with bin 0.  

### ./utilities
* get_SF_Ratio_output_summaries.py - a script that can read a bunch of output files from SF_Ratio.py and generate a .csv file with main results
* make_2Ns_distribution_plot.py - a script that can make a figure from a SF_Ratio.py output file
* SFS_modifications.py - has several utilities for handling SFSs
* twoDboxplot.py - called by Estimation_on_SFS_with_SLiM.py when run using a lognormal or gamma density.  Can be run as a standalone on an output file from Estimation_on_SFS_with_SLiM.py.
* compare_ratio_poissons_to_ratio_gaussians.py - simulate the ratio of two poisson random variables, and plot the histogram. Also plot the corresponding density of the ratio of two gaussians using ex (1) of Díaz-Francés, E. and F. J. Rubio (2013). "On the existence of a normal approximation to the distribution of the ratio of two independent normal random variables." Statistical Papers 54: 309-323.
* simulate_WF_SFSs_for_SF_Ratios.py - simulate a data set for SF_Ratios.py 
### ./slim_work
Contains folders and files used for generating simualted data sets with SLiM.  These are used by  ./performance/Estimation_on_SLiM_SFS_simulations.py  and ./performance/Simulate_SFS_with_SLiM.py.
#### ./slim_work/output
Contains the results from  ./performance/Simulate_SFS_with_SLiM.py for a constant lognormal run
#### ./slim_work/functions
Contains SLiM function files required by  ./performance/Simulate_SFS_with_SLiM.py.
#### ./slim_work/models
Contains demograhpic model files to be run by SLiM using   ./performance/Simulate_SFS_with_SLiM.py.
