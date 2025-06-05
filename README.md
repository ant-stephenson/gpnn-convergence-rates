# gp-prediction

To run simulations modify the bash script `sim_gpnn_limits.sh` to run whichever
form of simulation desired. This script calls the `simulate.py` script which
loops over parameter values etc and outputs the results of the simulation to a
.csv file, which can then be analysed and plotted by the various files in the
`plots` directory. `plot_utils.py` contains functions to read in the .csv
outputs using `pandas`.

Note that at present the `array_idx` variable will, when non-zero, override some
of the settings in the bash script and this needs to be checked in the
`simulate.py` file. 