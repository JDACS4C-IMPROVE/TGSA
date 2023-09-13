echo SETTINGS

# General Settings
export PROCS=4
export PPN=4
export WALLTIME=01:00:00
export NUM_ITERATIONS=3
export POPULATION_SIZE=2

# GA Settings
export GA_STRATEGY='mu_plus_lambda'
export OFFSPRING_PROPORTION=0.5
export MUT_PROB=0.8
export CX_PROB=0.2
export MUT_INDPB=0.5
export CX_INDPB=0.5
export TOURNSIZE=4

# Add any additional settings needed for your system. General settings and system settings need to be set by user, while GA settings don't need to be changed.
# Default settings for lambda and polaris are given here.

# If you have write access to the shared filesystem on your computation system (such as /lambda_stor),
# you can save there. If not, make a directory in /tmp or somewhere else you can write.

# Lambda Settings
#export CANDLE_CUDA_OFFSET=2
##export CANDLE_DATA_DIR=/homes/ac.jjiang/data_dir
#export CANDLE_DATA_DIR=/candle_data_dir

