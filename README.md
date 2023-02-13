
## Derandomized novelty detection with FDR control via conformal e-values

This repository contains a Python implementation of the following paper: "Derandomized novelty detection with FDR control via conformal e-values".
The repository includes an implementation of the proposed derandomized novelty detection method (*E-AdaDetect* and *E-OC-Conformal*).
Baseline methods implementation alongside code for synthetic and real data experiments are included in this repository.

### Abstract

Conformal prediction and other randomized model-free inference techniques are gaining increasing attention as general solutions to rigorously calibrate the output of any machine learning algorithm for novelty detection.
This paper contributes to the field by developing a novel method for mitigating their algorithmic randomness, leading to an even more interpretable and reliable framework for powerful novelty detection under false discovery rate control.
The idea is to leverage suitable conformal *e-values* instead of *p-values* to quantify the significance of each finding,
which allows the evidence gathered from multiple mutually dependent analyses of the same data to be seamlessly aggregated. 
Further, the proposed method can reduce randomness without much loss of power, partly thanks to an innovative way of weighting conformal e-values based on additional side information carefully extracted from the same data.
Simulations with synthetic and real data confirm this solution can be effective at eliminating random noise in the inferences obtained with state-of-the-art alternative techniques, sometimes also leading to higher power.

### Usage Instructions

#### Setup a conda environment
You can create a conda environment from requirements.yml file or make sure your virtual environment includes all requirements (specified in `requirements.yml`).

run the following commands to create a conda environment from `requirements.yml` file:
```
conda env update -f requirements.yml
conda activate d-rand-conformal
```

#### Run experiments
To run a single experiment with the desired parameters, use `main.py`. 
For example, a single run of *E-AdaDetect* with random forest classifier: 
```
python main.py --save_path ./results/ --model RF --algorithm E_value_AdaDetect --alpha 0.1 --n_cal 500 --n_train 1250 --n_test 3000 --dataset artificial_gaussian --n_features 500
```

To run an experiment with *M* repetitions (with different seeds), use `wrapper.py`.
For example, an experiments with 10 repetitions:
```
python wrapper.py --save_path ./results/ --n_seeds 10 --model RF --algorithm E_value_AdaDetect --alpha 0.1 --n_cal 500 --n_train 1250 --n_test 3000 --dataset artificial_gaussian --n_features 500
```
Example of experiment with **fixed** test set and null samples:
```
python wrapper.py --save_path ./results/ --n_seeds 10 --sv_exp --reuse_xnull --model RF --algorithm E_value_AdaDetect --alpha 0.1 --n_cal 500 --n_train 1250 --n_test 3000 --dataset artificial_gaussian --n_features 500
```

For more details on command line arguments, please run the following command:
```
python main.py --help
```
or
```
python wrapper.py --help
```

#### Produce plots
To produce a plot, use `plot_main.py`. 
For example, plot of signal amplitude experiment: 
```
python ./plot_main.py ./plots/synthetic/signal_amplitude/ --files ./results/synthetic/signal_amplitude/  --x mu_o --filter_keys n_features --filter_keys n_cal --filter_keys n_train --filter_keys n_test --filter_keys alpha --filter_keys model --filter_values 500 --filter_values 500 --filter_values 1250 --filter_values 3000 --filter_values_float 0.1 --filter_values_str RF --save_name _d_500_RF_a_0.1 --a_t2plot 0.05 --plot_variance
```
Note: No filter flags are needed if all files in the directory belong to one experiment.

For more details on command line arguments, please run the following command:
```
python plot_main.py --help
```