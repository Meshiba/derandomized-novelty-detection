
## Derandomized novelty detection with FDR control via conformal e-values

This repository contains a Python implementation of the following paper: "Derandomized novelty detection with FDR control via conformal e-values".
The repository includes an implementation of the proposed derandomized novelty detection method (*E-AdaDetect* and *E-OC-Conformal*).
Baseline methods implementation alongside code for synthetic and real data experiments are included in this repository.

The paper is available [here](https://arxiv.org/abs/2302.07294).

### Abstract

Conformal inference provides a general distribution-free method to rigorously calibrate the output of any machine 
learning algorithm for novelty detection.  While this approach has many strengths, it has the limitation of being 
randomized, in the sense that it may lead to different results when analyzing twice the same data, and this can hinder 
the interpretation of any findings. We propose to make conformal inferences more stable by leveraging suitable conformal
*e-values* instead of *p-values* to quantify statistical significance. This solution allows the evidence gathered from 
multiple analyses of the same data to be aggregated effectively while provably controlling the false discovery rate.
Further, we show that the proposed method can reduce randomness without much loss of power compared to standard 
conformal inference, partly thanks to an innovative way of weighting conformal e-values based on additional side information carefully extracted from the same data. Simulations with synthetic and real data confirm this solution can be effective at eliminating random noise in the inferences obtained with state-of-the-art alternative techniques, sometimes also leading to higher power.
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
For example, a single run of *E-AdaDetect* with logistic regression classifier, K=10 (number of train-calibration splits) and signal strength equal to 3.4: 
```
python main.py --save_path ./results/ --model LogisticRegression --algorithm E_value_AdaDetectERM --n_e_value 10 --alpha 0.1 --n_cal 1000 --n_train 2000 --n_test 1000 --dataset artificial_gaussian --n_features 100 --mu_o 3.4
```

To run an experiment with *M* repetitions (with different seeds), use `wrapper.py`.
For example, an experiments with 10 repetitions:
```
python wrapper.py --save_path ./results/ --n_seeds 10 --model LogisticRegression --algorithm E_value_AdaDetectERM --alpha 0.1 --n_cal 1000 --n_train 2000 --n_test 1000 --dataset artificial_gaussian --n_features 100 --mu_o 3.4
```
Example of experiment with **fixed** test set and null samples:
```
python wrapper.py --save_path ./results/ --n_seeds 10 --sv_exp --reuse_xnull --model LogisticRegression --algorithm E_value_AdaDetectERM --alpha 0.1 --n_cal 1000 --n_train 2000 --n_test 1000 --dataset artificial_gaussian --n_features 100 --mu_o 3.4
```

For more details on command line arguments, please run the following command:
```
python main.py --help
```
or
```
python wrapper.py --help
```

The *experiments* folder contains bash files used for running the numerical experiments from the paper.
In addition, the folder contains two additional files:
* *run_all_experiments.txt* - contains all command lines with relevant parameters for conducting the various experiments.
* *plot_all_experiments.txt* - contains all command lines for producing corresponding plots to the experiments.

The code is intended to run on a computing cluster using the SLURM scheduler.
