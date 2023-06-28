#!/bin/bash

while getopts "d:m:s:n:t:c:p:o:" flag; do
    case $flag in
        d) dataset=${OPTARG};;
        m) dim=${OPTARG};;
        s) save_path=${OPTARG};;
        n) n_train=${OPTARG};;
        t) n_test=${OPTARG};;
        c) n_cal=${OPTARG};;
        p) n_seeds=${OPTARG};;
        o) tp=${OPTARG};;
	\?) echo "Invalid option -$OPTARG" >&2
        exit 1;;
    esac
done

alpha=0.1
alphas_bh=(0.005 0.01 0.015 0.02 0.025 0.03 0.035 0.035 0.04 0.045 0.05 0.055 0.06 0.065 0.07 0.075 0.08 0.085 0.09 0.095 0.1)

if [ $tp = 0.1 ]
then
	test_p=(0.1)
	mus_o=(2.8 3.4)
fi
if [ $tp = 0.5 ]
then
	test_p=(0.5)
	mus_o=(1.1 1.6)
fi

models=('LogisticRegression')
algorithms=('E_value_AdaDetectERM')
n_e_values=(10)

for alpha_bh in ${alphas_bh[@]}; do
	for test_purity in ${test_p[@]}; do
		for mu_o in ${mus_o[@]}; do
			for algo in ${algorithms[@]}; do
				for n_e in ${n_e_values[@]}; do
					for model in ${models[@]}; do
		  				bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha $alpha --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_features $dim --n_seeds $n_seeds --n_e_value $n_e --alpha_t $alpha_bh --agg_alpha_t avg --mu_o $mu_o --weight_metric t-test --test_purity $test_purity"
					done
				done
			done
		done
	done
done


if [ $tp = 0.1 ]
then
	test_p=(0.1)
	mus_o=(2.6 3.6)
fi
if [ $tp = 0.5 ]
then
	test_p=(0.5)
	mus_o=(2.3 3.4)
fi
models=('OC-SVM')
algorithms=('E_value_ConformalOCC')
n_e_values=(10)
for alpha_bh in ${alphas_bh[@]}; do
	for test_purity in ${test_p[@]}; do
		for mu_o in ${mus_o[@]}; do
			for algo in ${algorithms[@]}; do
				for n_e in ${n_e_values[@]}; do
					for model in ${models[@]}; do
		  				bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha $alpha --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_features $dim --n_seeds $n_seeds --n_e_value $n_e --alpha_t $alpha_bh --agg_alpha_t avg --mu_o $mu_o --weight_metric t-test --test_purity $test_purity"
					done
				done
			done
		done
	done
done

