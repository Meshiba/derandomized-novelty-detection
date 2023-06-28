#!/bin/bash

while getopts "d:m:s:n:t:c:p:a:" flag; do
    case $flag in
        d) dataset=${OPTARG};;
        m) dim=${OPTARG};;
        s) save_path=${OPTARG};;
        n) n_train=${OPTARG};;
        t) n_test=${OPTARG};;
        c) n_cal=${OPTARG};;
        p) n_seeds=${OPTARG};;
        a) alpha=${OPTARG};;
	\?) echo "Invalid option -$OPTARG" >&2
        exit 1;;
    esac
done

mus_o=(2.8 3.4)

models=('LogisticRegression')
algorithms=('AdaDetectERM')

for mu_o in ${mus_o[@]}; do
	for algo in ${algorithms[@]}; do
		for model in ${models[@]}; do
			bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_features $dim --n_seeds $n_seeds --mu_o $mu_o --alpha $alpha"
		done
	done
done

algorithms=('E_value_AdaDetectERM')
n_e_values=(1 3 5 7 10 15 20 30)

for mu_o in ${mus_o[@]}; do
	for algo in ${algorithms[@]}; do
		for n_e in ${n_e_values[@]}; do
			for model in ${models[@]}; do
				bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_features $dim --n_seeds $n_seeds --n_e_value $n_e --mu_o $mu_o --alpha $alpha --alpha_t 0.01 --agg_alpha_t avg --weight_metric t-test"
			done
		done
	done
done
