#!/bin/bash

while getopts "d:m:s:n:t:c:p:a:o:" flag; do
    case $flag in
        d) dataset=${OPTARG};;
        m) dim=${OPTARG};;
        s) save_path=${OPTARG};;
        n) n_train=${OPTARG};;
        t) n_test=${OPTARG};;
        c) n_cal=${OPTARG};;
        p) n_seeds=${OPTARG};;
        a) alpha=${OPTARG};;
        o) tp=${OPTARG};;
	\?) echo "Invalid option -$OPTARG" >&2
        exit 1;;
    esac
done

soft_ranks=(0 1 5 7 10 25 50 75 100 125 150 175 200 300 400 500 600 700)

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
algorithms=('CalibratorAdaDetectERM')
n_e_values=(10)

for soft_rank_r in ${soft_ranks[@]}; do
	for test_purity in ${test_p[@]}; do
		for mu_o in ${mus_o[@]}; do
			for algo in ${algorithms[@]}; do
				for n_e in ${n_e_values[@]}; do
					for model in ${models[@]}; do
		  				bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha $alpha --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_features $dim --n_seeds $n_seeds --n_e_value $n_e --alpha_t 0.01 --agg_alpha_t avg --mu_o $mu_o --weight_metric t-test --calibrator_type soft-rank --soft_rank_r $soft_rank_r --test_purity $test_purity"
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
algorithms=('CalibratorConformalOCC')
n_e_values=(10)
for soft_rank_r in ${soft_ranks[@]}; do
	for test_purity in ${test_p[@]}; do
		for mu_o in ${mus_o[@]}; do
			for algo in ${algorithms[@]}; do
				for n_e in ${n_e_values[@]}; do
					for model in ${models[@]}; do
		  				bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha $alpha --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_features $dim --n_seeds $n_seeds --n_e_value $n_e --alpha_t 0.01 --agg_alpha_t avg --mu_o $mu_o --weight_metric t-test --calibrator_type soft-rank --soft_rank_r $soft_rank_r --test_purity $test_purity"
					done
				done
			done
		done
	done
done

