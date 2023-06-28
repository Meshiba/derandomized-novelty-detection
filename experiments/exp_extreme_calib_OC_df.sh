#!/bin/bash

while getopts "d:m:s:t:p:a:o:" flag; do
    case $flag in
        d) dataset=${OPTARG};;
        m) dim=${OPTARG};;
        s) save_path=${OPTARG};;
        t) n_test=${OPTARG};;
        p) n_seeds=${OPTARG};;
        a) alpha=${OPTARG};;
        o) tp=${OPTARG};;
	\?) echo "Invalid option -$OPTARG" >&2
        exit 1;;
    esac
done

cals=(500 1000 5000 10000 15000 20000 25000 30000)
n=${#cals[@]}
trains=(1500 2000 6000 11000 16000 21000 26000 31000)

if [ $tp = 0.05 ]
then
	test_p=(0.05)
	mus_o=(3.3 3.7)
fi
if [ $tp = 0.1 ]
then
	test_p=(0.1)
	mus_o=(3.2 3.6)
fi
models=('OC-SVM')
algorithms=('ConformalOCC')
for (( c=0; c<$n; c++ )) do
	for algo in ${algorithms[@]}; do
		for test_purity in ${test_p[@]}; do
			for mu_o in ${mus_o[@]}; do
				for model in ${models[@]}; do
			 		bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha $alpha --n_cal ${cals[$c]} --n_train ${trains[$c]} --n_test $n_test --dataset $dataset --save_path $save_path --n_features $dim --n_seeds $n_seeds --mu_o $mu_o --test_purity $test_purity"
				done
			done
		done
	done
done

algorithms=('E_value_ConformalOCC')
n_e_values=(10)
for (( c=0; c<$n; c++ )) do
	for test_purity in ${test_p[@]}; do
		for mu_o in ${mus_o[@]}; do
			for algo in ${algorithms[@]}; do
				for n_e in ${n_e_values[@]}; do
					for model in ${models[@]}; do
		  				bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha $alpha --n_cal ${cals[$c]} --n_train ${trains[$c]} --n_test $n_test --dataset $dataset --save_path $save_path --n_features $dim --n_seeds $n_seeds --n_e_value $n_e --alpha_t 0.05 --agg_alpha_t avg --mu_o $mu_o --weight_metric t-test --test_purity $test_purity"
					done
				done
			done
		done
	done
done

