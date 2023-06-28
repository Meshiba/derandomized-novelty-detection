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

test_p=(0.05 0.1 0.2 0.3 0.4 0.5)

models=('IF')
algorithms=('ConformalOCC')

for test_purity in ${test_p[@]}; do
	for algo in ${algorithms[@]}; do
		for model in ${models[@]}; do
			bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha $alpha --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_seeds $n_seeds --test_purity $test_purity"
		done
	done
done

algorithms=('E_value_ConformalOCC')
n_e_values=(70)
agg_a_t_list=('avg')

for test_purity in ${test_p[@]}; do
	for a_t in ${agg_a_t_list[@]}; do
		for algo in ${algorithms[@]}; do
			for n_e in ${n_e_values[@]}; do
				for model in ${models[@]}; do
					bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha $alpha --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_seeds $n_seeds --n_e_value $n_e --agg_alpha_t $a_t --alpha_t 0.05 --weight_metric t-test --test_purity $test_purity"
				done
			done
		done
	done
done

models=('RF' 'LogisticRegression')
algorithms=('AdaDetectERM')

for test_purity in ${test_p[@]}; do
	for algo in ${algorithms[@]}; do
		for model in ${models[@]}; do
			bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha $alpha --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_seeds $n_seeds --test_purity $test_purity"
		done
	done
done

algorithms=('E_value_AdaDetectERM')
n_e_values=(10)
agg_a_t_list=('avg')

for test_purity in ${test_p[@]}; do
	for a_t in ${agg_a_t_list[@]}; do
		for algo in ${algorithms[@]}; do
			for n_e in ${n_e_values[@]}; do
				for model in ${models[@]}; do
					bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha $alpha --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_seeds $n_seeds --n_e_value $n_e --agg_alpha_t $a_t --alpha_t 0.01 --weight_metric t-test --test_purity $test_purity"
				done
			done
		done
	done
done

