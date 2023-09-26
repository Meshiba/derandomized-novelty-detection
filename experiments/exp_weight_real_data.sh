#!/bin/bash

while getopts "d:m:s:n:t:c:p:a:o:" flag; do
    case $flag in
        d) dataset=${OPTARG};;
        s) save_path=${OPTARG};;
        n) n_train=${OPTARG};;
        t) n_test=${OPTARG};;
        c) n_cal=${OPTARG};;
        p) n_seeds=${OPTARG};;
        a) alpha=${OPTARG};;
        o) test_purity=${OPTARG};;
	\?) echo "Invalid option -$OPTARG" >&2
        exit 1;;
    esac
done


models=('RF')
algorithms=('E_value_AdaDetectERM')
n_e_values=(10)
agg_a_t_list=('avg')

for a_t in ${agg_a_t_list[@]}; do
	for algo in ${algorithms[@]}; do
		for n_e in ${n_e_values[@]}; do
			for model in ${models[@]}; do
				bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha $alpha --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_seeds $n_seeds --n_e_value $n_e --agg_alpha_t $a_t --alpha_t 0.01 --sv_exp --reuse_xnull --weight_metric t-test --test_purity $test_purity --random_params --random_params_combined"
				bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha $alpha --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_seeds $n_seeds --n_e_value $n_e --agg_alpha_t $a_t --alpha_t 0.01 --sv_exp --reuse_xnull --weight_metric uniform --test_purity $test_purity --random_params --random_params_combined"
				bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha $alpha --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_seeds $n_seeds --n_e_value $n_e --agg_alpha_t $a_t --alpha_t 0.01 --sv_exp --reuse_xnull --weight_metric avg_score --test_purity $test_purity --random_params --random_params_combined"
			done
		done
	done
	
done


