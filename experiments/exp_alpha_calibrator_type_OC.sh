#!/bin/bash

while getopts "d:m:s:n:t:c:p:b:r:o:" flag; do
    case $flag in
        d) dataset=${OPTARG};;
        m) dim=${OPTARG};;
        s) save_path=${OPTARG};;
        n) n_train=${OPTARG};;
        t) n_test=${OPTARG};;
        c) n_cal=${OPTARG};;
        p) n_seeds=${OPTARG};;
        b) calib_type=${OPTARG};;
	r) soft_rank_r=${OPTARG};;
        o) tp=${OPTARG};;
	\?) echo "Invalid option -$OPTARG" >&2
        exit 1;;
    esac
done

if  [ $tp = 0.1 ]
then
	test_p=(0.1)
	mus_o=(3.6)
fi
if [ $tp = 0.5 ]
then
	test_p=(0.5)
	mus_o=(3.4)
fi

models=('OC-SVM')
algorithms=('CalibratorConformalOCC')
n_e_values=(10)

for alpha in ${alphas[@]}; do
	for test_purity in ${test_p[@]}; do
		for mu_o in ${mus_o[@]}; do
    			for algo in ${algorithms[@]}; do
      				for n_e in ${n_e_values[@]}; do
        				for model in ${models[@]}; do
        	  				bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha $alpha --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_features $dim --n_seeds $n_seeds --n_e_value $n_e --agg_alpha_t avg --alpha_t 0.05 --mu_o $mu_o --sv_exp --reuse_xnull --weight_metric t-test --calibrator_type $calib_type --soft_rank_r $soft_rank_r --test_purity $test_purity"
        				done
      				done
    			done
		done
	done
done

