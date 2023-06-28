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

if [ $tp = 0.1 ]
then
	test_p=(0.1)
	mus_o=(3.4)
fi
if [ $tp = 0.5 ]
then
	test_p=(0.5)
	mus_o=(1.6)
fi
alphas=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4)
n_alphas=${#alphas[@]}
alphas_bh=(0.005 0.01 0.015 0.02 0.025 0.03 0.035 0.04)

models=('LogisticRegression')
algorithms=('AdaDetectERM')

for algo in ${algorithms[@]}; do
	for alpha in ${alphas[@]}; do
		for test_purity in ${test_p[@]}; do
			for mu_o in ${mus_o[@]}; do
    				for model in ${models[@]}; do
    			 		bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha $alpha --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_features $dim --n_seeds $n_seeds --mu_o $mu_o --sv_exp --reuse_xnull --test_purity $test_purity"
    				done
			done
		done
	done
done

algorithms=('E_value_AdaDetectERM')
n_e_values=(10)

for (( c=0; c<$n_alphas; c++ )) do
	for test_purity in ${test_p[@]}; do
		for mu_o in ${mus_o[@]}; do
			for algo in ${algorithms[@]}; do
				for n_e in ${n_e_values[@]}; do
					for model in ${models[@]}; do
		  				bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha ${alphas[$c]} --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_features $dim --n_seeds $n_seeds --n_e_value $n_e --alpha_t ${alphas_bh[$c]} --agg_alpha_t avg --mu_o $mu_o --sv_exp --reuse_xnull --weight_metric t-test --test_purity $test_purity"
					done
				done
			done
		done
	done
done

if [ $tp = 0.1 ]
then
	test_p=(0.1)
	mus_o=(3.6)
fi
if [ $tp = 0.5 ]
then
	test_p=(0.5)
	mus_o=(3.4)
fi

alphas_bh=(0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2)
models=('OC-SVM')
algorithms=('ConformalOCC')

for algo in ${algorithms[@]}; do
	for alpha in ${alphas[@]}; do
		for test_purity in ${test_p[@]}; do
			for mu_o in ${mus_o[@]}; do
    				for model in ${models[@]}; do
    			 		bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha $alpha --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_features $dim --n_seeds $n_seeds --mu_o $mu_o --sv_exp --reuse_xnull --test_purity $test_purity"
    				done
			done
		done
	done
done

algorithms=('E_value_ConformalOCC')
n_e_values=(10)

for (( c=0; c<$n_alphas; c++ )) do
	for test_purity in ${test_p[@]}; do
		for mu_o in ${mus_o[@]}; do
			for algo in ${algorithms[@]}; do
				for n_e in ${n_e_values[@]}; do
					for model in ${models[@]}; do
		  				bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha ${alphas[$c]} --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_features $dim --n_seeds $n_seeds --n_e_value $n_e --alpha_t ${alphas_bh[$c]} --agg_alpha_t avg --mu_o $mu_o --sv_exp --reuse_xnull --weight_metric t-test --test_purity $test_purity"
					done
				done
			done
		done
	done
done
