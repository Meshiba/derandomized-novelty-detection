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
        o) test_purity=${OPTARG};;
	\?) echo "Invalid option -$OPTARG" >&2
        exit 1;;
    esac
done

if [ $test_purity = 0.05 ]
then
	mus_o=(3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5 5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8 5.9 6 6.1 6.2 6.3 6.4)
fi
if [ $test_purity = 0.1 ]
then
	mus_o=(1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4 4.5 4.6)
fi
if [ $test_purity = 0.2 ]
then
	mus_o=(1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5)
fi
if [ $test_purity = 0.3 ]
then
	mus_o=(0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0)
fi
if [ $test_purity = 0.4 ]
then
	mus_o=(0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5)
fi
if [ $test_purity = 0.5 ]
then
	mus_o=(0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5)
fi

models=('LogisticRegression')
algorithms=('AdaDetectERM')

for algo in ${algorithms[@]}; do
	for mu_o in ${mus_o[@]}; do
		for model in ${models[@]}; do
			bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha $alpha --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_features $dim --n_seeds $n_seeds --mu_o $mu_o --sv_exp --reuse_xnull --test_purity $test_purity"
		done
	done
done

algorithms=('E_value_AdaDetectERM')
n_e_values=(10)
agg_a_t_list=('avg')

for a_t in ${agg_a_t_list[@]}; do
	for mu_o in ${mus_o[@]}; do
		for algo in ${algorithms[@]}; do
			for n_e in ${n_e_values[@]}; do
				for model in ${models[@]}; do
					bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha $alpha --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_features $dim --n_seeds $n_seeds --n_e_value $n_e --agg_alpha_t $a_t --alpha_t 0.01 --mu_o $mu_o --weight_metric t-test --sv_exp --reuse_xnull --test_purity $test_purity"
				done
			done
		done
	done
done

mus_o=(1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4 4.5 4.6)

models=('OC-SVM')
algorithms=('ConformalOCC')

for algo in ${algorithms[@]}; do
	for mu_o in ${mus_o[@]}; do
    		for model in ${models[@]}; do
    	 		bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha $alpha --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_features $dim --n_seeds $n_seeds --mu_o $mu_o --sv_exp --reuse_xnull --test_purity $test_purity"
    		done
	done
done

algorithms=('E_value_ConformalOCC')
n_e_values=(10)
agg_a_t_list=('avg')

for a_t in ${agg_a_t_list[@]}; do
	for mu_o in ${mus_o[@]}; do
    		for algo in ${algorithms[@]}; do
      			for n_e in ${n_e_values[@]}; do
        			for model in ${models[@]}; do
          				bash ./create_tmp_empty.sh "python ./wrapper.py --model $model --algorithm $algo --alpha $alpha --n_cal $n_cal --n_train $n_train --n_test $n_test --dataset $dataset --save_path $save_path --n_features $dim --n_seeds $n_seeds --n_e_value $n_e --agg_alpha_t $a_t --alpha_t 0.05 --mu_o $mu_o --weight_metric t-test --sv_exp --reuse_xnull --test_purity $test_purity"
        			done
      			done
    		done
	done
done

