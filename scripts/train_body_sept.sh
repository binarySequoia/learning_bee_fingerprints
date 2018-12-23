#64 Dimensions
CUDA_VISIBLE_DEVICES=1 python ../py_scripts/5_1_train_network_w_dataflow.py -n alex_net64_reg -o body_sept_alex_net64_margin0_5_reg_exp_1 -m 0.5 -a
CUDA_VISIBLE_DEVICES=1 python ../py_scripts/5_1_train_network_w_dataflow.py -n alex_net64_reg -o body_sept_alex_net64_margin1_reg_exp_1 -m 1.0 -a
CUDA_VISIBLE_DEVICES=1 python ../py_scripts/5_1_train_network_w_dataflow.py -n alex_net64_reg -o body_sept_alex_net64_margin2_reg_exp_1 -m 2.0 -a
CUDA_VISIBLE_DEVICES=1 python ../py_scripts/5_1_train_network_w_dataflow.py -n alex_net64_reg -o body_sept_alex_net64_margin5_reg_exp_1 -m 5.0 -a

#128 Dimensions
CUDA_VISIBLE_DEVICES=1 python ../py_scripts/5_1_train_network_w_dataflow.py -n alex_net128_reg -o body_sept_alex_net128_margin0_5_reg_exp_1 -m 0.5 -a
CUDA_VISIBLE_DEVICES=1 python ../py_scripts/5_1_train_network_w_dataflow.py -n alex_net128_reg -o body_sept_alex_net128_margin1_reg_exp_1 -m 1.0 -a
CUDA_VISIBLE_DEVICES=1 python ../py_scripts/5_1_train_network_w_dataflow.py -n alex_net128_reg -o body_sept_alex_net128_margin2_reg_exp_1 -m 2.0 -a
CUDA_VISIBLE_DEVICES=1 python ../py_scripts/5_1_train_network_w_dataflow.py -n alex_net128_reg -o body_sept_alex_net128_margin5_reg_exp_1 -m 5.0 -a

#256 Dimensions
CUDA_VISIBLE_DEVICES=1 python ../py_scripts/5_1_train_network_w_dataflow.py -n alex_net256_reg -o body_sept_alex_net256_margin0_5_reg_exp_1 -m 0.5 -a
CUDA_VISIBLE_DEVICES=1 python ../py_scripts/5_1_train_network_w_dataflow.py -n alex_net256_reg -o body_sept_alex_net256_margin1_reg_v2_exp_1 -m 1.0 -a
CUDA_VISIBLE_DEVICES=1 python ../py_scripts/5_1_train_network_w_dataflow.py -n alex_net256_reg -o body_sept_alex_net256_margin2_reg_v2_exp_1 -m 2.0 -a
CUDA_VISIBLE_DEVICES=1 python ../py_scripts/5_1_train_network_w_dataflow.py -n alex_net256_reg -o body_sept_alex_net256_margin5_reg_v2_exp_1 -m 5.0 -a
