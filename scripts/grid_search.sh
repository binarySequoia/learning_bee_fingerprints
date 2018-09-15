#python ../py_scripts/5_train_network.py -n alex_net128 -o alex_net128_margin0_2 -m 0.2 -e 1
#python ../py_scripts/5_train_network.py -n alex_net128 -o alex_net128_margin0_5 -m 0.5 -e 1
#python ../py_scripts/5_train_network.py -n alex_net128 -o alex_net128_margin1 -m 1.0 -e 1
#python ../py_scripts/5_train_network.py -n alex_net128 -o alex_net128_margin2 -m 2.0 -e 1
#python ../py_scripts/5_train_network.py -n alex_net128 -o alex_net128_margin5 -m 5.0 -e 1


#python ../py_scripts/5_train_network.py -n alex_net64 -o alex_net64_margin0_2 -m 0.2 -e 1
#python ../py_scripts/5_train_network.py -n alex_net64 -o alex_net64_margin0_5 -m 0.5 -e 1
#python ../py_scripts/5_train_network.py -n alex_net64 -o alex_net64_margin1 -m 1.0 -e 1
#python ../py_scripts/5_train_network.py -n alex_net64 -o alex_net64_margin2 -m 2.0 -e 1
#python ../py_scripts/5_train_network.py -n alex_net64 -o alex_net64_margin5 -m 5.0 -e 1

#python ../py_scripts/5_train_network.py -n alex_net128_reg -o alex_net128_margin0_2_reg -m 0.2 -e 1
#python ../py_scripts/5_train_network.py -n alex_net128_reg -o alex_net128_margin0_5_reg -m 0.5 -e 1
#python ../py_scripts/5_train_network.py -n alex_net128_reg -o alex_net128_margin1_reg -m 1.0 -e 1
#python ../py_scripts/5_train_network.py -n alex_net128_reg -o alex_net128_margin2_reg -m 2.0 -e 1
#python ../py_scripts/5_train_network.py -n alex_net128_reg -o alex_net128_margin5_reg -m 5.0 -e 1


#python ../py_scripts/5_train_network.py -n alex_net64_reg -o alex_net64_margin0_2_reg -m 0.2 -e 1
#python ../py_scripts/5_train_network.py -n alex_net64_reg -o alex_net64_margin0_5_reg -m 0.5 -e 1
#python ../py_scripts/5_train_network.py -n alex_net64_reg -o alex_net64_margin1_reg -m 1.0 -e 1
#python ../py_scripts/5_train_network.py -n alex_net64_reg -o alex_net64_margin2_reg -m 2.0 -e 1
#python ../py_scripts/5_train_network.py -n alex_net64_reg -o alex_net64_margin5_reg -m 5.0 -e 1


python ../py_scripts/5_train_network.py -n alex_net128_reg -o alex_net128_margin2_reg_alpha_1_25 -m 2.0 -e 1 --alpha 1.25
python ../py_scripts/5_train_network.py -n alex_net128_reg -o alex_net128_margin2_reg_alpha_1_50 -m 2.0 -e 1 --alpha 1.50
python ../py_scripts/5_train_network.py -n alex_net128_reg -o alex_net128_margin2_reg_alpha_2_0 -m 2.0 -e 1 --alpha 2.0

python ../py_scripts/5_train_network.py -n alex_net128 -o alex_net128_margin2_alpha_1_25 -m 2.0 -e 1 --alpha 1.25
python ../py_scripts/5_train_network.py -n alex_net128 -o alex_net128_margin2_alpha_1_50 -m 2.0 -e 1 --alpha 1.50
python ../py_scripts/5_train_network.py -n alex_net128 -o alex_net128_margin2_alpha_2_0 -m 2.0 -e 1 --alpha 2.0

python ../py_scripts/5_train_network.py -n alex_net64 -o alex_net64_margin2_alpha_1_25 -m 2.0 -e 1 --alpha 1.25
python ../py_scripts/5_train_network.py -n alex_net64 -o alex_net64_margin2_alpha_1_50 -m 2.0 -e 1 --alpha 1.50
python ../py_scripts/5_train_network.py -n alex_net64 -o alex_net64_margin2 -m 2.0 -e 1 --alpha 2.0

python ../py_scripts/5_train_network.py -n alex_net64_reg -o alex_net64_margin2_reg_alpha_1_25 -m 2.0 -e 1 --alpha 1.25
python ../py_scripts/5_train_network.py -n alex_net64_reg -o alex_net64_margin2_reg_alpha_1_50 -m 2.0 -e 1 --alpha 1.50
python ../py_scripts/5_train_network.py -n alex_net64_reg -o alex_net64_margin2_reg_alpha_2_0 -m 2.0 -e 1 --alpha 2.0