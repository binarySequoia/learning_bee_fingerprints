NETWORKS=(alex_net alex_net_softmax alex_net_sigmoid alex_net_dbl_sigmoid vggnet_8 vggnet_8_softmax vggnet_8_sigmoid vggnet_8_dbl_sigmoid vggnet_8v2 vggnet_8v2_softmax vggnet_8v2_sigmoid vggnet_8v2_dbl_sigmoid vggnet_6 vggnet_6_softmax vggnet_6_sigmoid vggnet_6_dbl_sigmoid)

for i in ${NETWORKS[@]}
do
    python ../py_scripts/6_model_stats.py -n $i
done