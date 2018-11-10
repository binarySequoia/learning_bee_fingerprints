import sys
sys.path.append("../networks")

from alex_net.alex_net import alex_net
from alex_net.alex_net_softmax import alex_net_softmax
from alex_net.alex_net_sigmoid import alex_net_sigmoid
from alex_net.alex_net_dbl_sigmoid import alex_net_dbl_sigmoid
from alex_net.alex_net128 import alex_net128
from alex_net.alex_net64 import alex_net64
from alex_net.alex_net128_reg import alex_net128_reg
from alex_net.alex_net64_reg import alex_net64_reg
from alex_net.alex_net256_reg import alex_net256_reg

from vgg16.vggnet_6 import vggnet_6
from vgg16.vggnet_6_softmax import vggnet_6_softmax
from vgg16.vggnet_6_sigmoid import vggnet_6_sigmoid
from vgg16.vggnet_6_dbl_sigmoid import vggnet_6_dbl_sigmoid

from vgg16.vggnet_8 import vggnet_8
from vgg16.vggnet_8_softmax import vggnet_8_softmax
from vgg16.vggnet_8_sigmoid import vggnet_8_sigmoid
from vgg16.vggnet_8_dbl_sigmoid import vggnet_8_dbl_sigmoid

from vgg16.vggnet_8v2 import vggnet_8v2
from vgg16.vggnet_8v2_softmax import vggnet_8v2_softmax
from vgg16.vggnet_8v2_sigmoid import vggnet_8v2_sigmoid
from vgg16.vggnet_8v2_dbl_sigmoid import vggnet_8v2_dbl_sigmoid

from inception.one_layer_inception import one_layer_inception

from shallow.shallow import shallow

NETWORK_ZOO = {
    "alex_net" : alex_net,
    "alex_net_softmax" : alex_net_softmax,
    "alex_net_sigmoid" : alex_net_sigmoid,
    "alex_net_dbl_sigmoid" : alex_net_dbl_sigmoid,
    "vggnet_6" : vggnet_6,
    "vggnet_6_softmax" : vggnet_6_softmax,
    "vggnet_6_sigmoid" : vggnet_6_sigmoid,
    "vggnet_6_dbl_sigmoid" : vggnet_6_dbl_sigmoid,
    "vggnet_8" : vggnet_8,
    "vggnet_8_softmax" : vggnet_8_softmax,
    "vggnet_8_sigmoid" : vggnet_8_sigmoid,
    "vggnet_8_dbl_sigmoid" : vggnet_8_dbl_sigmoid,
    "vggnet_8v2" : vggnet_8v2,
    "vggnet_8v2_softmax" : vggnet_8v2_softmax,
    "vggnet_8v2_sigmoid" : vggnet_8v2_sigmoid,
    "vggnet_8v2_dbl_sigmoid" : vggnet_8v2_dbl_sigmoid,
    "alex_net128": alex_net128,
    "alex_net64": alex_net64,
    "one_layer_inception" : one_layer_inception,
    "alex_net128_reg" : alex_net128_reg,
    "alex_net64_reg" : alex_net64_reg,
    "alex_net256_reg" : alex_net256_reg,
    "shallow": shallow
}

def get_networks_list():
    return NETWORK_ZOO.keys()

def base_network(name, input_shape=[230, 105, 3]):
    
    if name in NETWORK_ZOO.keys():
        return NETWORK_ZOO[name](input_shape)
    else:
        net_list = "\t\t\n".join(get_networks_list())
        print("Available Networks:\n" + net_list)
        raise Exception("No network found with name {}".format(name))

        

    