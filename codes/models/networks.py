import torch
import logging
import math
logger = logging.getLogger('base')

from models.modules.Inv_arch import *
from models.modules.Subnet_constructor import subnet


def define_G(opt):
    opt_net = opt['network_G']
    print('network_G: ', opt_net)
    which_model = opt_net['which_model_G']
    print('which_model: ', which_model)
    if which_model == 'EDSR' or  which_model == 'DBC':
        from models.modules.EDSR_arch import EDSR_arch
        netG = EDSR_arch()
    elif which_model == 'IRN':
        subnet_type = which_model['subnet_type']
        if opt_net['init']:
            init = opt_net['init']
        else:
            init = 'xavier'
        down_num = int(math.log(opt_net['scale'], 2))
        netG = InvRescaleNet(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init), opt_net['block_num'], down_num)
    else:
        raise NotImplementedError('G model [{:s}] not recognized'.format(which_model))

    return netG


def define_Analysis(opt):
    opt_net = opt['network_A']
    which_model = opt_net['which_model_A']
    if which_model == 'EDSR' or which_model == 'DBC':
        from models.modules.EDSR_arch import analysis_arch
    netA = analysis_arch()
    return netA

def define_S(opt):
    opt_net = opt['network_syntax']
    which_model = opt_net['which_model_syntax']
    if which_model == 'Neural_Syntax':
        from models.modules.EDSR_arch import Net
    netS = Net()
    return netS
