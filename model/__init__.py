import logging
logger = logging.getLogger('base')

def create_model(opt):
    model = opt['name']
    # image restoration
    if model == 'model_l_base':  # PSNR-oriented super resolution
        from .model_l_base import UNet_Large_Basic as M
    elif model == 'model_l_pa':  # GAN-based super resolution, SRGAN / ESRGAN
        from .model_l_pa import UNet_Large_PA as M
    elif model == 'model_s_base':
        from .model_s_base import UNet_Small_Base as M
    elif model == 'model_s_pa':
        from .model_s_pa import UNet_Small_PA as M
    elif model == 'model_m_pa':
        from .model_m_pa import UNet_Medium_PA as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    superres = True if 'sr' in opt['model'] else False
    in_channel = opt["in_channel"] if opt.get("in_channel") is not None else 3
    out_channel = opt["out_channel"] if opt.get("in_channel") is not None else 3
    m = M(in_channel=in_channel, out_channel=out_channel ,super_res=superres)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
