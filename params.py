import os
import utils
from torch.utils.tensorboard import SummaryWriter


# unit of length
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9


def str2bool(v):
    """ 
    Simple query parser for configArgParse (which doesn't support native bool from cmd)
    Ref: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')
    

def str2none(v):
    """ support None as a cmd input """
    if v.lower() in ('none', 'None'):
        return None
    elif isinstance(v, str):
        return v
    else:
        raise ValueError(f'Only str or None available, not {type(v)}')
    

def int2none(v):
    """ support None as a cmd input """
    if v.lower() in ('none', 'None'):
        return None
    elif isinstance(v, str):
        return int(v)
    else:
        raise ValueError(f'Only int or None available, not {type(v)}')
    

class PMap(dict):
    # you can use class as a dictionary (ex. my_class['item_name'])
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def add_parameters(p):
    """ Parameters that can be changed on cmd through argument parser """
    # general
    p.add_argument('--channel', type=int, default=3, help='set color, by default green. 3 for full-color')
    p.add_argument('--slm_type', type=str, default='leto', help='SLM model name')
    p.add_argument('--ls_type', type=str, default='ReadyBeam', help='Laser model')
    p.add_argument('--dev_id', type=int, default=0, help='GPU ID')

    # data
    p.add_argument('--data_path', type=str2none, default=None, nargs='+', help='Data directory')
    p.add_argument('--out_path', type=str2none, default=None, help='Output directory')
    p.add_argument('--meta_path', type=str2none, default=None, help='Optimized metasurface directory')
    p.add_argument('--data_size', type=int2none, default=None, help='Total number of data. If None, use all')
    p.add_argument('--data_type', type=str, default='2d', help='Input data type, 2d/2.5d')
    p.add_argument('--target_type', type=str, default='2d', help='Target data type, 2d/3d')
    p.add_argument('--shuffle', type=str2bool, default=False, help='If True, shuffle data')
    p.add_argument('--augment', type=str2bool, default=False, help='If True, apply data augmentation to image')
    p.add_argument('--resize_than_crop', type=str2bool, default=True, help='If True, resize image rather than crop & pad')
    p.add_argument('--efficient_loader', type=str2bool, default=False, help='If True, save target batch as .pt for fast read-save')

    # SLM
    p.add_argument('--num_frames', type=int, default=1, help='Number of time-multiplexed frames')
    p.add_argument('--slm_init_range', type=float, default=1.0, help='Initial SLM phase range')

    # metasurface
    p.add_argument('--meta_type', type=str2none, default=None, help='Types of metasurface. None/ideal/SiN')
    p.add_argument('--proxy_path', type=str, default='./data/proxy_SiN/fitted_params.mat', help='Metasurface proxy model directory')
    p.add_argument('--meta_init_range', type=float, default=1e-3, help='Metasurface initial random phase or geometry range')
    p.add_argument('--meta_err', type=str2bool, default=False, help='If True, simulate fabrication & alignment error of metasurface')
    p.add_argument('--max_err', type=str2bool, default=False, help='If True, meta_err is maximized')
    p.add_argument('--lr_meta', type=float, default=5e-3, help='Metasurface learning rate')
    p.add_argument('--num_pre_iters', type=int, default=0, help='Number of CGH optimization iterations without metasurface')
    p.add_argument('--num_alt_iters', type=int, nargs=2, default=(1,1), help='(slm, meta) step numbers for alternating updates')
    p.add_argument('--clip_method', type=str, default='naive', help='Method for clip range of meta geometry. naive/ste')
    p.add_argument('--size_init', type=float, default=None, help='Initial offset of geometry of metasurface')

    # optimization
    p.add_argument('--optim_method', type=str, default='sgd', help='Optimizaiton method. sgd/sgd_pol/joint')
    p.add_argument('--num_iters', type=int, default=1000, help='Total iteration number of optimizaiton')
    p.add_argument('--lr_slm', type=float, default=1e-1, help='Learning rate of SLM')
    p.add_argument('--tensorboard', type=str2bool, default=False, help='If True, use Tensorboard writer')

    # propagation
    p.add_argument('--prop_model', type=str, default='ideal', help='Forward propagation model. ideal/model_cnn/model_physical')
    p.add_argument('--prop_dist_0D', type=float, default=20*mm, help='Propagation disatnce from SLM to 0D plane')
    p.add_argument('--eval_plane_idx', type=int, default=0, help='Plane idx of 2D propagation, default 0')
    p.add_argument('--physical_iris', type=str2bool, default=True, help='If True, consider wavelength-dependent aperture')
    p.add_argument('--aperture', type=float, default=1.0, help='Relative aperture size')
    p.add_argument('--rand_focus_plane', type=str2bool, default=False, help='If True, randomly select focused plane. Only valid for 2D data_type')


def set_configs(opt):
    if opt.out_path is not None:
        # create output path
        utils.cond_mkdir(opt.out_path)

    # convert list to str for a single-length data path
    if opt.data_path is not None:
        opt.data_path = opt.data_path[0] if len(opt.data_path) == 1 else opt.data_path

    # configure parameters
    add_params_gpu(opt.dev_id, opt)
    add_params_channel(opt.channel, opt)
    add_params_tensorboard(opt.tensorboard, opt)
    add_params_slm(opt.slm_type, opt)
    add_params_ls(opt.ls_type, opt)
    add_params_prop(opt)
    add_params_meta(opt.meta_type, opt)
    
    return PMap(vars(opt)) # either opt['key'] or opt.key are avalilbale


def add_params_gpu(dev_id, opt):
    # configure gpu
    opt.dev = utils.device(dev_id)


def add_params_channel(channel, opt):
    opt.num_ch = 1 if channel < 3 else 3
    opt.chan_str = ('red', 'green', 'blue', 'rgb')[channel]
    

def add_params_tensorboard(tensorboard, opt, fname='tb'):
    if opt.out_path is not None:
        # type 'tensorboard --logdir {tb_path}' in terminal to open tensorboard
        if tensorboard:
            tb_path = os.path.join(opt.out_path, fname, opt.chan_str)
            utils.cond_mkdir(tb_path)
            opt.writer = SummaryWriter(tb_path)
        else:
            opt.writer = None


def add_params_slm(slm_type, opt):
    ''' Define SLM parameters '''
    if slm_type == 'leto':
        # Holoeye Leto
        opt.slm_res = (1080, 1920)
        opt.roi_res = (900, 1600)
        opt.feature_size = (6.4*um, 6.4*um)
        opt.invert_phase = True

    elif slm_type == 'leto_small':
        # smaller slm for quick debugging
        opt.slm_res = (540, 960)
        opt.roi_res = (450, 800)
        opt.feature_size = (6.4*um, 6.4*um)
        opt.invert_phase = True

    elif slm_type == 'meta_proxy':
        # low-res square coordinate to visulize metasurface proxy model
        opt.slm_res = (100, 100)
        opt.roi_res = (100, 100)
        opt.feature_size = (6.4*um, 6.4*um)
        opt.invert_phase = True

    else:
        raise ValueError(f'Unsupported SLM model: {slm_type}')


def add_params_ls(ls_type, opt):
    ''' Define Light source parameters '''
    if ls_type == 'ReadyBeam':
        opt.rgb_wvls = [638*nm, 520*nm, 450*nm]
    else:
        raise ValueError(f'Unsupported light source: {ls_type}')
        
    if opt.physical_iris:
        # calculate wavelength-dependent aperture size
        opt.aperture = [opt.aperture * opt.rgb_wvls[-1] / w for w in opt.rgb_wvls]    
    else:
        # full aperture
        opt.aperture = [opt.aperture] * 3

    # select color channel
    opt.wvl = [opt.rgb_wvls[opt.channel]] if opt.channel < 3 else opt.rgb_wvls
    opt.aperture = [opt.aperture[opt.channel]] if opt.channel < 3 else opt.aperture

    
def add_params_prop(opt):
    ''' Define propagation parameters '''
    eyepiece = 50*mm # eyepiece focal length
    min_d, max_d = [0.0, 3.0] # diopter range of front-last focal plane
    num_planes = 7 # number focal planes
    diopters = [min_d + idx / (num_planes-1) * (max_d - min_d) for idx in range(num_planes)] # plane distance in Diopter
    opt.prop_dists = utils.diopter_to_dist(diopters, eyepiece, opt.prop_dist_0D, 0) # plane distance in meter
        
    # prop_dists for focal stack generation
    opt.fs_prop_dists = opt.prop_dists
        
    # replication for RGB distance
    opt.prop_dists = [opt.prop_dists, opt.prop_dists, opt.prop_dists]

    # select single plane for 2D supervision
    if opt.target_type == '2d':
        opt.prop_dists = [[d[opt.eval_plane_idx]] for d in opt.prop_dists]

    # select channel
    if opt.channel < 3:
        opt.prop_dists = [opt.prop_dists[opt.channel]]

    # finalized number of planes
    opt.num_planes = len(opt.prop_dists[0])
        

def add_params_meta(meta_type, opt):
    if meta_type is not None:
        # maximum alignment and fabrication error
        opt.align_err = 10
        opt.fab_err = 0.018

        if meta_type in ('ideal',):
            opt.size_limit = None

        elif meta_type in ('SiN',):
            opt.size_limit = (0.283, 0.776)
            opt.size_init = (opt.size_limit[0] + opt.size_limit[1]) / 2 if opt.size_init is None else opt.size_init

        else:
            raise ValueError(f'Unsupported metasurface type: {meta_type}')