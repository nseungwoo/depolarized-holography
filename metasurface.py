# package
import torch
import torch.nn as nn
import random
import os
import math
import scipy.io
import matplotlib.pyplot as plt

# codes
import utils


def get_meta(meta_type, **opt):
    # no metasurface
    if meta_type is None:
        return None
    
    # ideal meta with fully independent modulation for polarization & wavelength
    elif meta_type == 'ideal':
        return ideal_meta(**opt)
    
    # meta made of SiN, modeled by proxy function
    elif meta_type in ('SiN',):
        return proxy_meta(**opt)
    
    else:
        raise ValueError(f'Unsupported metasurface type: {meta_type}')


class ideal_meta(nn.Module):
    def __init__(self, slm_res, meta_init_range, **opt):
        '''
        Initialize ideal metasurface profile. Always full-color
            :param slm_res: tuple, slm_res (=meta_res)
            :param meta_init_range: scalar float, random phase range of initial meta
        '''
        super(ideal_meta, self).__init__()
        self.opt = opt
        self.meta_phase = nn.Parameter(self.init_meta(slm_res, meta_init_range).detach(), requires_grad=True)


    def forward(self, meta_err=False, max_err=False):
        '''
        Forward function of metasurface
            :param meta_err: bool. If True, simulate fabrication & align error
            :param max_err: bool. If True, simulate meta with maximum error
            :return out: tensor shape of (P,C,H,W), complex amplitude of metasurface
        '''
        meta_phase = fabrication(self.meta_phase, meta_err, max_err, self.opt['fab_err'])
        meta_complex = torch.exp(1j * meta_phase)
        return misalign(meta_complex, meta_err, max_err, self.opt['align_err'])
    

    def init_meta(self, meta_res, init_range):
        ''' 
        Initialize metasurface 
            :param meta_res: tuple, metasurface resolution
            :param init_range: scalar float, random phase range of initial meta
            :return meta_phase: float tensor shape of (2,1,H,W), metasurface phase profile
        '''
        init_phase = 2 * torch.pi * init_range * (-0.5 + torch.rand(2, 1, *meta_res))
        return init_phase.to(self.opt['dev'])
    
    
    def save_as_image(self, path, idx=0):
        ''' Save phase profile as images '''
        meta_path = os.path.join(path, 'ideal_meta_phase')
        utils.cond_mkdir(meta_path)

        # save pol & color
        for P, pol in enumerate(('x', 'y')):
            phase = (self.meta_phase[P,...] + torch.pi) / (2 * torch.pi)
            utils.imsave(os.path.join(meta_path, f'pol_{pol}_iter_{idx}.png'), phase)


class proxy_meta(nn.Module):
    def __init__(self, slm_res, meta_init_range, size_limit, size_init, **opt):
        super(proxy_meta, self).__init__()
        self.opt = opt
        self.size_limit = size_limit
        self.size_init = size_init
        # Geometry map
        # NOTE: (self.H, self.W) here are equivalent (l,w) in the mauscript
        self.H = nn.Parameter(self.init_meta(slm_res, meta_init_range).detach(), requires_grad=True)
        self.W = nn.Parameter(self.init_meta(slm_res, meta_init_range).detach(), requires_grad=True)
        # proxy
        self.read_proxy_data(opt['proxy_path'])

        # clip method
        if opt['clip_method'] in ('naive'):
            self.clip = torch.clip
        elif opt['clip_method'] in ('ste'):
            self.clip = utils.clip_ste.apply


    def forward(self, meta_err=False, max_err=False):
        # get geometry map
        H = fabrication(self.H, meta_err, max_err, self.opt['fab_err'], self.size_limit, self.clip)
        W = fabrication(self.W, meta_err, max_err, self.opt['fab_err'], self.size_limit, self.clip)

        # get complex profile
        meta_complex, _, _ = self.complex_from_geometry(H, W)

        # misalignment
        meta_complex = misalign(meta_complex, meta_err, max_err, self.opt['align_err'])

        # single color
        if self.opt['channel'] < 3:
            meta_complex = meta_complex[:,self.opt['channel']:self.opt['channel']+1,:,:]

        return meta_complex # shape of (2,C,H,W)


    def init_meta(self, meta_res, init_range):
        ''' Initialize geometry (H,W) '''
        size_range = self.size_limit[1] - self.size_limit[0]
        init_geometry = init_range * size_range * (-0.5 + torch.rand(*meta_res)) \
                            + self.size_init
        return init_geometry.to(self.opt['dev'])


    def read_proxy_data(self, path):
        # read polynomial coefficients from matlab file
        coeffs = scipy.io.loadmat(path)

        # load data
        self.abs_t_xx = torch.tensor(coeffs['abs_t_xx']).to(torch.float32)
        self.abs_t_yy = torch.tensor(coeffs['abs_t_yy']).to(torch.float32)
        self.arg_t_xx = torch.tensor(coeffs['arg_t_xx']).to(torch.float32)
        self.arg_t_yy = torch.tensor(coeffs['arg_t_yy']).to(torch.float32)
         
        ## reshape coefficients [3,n] to [n,1,3,1,1]
        self.abs_t_xx = self.abs_t_xx.transpose(1,0).view(-1,1,3,1,1).to(self.opt['dev']).requires_grad_(False)
        self.abs_t_yy = self.abs_t_yy.transpose(1,0).view(-1,1,3,1,1).to(self.opt['dev']).requires_grad_(False)
        self.arg_t_xx = self.arg_t_xx.transpose(1,0).view(-1,1,3,1,1).to(self.opt['dev']).requires_grad_(False)
        self.arg_t_yy = self.arg_t_yy.transpose(1,0).view(-1,1,3,1,1).to(self.opt['dev']).requires_grad_(False)

        # order of polynomials
        self.order = math.floor(math.sqrt(self.abs_t_xx.shape[0] * 2))
        

    def complex_from_geometry(self, H, W):
        '''Calculate complex amplitude of metasurface from geomtry maps'''
        # get polynomials of H, W such as (H^2 * W)
        polynomials = gen_polynomials(H, W, self.order) # (order,H,W)
        polynomials = polynomials.view(polynomials.shape[0], 1, 1, *polynomials.shape[-2:]) # (order,1,1,H,W)

        # phase
        phase_xx = torch.sum(self.arg_t_xx * polynomials, dim=0)
        phase_yy = torch.sum(self.arg_t_yy * polynomials, dim=0)
        meta_phase = torch.cat((phase_xx, phase_yy), 0)
        meta_phase = 2 * torch.pi * (meta_phase - 0.5) # proxy outputs [0,1] range, convert to zero-centered 2pi

        # amplitude
        abs_xx = torch.sum(self.abs_t_xx * polynomials, dim=0)
        abs_yy = torch.sum(self.abs_t_yy * polynomials, dim=0)
        meta_abs = torch.cat((abs_xx, abs_yy), 0)

        return (meta_abs * torch.exp(1j* meta_phase)), meta_phase, meta_abs # all shape of (2,3,H,W)


    def save_as_image(self, path, idx=0):
        ''' Save metasurface phase and amplitude as image '''
        meta_path = os.path.join(path, 'SiN_meta_profile')
        utils.cond_mkdir(meta_path)

        # geometry
        H = fabrication(self.H, meta_err=False, size_limit=self.size_limit, clip_fn=self.clip)
        W = fabrication(self.W, meta_err=False, size_limit=self.size_limit, clip_fn=self.clip)
        utils.cond_mkdir(os.path.join(meta_path, 'geometry'))
        utils.imsave(os.path.join(meta_path, 'geometry', f'H_{idx}.png'), H)
        utils.imsave(os.path.join(meta_path, 'geometry', f'W_{idx}.png'), W)

        # meta profile
        _, meta_phase, meta_abs = self.complex_from_geometry(H,W)
        meta_phase = ((meta_phase + torch.pi) % (2 * torch.pi)) / (2 * torch.pi)
        utils.cond_mkdir(os.path.join(meta_path, 'phase'))
        utils.cond_mkdir(os.path.join(meta_path, 'amplitude'))
        
        for P, pol_str in enumerate(('x', 'y')):
            for C, chan_str in enumerate(('red', 'green', 'blue')):
                fname = f'{chan_str}_{pol_str}_{idx}.png'
                utils.imsave(os.path.join(meta_path, 'phase', fname), meta_phase[P,C,...])
                utils.imsave(os.path.join(meta_path, 'amplitude', fname), meta_abs[P,C,...])


    def plot_proxy(self):
        '''Plot metasurface proxy model'''
        # generate H,W meshgrid map 
        grid_size = 100

        # TODO: range limited to opt.size_limit
        geometry_axis = torch.linspace(*self.size_limit, grid_size).to(self.opt['dev'])
        w, l = torch.meshgrid(geometry_axis, geometry_axis, indexing='ij')

        # phase of metasurface
        _, phase, _ = self.complex_from_geometry(l, w)            

        # plot
        for ch in range(3):
            fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

            for pol in range(2):
                # normalize with 2pi
                data = (phase[pol, ch, :, :].squeeze().cpu().numpy()) / (2 * math.pi) + 0.5
                
                # Plot the surface
                axs[pol].plot_surface(l.cpu().numpy(), w.cpu().numpy(), data, cmap='viridis')

                # axis
                axs[pol].set_zlim([0, 1.2])
                axs[pol].invert_yaxis()
                
                # label
                axs[pol].set_xlabel('l')
                axs[pol].set_ylabel('w')

                # title
                title = '2$\pi$ normalized phase: ' + ('Red (638 nm)', 'Green (520 nm)', 'Blue (450 nm)')[ch]
                fig.suptitle(title, rotation=0, size='large')
                
                for pol in range(2):
                    axs[pol].set_title(('x-pol', 'y-pol')[pol])
                
        plt.show()


def load_meta(meta, opt):
    '''Load optimized metasurface module from saved pt file'''
    if opt.meta_path is not None:
        print(f'- Metasurface loaded from [{opt.meta_path}]')
        meta.load_state_dict(torch.load(os.path.join(opt.meta_path, 'best_meta.pt'), map_location=opt.dev))
        meta.eval()
    else:
        if opt.meta_type is None:
            print('- No metasurface')
        else:
            print('- No metasurface loaded. It should exist inside propagation model')
        meta = None
    return meta


def fabrication(meta_profile, meta_err=False, max_err=False, fab_err=None, size_limit=None, clip_fn=torch.clip):
    ''' Simulate fabrication error that affects phase profile
        :param meta_profile: phase or geomtry of metasurface, tensor of any shape
        :param meta_err: bool. If True, simulate fabrication error
        :param fab_err: float, standard deviation of fabrication error
        :param size_limit: tuple of floats (min, max). Limit of metasurface 
    '''
    if meta_err:
        # constant error for all pixel
        if max_err:
            meta_fab_err = torch.tensor(fab_err)
        else:
            meta_fab_err = fab_err * torch.randn(1).abs()
        meta_profile = meta_profile + meta_fab_err.to(meta_profile.device)

    if size_limit is not None:
        meta_profile = clip_fn(meta_profile, *size_limit)

    return meta_profile


def misalign(meta_complex, meta_err=False, max_err=False, align_err=None):
    ''' Simulate alignemnt error to complex profile '''
    if meta_err:
        # maximum misalignment
        if max_err:
            [align_x, align_y] = [align_err] * 2
        # randomly selected misalignment
        else:
            [align_x, align_y] = [random.randint(-align_err, align_err) for i in range(2)]

        # padding, (left, right, top, bottom)
        pad_size = (max(0,align_x), max(0,-align_x), max(0,align_y), max(0,-align_y))
        meta_complex = nn.functional.pad(meta_complex, pad_size, mode='constant', value=1.0)
        # shift
        meta_complex = meta_complex[..., 0:-align_y, :] if align_y > 0 \
                            else meta_complex[..., -align_y:, :]
        meta_complex = meta_complex[..., :, 0:-align_x] if align_x > 0 \
                            else meta_complex[..., :, -align_x:]
    return meta_complex


def gen_polynomials(H, W, order=1):
    '''generate polynoimals of H and W such as (H^2*W, H*W^2, ...)
        :param H, W: tensors, any identical shape
        :return polys: tensor shape of [order, *var.shape]
    '''
    H = H.unsqueeze(0) # H
    W = W.unsqueeze(0) # W

    # Generate polynomials
    polys = []          
    for i in range(order):
        for j in range(i+1):
            polys.append( (W ** (i-j)) * (H ** (j)))
    polys = torch.cat(polys, dim=0)

    return polys


if __name__ == '__main__':
    '''Visualize metasurface proxy model'''
    import configargparse
    import params
    import matplotlib.pyplot as plt

    # Command line argument processing of parameters
    torch.set_default_dtype(torch.float32)
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file')
    params.add_parameters(p)
    opt = p.parse_args()
    opt = params.set_configs(opt)

    # metasurface
    meta = get_meta(**opt)
    meta.plot_proxy()
    