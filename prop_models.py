# package
import torch
import torch.nn as nn
import math

# codes
import utils


def prop_model(prop_model, **opt):
    ''' Select propagation model '''
    # Ideal propagation
    if prop_model.lower() in ('asm', 'ideal'):
        return prop_ideal(**opt)
    else:
        raise ValueError(f'Unsupported propagation model: {prop_model}')
    
    
class ASM(nn.Module):
    ''' Ideal ASM propagation. Perform ASM for input 4-D tensor (N,C,H,W) '''
    def __init__(self, feature_size, prop_dists, wvl, aperture):
        ''' 
        Initialization
            :param feature size: a tuple (dy,dx), slm pixel pitch
            :param prop_dists: a nested list [[d1,d2,..], [d1,d2,...]] of CxD shape, prop dist of each wavelength
            :param wvl: a list of length C, wavelengths of light source
        '''
        super(ASM, self).__init__()
        self.feature_size = feature_size
        self.prop_dists = prop_dists
        self.wvl = wvl
        self.aperture = aperture
        self.Hs = None


    def forward(self, input_field, H_aberr=None):
        ''' 
        Forward propagation
            :param  input_field: tensor shape of (N,C,H,W)
            :param Hs: tensor shape of (D,C,2H,2W). If None, use ideal transfer functions
            :param H_aberr: tensor shape of (D,C,2H,2W). Addition aberration term added if not None
            :return output_field: tensor shape of (N,D,C,H,W)
        '''
        # transfer function
        if self.Hs is None:
            self.Hs = self.compute_Hs(input_field)

        # zero-padding
        pad_shape = [2*size for size in input_field.shape[-2:]]
        u1 = utils.pad_image(input_field, pad_shape)

        # propagation
        U1 = utils.FT(u1)
        U2 = U1.unsqueeze(1) * self.Hs.unsqueeze(0)

        # aberration
        if H_aberr is not None:
            U2 = U2 * H_aberr.unsqueeze(0)

        u2 = utils.iFT(U2)

        # crop to original size
        u2 = utils.crop_image(u2, input_field.shape[-2:])

        return u2


    def compute_Hs(self, input_field):
        ''' 
        transfer functions defined by parameters in self
            :param input_field: tensor shape of (...,H,W), propagation field. Determines shape of Hs and GPU device 
            :return Hs: tensor shape of (D,C,H,W)
        '''
        # compute transfer function
        Hs = []
        # for each wavelength
        for dist, w, a in zip(self.prop_dists, self.wvl, self.aperture):
            Hs_depth = []
            # for each depth
            for d in dist:
                H  = self.compute_single_H(input_field.shape[-2:], d, w, a).to(input_field.device)
                Hs_depth.append(H.unsqueeze(0))

            Hs_depth = torch.cat(Hs_depth, dim=0) # (D,H,W) tensor
            Hs.append(Hs_depth.unsqueeze(1))

        Hs = torch.cat(Hs, dim=1) # (D,C,H,W) tensor
        return Hs


    def compute_single_H(self, field_shape, dist, wvl, aperture):
        ''' 
        returns a single transfer function
            :param field_shape: tupe (H,W), shape of propagation field
            :param dist: scalar float, propagation distance
            :param wvl: scalar float, wavelength
            :param aperture: scalar float, normalized aperture size
            :return H: complex tensor shape of (H,W), transfer function
        '''
        # prop parameters
        dy, dx = self.feature_size
        ny, nx = field_shape
        pad_ny, pad_nx = 2*ny, 2*nx

        # frequency domain sampling
        dfx = 1/(pad_nx * dx)
        dfy = 1/(pad_ny * dy)

        # freuency coordinate
        ix = torch.arange(math.ceil(-pad_nx/2),math.ceil(pad_nx/2))
        iy = torch.arange(math.ceil(-pad_ny/2),math.ceil(pad_ny/2))
        FY, FX = torch.meshgrid(iy*dfy, ix*dfx, indexing='ij')

        # bandlimited ASM
        fy_max = 1/math.sqrt((2*dist*(1/(dy*pad_ny)))**2 + 1) / wvl
        fx_max = 1/math.sqrt((2*dist*(1/(dx*pad_nx)))**2 + 1) / wvl
        H_filter = (torch.abs(FX)<fx_max) & (torch.abs(FY)<fy_max)

        # normalized coord
        X = FX / torch.max(FX.abs())
        Y = FY / torch.max(FY.abs())

        # Fourier plane iris
        H_aperture = (X**2 + Y**2 < aperture**2)
        
        # transfer function 
        H_exp = 2 * torch.pi * dist * torch.sqrt(1/wvl**2 - FX**2 - FY**2)
        H = H_aperture * H_filter * torch.exp(1j * H_exp)
        return H


class prop_ideal(nn.Module):
    ''' Ideal propagation model for depolarized-holography '''
    def __init__(self, feature_size, prop_dists, wvl, aperture, **opt):
        '''
        Initialization. All params are for ASM 
            :param feature size: a tuple (dy,dx), slm pixel pitch
            :param prop_dists: a nested list [[d1,d2,..], [d1,d2,...]] of CxD shape, prop dist of each wavelength
            :param wvl: a list of length C, wavelengths of light source
        '''
        super(prop_ideal, self).__init__()
        self.ASM = ASM(feature_size, prop_dists, wvl, aperture)
        self.filter = None
        self.aperture = aperture
        self.pol2idx = {'x':slice(0,1), 'y':slice(1,2), 'xy':slice(0,2)}
        self.pol2amp = {'x':1.0, 'y':1.0, 'xy':1 / math.sqrt(2)}
        self.opt = opt


    def forward(self, slm_phase, meta=None, pol='xy', *args):
        '''
        Propagation with metasurface
            :param slm_phase: tensor shape of (N,C,H,W), 0-2pi slm phase
            :param meta: tensor shape of (P,C,H,W), complex amplitude of metasurface
            :return recon: tensor shape of (N,P,D,C,H,W), reconstructed complex wavefront
        '''
        # phase -> copmlex amplitude
        u_in = torch.exp(1j * slm_phase)

        # 4-f filtering of SLM
        if self.filter is None:
            self.filter = self.init_filter().to(slm_phase.device)
        u_in = apply_filter(u_in, self.filter)

        # metasurface at relayed SLM plane
        u_in = u_in.unsqueeze(1) # add polarization channel, shape of (N,P,C,H,W)
        if meta is not None:
            u_in = u_in * meta.unsqueeze(0)

        # ASM propagation
        u_in = u_in.reshape(-1, *u_in.shape[2:]) # shape of (N*P,C,H,W)
        recon = self.ASM(u_in) # shape of (N*P,D,C,H,W)
        recon = recon.reshape(slm_phase.shape[0], -1, *recon.shape[1:]) # shape of (N,P,D,C,H,W)

        # polarization
        if meta is not None:
            recon = self.pol2amp[pol] * recon[:, self.pol2idx[pol], ...]

        return recon
    

    def init_filter(self):
        # two times bigger fourier filter
        filter_res = [2*s for s in self.opt['slm_res']]
        filter = []

        # for each wavelength
        for a in self.aperture:
            filter.append(fourier_filter(filter_res, a, 'circ').unsqueeze(0))
        filter = torch.cat(filter, dim=0)
        return filter.unsqueeze(0) # shape of (1,C,H,W)


def fourier_filter(res, aperture=1.0, method='circ'):
    ''' Generate Fourier Filter shape of res'''
    ny, nx = res

    ## define coordinates
    ix = torch.arange(math.ceil(-nx/2), math.ceil(nx/2))
    iy = torch.arange(math.ceil(-ny/2), math.ceil(ny/2))
    Y, X = torch.meshgrid(iy, ix, indexing='ij')
    Y = Y / torch.max(Y.abs())
    X = X / torch.max(X.abs())

    if method in ('circ'):
        filter = torch.zeros(res)
        filter[X**2 + Y**2 < aperture**2] = 1.0
    elif method in ('tukey'):
        # radius
        r1 = 0.9
        r2 = 1.05
        R = torch.sqrt(X**2 + Y**2)
        filter = (R > r1*aperture) * (R < r2*aperture) \
                    * torch.cos(math.pi / 2 * (R/aperture - r1) / (r2 - r1))
        filter += (R < r1*aperture)
    else:
        raise ValueError(f'Unsupported Fourier filter method: {method}')
    
    return filter.to(torch.float32)
   

def apply_filter(u_in, filter):
    '''
    Apply fourier filter to input complex wavefront
        :param u_in: tensor shape of (... ,H,W), complex wavefront
        :param filter: tensor shape of (...,2H,2W), filter at Fourier domain
        :return u_out: tensor shape of (...,H,W), filtered wavefront
    '''
    # pad field
    pad_u_in = utils.pad_image(u_in, filter.shape[-2:])

    # convolution
    u_out = utils.iFT(utils.FT(pad_u_in) * filter)

    # crop field
    u_out = utils.crop_image(u_out, u_in.shape[-2:])
    return u_out