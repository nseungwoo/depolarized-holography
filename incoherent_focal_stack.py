import torch
import torch.fft
import math


def ITF(field, z, dx, dy, wavelength, aperture=1.0):
    """
    Simulate Incoherent Transfer Fuction
        :param field: shape of (H,W), tensor
        :param z: scalar, propagation distance
        :param dx, dy: pixel pitch
        :return out: shape of (1,1,H,W)
    """
    m, n = field.shape
    Lx, Ly = float(dx * n), float(dy * m)

    angX = math.asin(wavelength / (2 * dx))
    angY = math.asin(wavelength / (2 * dy))
    marX = math.fabs(z) * math.tan(angX)
    marX = math.ceil(marX / dx)
    marY = math.fabs(z) * math.tan(angY)
    marY = math.ceil(marY / dy)
    pad_field = torch.nn.functional.pad(field, (marX, marX, marY, marY)).to(field.device)

    fy = torch.linspace(-1 / (2 * dy) + 0.5 / (2 * Ly), 1 / (2 * dy) - 0.5 / (2 * Ly), m+ 2*marY).to(field.device)
    fx = torch.linspace(-1 / (2 * dx) + 0.5 / (2 * Lx), 1 / (2 * dx) - 0.5 / (2 * Lx), n+ 2*marX).to(field.device)
    dfx = (1 / dx) / n
    dfy = (1 / dy) / m
    fY, fX = torch.meshgrid(fy, fx, indexing='ij')

    # circular fourier filter
    nfX = fX / torch.max(fX.abs())
    nfY = fY / torch.max(fY.abs())
    BL_FILTER = (nfX**2 + nfY**2 < aperture**2)

    # energy normalization
    BL_FILTER = BL_FILTER / torch.sqrt(torch.sum(BL_FILTER) / torch.numel(BL_FILTER))

    # set transfer function
    GammaSq = (1 / wavelength) ** 2 - fX ** 2 - fY ** 2
    TF = torch.exp(-2 * 1j * math.pi * torch.sqrt(GammaSq) * z)
    TF = TF * BL_FILTER
    cpsf = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(TF), norm='ortho'))  # coherent psf
    ipsf = torch.abs(cpsf) ** 2  # incoherent psf
    OTF = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(ipsf), norm='ortho'))

    # apply transfer function
    max_fx = 1 / (wavelength * ((2 * dfx * z) ** 2 + 1) ** 0.5)
    max_fy = 1 / (wavelength * ((2 * dfy * z) ** 2 + 1) ** 0.5)
    FT = (torch.abs(fX) < max_fx) * (torch.abs(fY) < max_fy)  # aliasing
    AS = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(pad_field), norm='ortho'))
    PropagatedField = abs(torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(AS * OTF * FT), norm='ortho')))
    out = PropagatedField[marY:m+marY,marX:n+marX]  # crop

    return out.unsqueeze(0).unsqueeze(0)


def Incoherent_focal_stack(image, depth_mask, prop_dists, wvl, feature_size, aperture, alpha=0.5):
    """
    Generate incoherent propagation focal stack from a single focused image and depth masks
        :param image: shape of (C,H,W), target rgb tensor
        :param depth_mask: shape of (D,H,W), target depthmap mask
        :param prop_dists: len D list, propagation distance of each plane. Far to near.
        :param wavelengths: tuple of wavelengths
        :param feature_size: tuple, SLM pixel ptich
        :param aperture: tuple, aperture size of each color channel
        :param alpha: coefficient for occlusion blending
        :return: shape of (D,C,H,W), simulated focal stack
    """
    n_channel, ny, nx = image.shape
    Np = len(prop_dists)
    dev = image.device

    # initialize focal stack
    layer_sum_color = torch.zeros(Np, n_channel, ny, nx).to(dev) # [D,C,H,W]

    for ch, wavelength in enumerate(wvl):
        for n in range(Np):
            layer_sum = torch.zeros(1, 1, ny, nx).to(dev) # [1,1,H,W]
            mask = torch.zeros(1, Np, ny, nx).to(dev) # [1,D,H,W]

            # To achieve k-th focal stack, apply incoherent propagation for all Np layers
            for k in range(Np):
                dz = prop_dists[n] - prop_dists[k]
                k_depth_mask = depth_mask[k, :, :] # [1,1,H,W] depth-k mask

                # Generate occlusion mask function by sequential propagation
                mask[:, k, :, :] = ITF(k_depth_mask.squeeze(), dz, *feature_size, wavelength, aperture[ch])
                mask[:, k, :, :] = 1.0 - mask[:, k, :, :] / torch.max(mask[:, k, :, :])
                mask[:, k, :, :] = torch.nan_to_num(mask[:, k, :, :], 1.0)

                # Incoherent propagation and Summation
                layer_intensity = (image[ch,:,:].abs() ** 2).unsqueeze(0) * k_depth_mask # [1,1,H,W]
                layer_intensity_prop = ITF(layer_intensity.squeeze(), dz, *feature_size, wavelength, aperture[ch])
                if k == 0: 
                    # first layer
                    layer_sum = (1.0 * mask[:,k,:,:]) * layer_sum + layer_intensity_prop
                elif k == (Np - 1): 
                    # last layer
                    layer_sum = (alpha * mask[:,k,:,:] + (1.0 - alpha) * mask[:,k-1,:,:]) * layer_sum \
                                    + alpha * layer_intensity_prop
                else:
                    # intermediate layer
                    layer_sum = (alpha * mask[:,k,:,:] + (1.0 - alpha) * mask[:,k-1,:,:]) * layer_sum + layer_intensity_prop

            # intensity to amplitude
            layer_sum_color[n, ch, :, :] = torch.sqrt(layer_sum.abs()) 

    return layer_sum_color / torch.max(layer_sum_color)


def gen_depthmap_mask(depthmap, num_planes):
    """
    Divide depthmap into binary masks with linear spacing
        :param depthmap: tensor shape of (C,H,W), normalized to [0,1]. NOTE: closer object is darker
        :param num_planes: scalar int, number of planes to divide
        :return depthmap_mask: tensor shape of (D,H,W)
    """
    # normalized division dists
    division_dists = torch.linspace(1, 0, 2 * num_planes - 1)[1::2]
    division_dists = [2.0, *division_dists, -1.0]

    # generate mask from far to close
    depthmap = depthmap[0,:,:]
    depthmap_mask = []

    for idx in range(len(division_dists) - 1):
        # distance range
        far_dist = division_dists[idx]
        near_dist = division_dists[idx+1]
    
        tmp_mask = torch.ones_like(depthmap)
        mask_idx = (depthmap > far_dist)
        tmp_mask[mask_idx] = 0

        mask_idx = (depthmap <= near_dist)
        tmp_mask[mask_idx] = 0

        depthmap_mask.append(tmp_mask.unsqueeze(0))

    depthmap_mask = torch.cat(depthmap_mask, dim=0)
    
    return depthmap_mask # shape of (D,H,W)