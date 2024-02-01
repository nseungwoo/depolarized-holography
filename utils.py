import os
import torch
import numpy as np
import skimage.io
import skimage.transform
# Warning: dtype sholud be float32 / complex64


def cond_mkdir(path):
    '''Create a new directory if not exists'''
    if not os.path.exists(path):
        os.makedirs(path)

        
def device(dev_id=0):
    '''Select GPU'''
    dev = torch.device(f'cuda:{dev_id}' if torch.cuda.is_available() else 'cpu')
    print('--', torch.cuda.get_device_name(dev_id), ' is available', f'cuda:{dev_id} selected --')
    return dev


def diopter_to_dist(diopters, focal_length, plane_dist, plane_idx=0):
    '''
    Convert dipotric distance seen through eyepiece into physical distance
        :param diopters: list of diopters to convert
        :param focal_length: scalar float, focal length of eyepiece
        :param plane_dist: scalar float, distance from SLM to a reference plane
        :param plane_idx: scalar int, index of reference plane
        :return dists: list of physical distance 
    '''
    dists = [1/(diopter + 1/focal_length) for diopter in diopters]
    z0 = dists[plane_idx]
    dists = [plane_dist - dist + z0 for dist in dists]
    return dists


def im2float(img, dtype=np.float32):
    """
    convert input image to float
    normalize to [0,1] if it is integer type such as uint8
        :param img: numpy image
    """
    if np.issubdtype(img.dtype, np.floating):
        return img.astype(dtype)
    elif np.issubdtype(img.dtype, np.integer):
        return img / dtype(np.iinfo(img.dtype).max)
    else:
        raise ValueError(f'Unsupported data type{img.dtype}')
    

def pad_image(img, target_shape, pad_val=0.0):
    '''
    Pad input image with pad_val
    Applies padding for first two dimensions for numpy, last two dimensions for tensor
        :param img: numpy or tensor image
        :param target_shape: tuple, target shape for padding
        :param pad_val: scalar
        :return img: padded image
    '''
    target_shape = np.array(target_shape)

    # numpy image
    if isinstance(img, np.ndarray):
        img_shape = np.array(img.shape)[0:2]
    # tensor
    elif isinstance(img, torch.Tensor):
        img_shape = np.array(img.shape)[-2:]
    else:
        raise ValueError(f'Unsupported data type for padding: {type(img)}')
    
    # calculate pad size
    if target_shape[0]<img_shape[0] or target_shape[1]<img_shape[1]:
        raise ValueError(f'Padding target size is smaller than input size: {target_shape} and {img_shape}')
    else:
        pad_front = target_shape//2 - img_shape//2
        pad_end = (target_shape+1)//2 - (img_shape+1)//2   

    # padding
    if isinstance(img, np.ndarray):
        # (top, bottom, left, right)
        if len(img.shape) == 3:
            pad_size = ((pad_front[0], pad_end[0]), (pad_front[1], pad_end[1]), (0,0))
        else:
            pad_size = ((pad_front[0], pad_end[0]), (pad_front[1], pad_end[1]))
        img = np.pad(img, pad_size, mode='constant', constant_values=pad_val)
    elif isinstance(img, torch.Tensor):
        # (left, right, top, bottom)
        pad_size = (pad_front[1], pad_end[1], pad_front[0], pad_end[0]) 
        img = torch.nn.functional.pad(img, pad_size, mode='constant', value=pad_val)

    return img

    
def crop_image(img, target_shape):
    '''
    Pad input image to target resolution
    Applies crop for first two dimensions for numpy, last two dimensions for tensor
        :param img: numpy or tensor image
        :param target_shape: tuple, target shape for crop
        :return img: cropped image
    '''
    # numpy image
    if isinstance(img, np.ndarray):
        img_shape = np.array(img.shape)[0:2]
    # tensor image
    elif isinstance(img, torch.Tensor):
        img_shape = np.array(img.shape)[-2:]
    else:
        raise ValueError(f'Unsupported data type for crop: {type(img)}')

    # target shape
    target_shape = np.array(target_shape)
    crop_front = img_shape//2 - target_shape//2
    crop_end = (img_shape+1)//2 - (target_shape+1)//2
    crop_slice = [slice(int(f), int(-e) if e else None) \
                   for f, e in zip(crop_front, crop_end)] # [slice(), slice()]
    
    # crop
    if isinstance(img, np.ndarray):
        img = img[(*crop_slice, ...)]
    elif isinstance(img, torch.Tensor):
        img = img[(..., *crop_slice)]

    return img


def srgb_gamma2lin(img):
    """
    Converts from sRGB to linear color space
    Used when read images
        :param img: any numpy or tensor, normalized to float [0,1]
    """
    thr = 0.04045
    if torch.is_tensor(img): # tensor
        low_val = img <= thr
        img_out = torch.zeros_like(img)
        img_out[low_val] = 25/323 * img[low_val]
        img_out[torch.logical_not(low_val)] = ((200*img[torch.logical_not(low_val)] + 11) / 211) ** (12/5)
    else: # numpy
        img_out = np.where(img <= thr, img / 12.92, ((img + 0.055) / 1.055) ** (12/5))

    return img_out


def FT(tensor):
    """ Perform 2D fft of a tensor for last two dimensions """
    tensor_shift = torch.fft.ifftshift(tensor, dim=(-2,-1))
    tensor_ft_shift = torch.fft.fft2(tensor_shift, norm='ortho')
    tensor_ft = torch.fft.fftshift(tensor_ft_shift, dim=(-2,-1))
    return tensor_ft


def iFT(tensor):
    """ Perform 2D ifft of a tensor for last two dimensions """
    tensor_shift = torch.fft.ifftshift(tensor, dim=(-2,-1))
    tensor_ift_shift = torch.fft.ifft2(tensor_shift, norm='ortho')
    tensor_ift = torch.fft.fftshift(tensor_ift_shift, dim=(-2,-1))
    return tensor_ift


def get_psnr(img1, img2):
    ''' Compute PSNR of two tensor images, range of [0,1]'''
    with torch.no_grad():
        mse = torch.mean((255 * (img1 - img2)) ** 2)
        return 20 * torch.log10(255.0 / mse.sqrt())    


def incoherent_sum(field, dim=(0), method='avg', keepdims=False):
    ''' 
    Perform incoherent summation
        :param field: tensor
        :param dim: dimension for summation
        :param method: str avg or sum
        :param keepdims: If True, keep shape of field
    '''
    if method == 'avg':
        return (field.abs() ** 2).mean(dim=dim, keepdims=keepdims).sqrt()
    elif method == 'sum':
        return (field.abs() ** 2).sum(dim=dim, keepdims=keepdims).sqrt()
    else:
        raise ValueError(f'Unsupported method: {method}')


def compute_scale(recon, target, dim=None):
    '''
    Compute scale that minimizes MSE btw recon and target
        :param recon, target: float tensor, should have same shape
        :param dim: tuple, dimensions that use same scale
    '''
    with torch.no_grad():
        # minimize MSE
        s = torch.mean((recon * target), dim=dim, keepdim=True) \
            / torch.mean(recon ** 2, dim=dim, keepdim=True)
        # Nan -> 0
        s = torch.nan_to_num(s, nan=0.0)
        return s
    
    
def imsave(path, img, res=None, invert=False):
    '''
    Save normalized tensor as an image 
        :param path: str, path for save
        :param img: tensor shape of (C,H,W), clipped to [0,1]
        :param res: tuple, resolution of saved image
    '''
    # convert to numpy
    img = img.squeeze()
    if len(img.shape) == 3:
        img = img.permute(1,2,0)
    img = img.cpu().detach().numpy()

    if res is not None:
        img = skimage.transform.resize(img ,res)
    
    # clip and convert to uint8
    if invert:
        img = 1.0 - img
    img = (255 * np.clip(img, 0, 1)).astype(np.uint8)
    
    # save image
    skimage.io.imsave(path, img, check_contrast=False)

    
class clip_ste(torch.autograd.Function):
    @staticmethod
    def forward(ctx, field, min_val, max_val, *args, **kwargs):
        out_field = torch.clip(field, min_val, max_val)
        maskout_grad = torch.ones_like(out_field)
        # save out for backward
        ctx.save_for_backward(maskout_grad)
        return out_field

    def backward(ctx, grad_output):
        maskout_grad, = ctx.saved_tensors
        return grad_output * maskout_grad, None, None
    