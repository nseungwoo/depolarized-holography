# package
import os
import random
import numpy as np
import torch
import skimage.io
import skimage.transform
from torch.utils.data import Dataset
import shutil

# codes
import utils
import incoherent_focal_stack


# image file extensions
img_exts = ('jpg', 'bmp', 'png')


def get_dirs(path):
    ''' Get all folders under path and return as a list '''
    dirs = os.listdir(path)
    dirs = [os.path.join(path, d) for d in dirs if os.path.isdir(os.path.join(path, d))] # get only folders
    return dirs # list of (path + folder)


def get_image_filenames(path):
    ''' Get all filenames of image under path '''
    files = os.listdir(path)
    files = [os.path.join(path, f) for f in files if f[-3:] in img_exts]
    return files # list of (path + file)


def get_2d_fnames(data_path):
    ''' 
    Get 2d data filenames
        :param data_path: str, file or folder name
        :return filenames: list of filenames
    option1) data_path = file name --> read only single image
    option2) data_path = folder name --> read all images under the folder
    '''
    # file
    if os.path.isfile(data_path):
        filenames = [data_path]
    # folder
    elif os.path.isdir(data_path):
        filenames = get_image_filenames(data_path)
    else:
        raise ValueError(f'Data path does not exist: {data_path}')

    return filenames 


def get_rgbd_fnames(data_path):
    '''
    Get rgbd (2.5d) data filenames
        :param data_path: str, file or folder name
        :return img_names: list of rgb image file names
        :return depth_names: list of depthmap image file names
    option1) data_path = file name --> read only single image of '{data_path}_rgb.ext' and '{data_path}_depth.ext'
    option2) data_path = folder name --> read all images under the folder
    '''
    # file
    if any([os.path.isfile(f'{data_path}_rgb.{ext}') for ext in img_exts]):
        img_fnames = [f'{data_path}_rgb.{ext}' for ext in img_exts if os.path.isfile(f'{data_path}_rgb.{ext}')]
        depth_fnames = [f'{data_path}_depth.{ext}' for ext in img_exts if os.path.isfile(f'{data_path}_depth.{ext}')]
    # folder
    elif os.path.isdir(data_path):
        fnames = get_image_filenames(data_path)
        fnames.sort()
        img_fnames = fnames[1::2]
        depth_fnames = fnames[::2]
    else:
        raise ValueError(f'Data path does not exist: {data_path}')

    return img_fnames, depth_fnames # both are list of (path + file)


def imread(path, gamma2lin=True, augment_state=0, **opt):    
    '''
    Read image with many options
        :param path: image filepath
        :param gamma2lin: convert image into amplitude
        :param augment_state: types of flip augmentation
        :return img: (C,H,W) tensor. NOTE: (H,W) is roi_res, not slm_res
    '''
    # read image file
    img = skimage.io.imread(path)
    
    # convert to 3 channel for gray images
    if len(img.shape) < 3:
        img = np.repeat(img[:,:,np.newaxis], 3, axis=2)

    # rotate 90 degree for vertical image
    if img.shape[0] > img.shape[1]:
        img = img.transpose(1,0,2)
    
    # resize image while keeping ratio
    if opt['resize_than_crop']:
        resize_ratio = max(opt['roi_res'][0] / img.shape[0], opt['roi_res'][1] / img.shape[1])
        resize_shape = [round(s * resize_ratio) for s in img.shape[0:2]]
        img = skimage.transform.resize(img, resize_shape)

    img = utils.im2float(img)

    # pad & crop image
    pad_size = [max(s1, s2) for s1, s2 in zip(opt['slm_res'], img.shape[0:2])]
    img = utils.crop_image(utils.pad_image(img, pad_size), opt['roi_res'])

    # exclude alpha channel if exists
    if img.shape[-1] > 3:
        img = img[:,:,0:3]

    # single channel image
    if opt['channel'] < 3:
        img = img[..., opt['channel'], np.newaxis] # (H,W,1)

    # convert to amplitude
    if gamma2lin:
        img = utils.srgb_gamma2lin(img)
        img = np.sqrt(img)

    # apply augmentation
    if augment_state % 4 == 1:
        img = img[::-1, :, :] # vertical flip
    elif augment_state % 4 == 2:
        img = img[:, ::-1, :] # horizontal flip
    elif augment_state % 4 == 3:
        img = img[::-1, ::-1, :] # vertical & horizontal flip

    # np to tensor
    img = torch.from_numpy(img.copy()).to(opt['dev']).permute(2,0,1)

    return img # tensor, shape of (C,H,W)


class TargetDataset(Dataset):
    def __init__(self, data_path, data_type, target_type, **opt):
        # parameters
        self.data_path = data_path
        self.data_type = data_type
        self.target_type = target_type
        self.opt = opt

        # get data filenames
        if self.data_type == '2d':
            self.target_names = get_2d_fnames(self.data_path)
            self.target_names.sort()
        elif self.data_type == '2.5d':
            self.target_names, self.depth_names = get_rgbd_fnames(self.data_path)
        
        # data size
        self.data_size = len(self.target_names) \
            if opt['data_size'] is None or (opt['data_size'] > len(self.target_names)) \
            else opt['data_size']

        # data order
        self.order = [i for i in range(self.data_size)]
        if opt['shuffle']:
            random.shuffle(self.order)
            
        # data augmentation
        if opt['augment']:
            self.augment_states = [random.randint(0,4) for i in range(self.data_size)]
        else:
            self.augment_states = [0 for i in range(self.data_size)]

    
    def __len__(self):
        return self.data_size
    
        
    def __getitem__(self, idx):
        # data idx
        idx = self.order[idx]

        # extract image name only
        target_name = os.path.splitext(os.path.split(self.target_names[idx])[1])[0] 
        if self.data_type == '2d':
            target_name = target_name + '_2D'
        elif self.data_type == '2.5d':
            target_name = target_name[:-4] + '_RGBD'

        # load from saved target if exists
        if self.opt['efficient_loader']:
            fname = os.path.join(self.opt['out_path'], 'target', f'{idx}.pt')
            if os.path.exists(fname):
                target_image = torch.load(fname)
                return target_image, target_name
        
        # read data (C,H,W)
        if self.data_type == '2d':
            target_image = self.load_2d(idx)
            target_depth = None
        elif self.data_type == '2.5d':
            target_image, target_depth = self.load_rgbd(idx)
        else:
            raise ValueError(f'Unsuppored data_type: {self.data_type}')
        
        # manage target type
        if self.target_type == '2d':
            target_image = target_image.unsqueeze(0)
        elif self.target_type == '3d':
            target_depth_mask = self.mask_from_depth(target_depth)
            # generate focal stack from RGBD
            target_image = incoherent_focal_stack.Incoherent_focal_stack(image=target_image,
                                                                         depth_mask=target_depth_mask,
                                                                         prop_dists=self.opt['fs_prop_dists'],
                                                                         wvl=self.opt['wvl'],
                                                                         feature_size=self.opt['feature_size'],
                                                                         aperture=self.opt['aperture'])
        else:
            raise ValueError(f'Unsuppored data_type: {self.data_type}')
        
        # save target if needed
        if self.opt['efficient_loader']:
            utils.cond_mkdir(os.path.join(self.opt['out_path'], 'target'))
            fname = os.path.join(self.opt['out_path'], 'target', f'{idx}.pt')
            if not os.path.exists(fname):
                torch.save(target_image, fname)
        
        return target_image, target_name # tensor shape of (D,C,H,W), str


    def load_2d(self, idx):
        ''' Load 2D image for input idx '''
        target_image = imread(self.target_names[idx], 
                              gamma2lin=True,
                              augment_state=self.augment_states[idx],
                              **self.opt)
        return target_image # (C,H,W)
    

    def load_rgbd(self, idx):
        ''' Load 2D image and depthmap for input idx '''
        target_image = imread(self.target_names[idx],
                              gamma2lin=True, 
                              augment_state=self.augment_states[idx],
                              **self.opt)
        
        target_depth = imread(self.depth_names[idx],
                              gamma2lin=False,
                              augment_state=self.augment_states[idx],
                              **self.opt)[0:1,:,:]
        
        return target_image, target_depth # both (C,H,W)
    

    def mask_from_depth(self, depthmap):
        # 2D withouth depthmap
        if depthmap is None:
            # midplane is focused plane
            plane_idx = self.opt['num_planes'] // 2

            # depth mask for single plane
            target_mask = torch.zeros(self.opt['num_planes'], *self.opt['roi_res']).to(self.opt['dev'])
            target_mask[plane_idx,:,:] = 1
            
        # RGBD
        else:
            target_mask = incoherent_focal_stack.gen_depthmap_mask(depthmap, self.opt['num_planes'])

        return target_mask
    

    def clear(self):
        # delte temporary .pt files saved during iterations
        if self.opt['efficient_loader']:
            path = os.path.join(self.opt['out_path'], 'target')
            if os.path.exists(path):
                shutil.rmtree(path)
                print(f'- Target data in [{path}] deleted')
