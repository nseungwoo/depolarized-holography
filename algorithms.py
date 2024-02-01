# package
import torch
import math
from tqdm import tqdm

# codes
import utils


def load_alg(alg):
    """Choose optimization algorithm"""
    if alg == "sgd":
        # CGH optimization
        return sgd

    elif alg == "sgd_pol":
        # polarization-dependent CGH optimization
        return sgd_pol

    elif alg == "joint":
        # Metasurface - SLM phase joint optimization
        return joint

    else:
        raise ValueError(f"Unsupported optimization method: {alg}")


def sgd(init_phase, target, forward_prop, meta=None, loss_fn=torch.nn.MSELoss(), **opt):
    '''
    SGD for single SLM optimization
        :param init_phase: tesnor shape of (N,C,H,W)
        :param target: tensor shape of (D,C,H',W'), (H',W') is roi_res
        :param forward_prop: forward propagation model from prop_models
        :param meta: metasurface model (nn.Module) defined in metasurface.py
        :param loss_fn: torch loss
        :return : dictionary of optimization results
    '''
    # initialization
    slm_phase = init_phase.clone().detach()
    slm_phase.requires_grad_(True)
    optimizer = torch.optim.Adam([{"params":slm_phase, "lr":opt['lr_slm']}])

    best_loss = 1e9

    # iteration
    pbar = tqdm(range(opt['num_iters']), ascii=True, ncols=200)
    for iter in pbar:
        optimizer.zero_grad()

        # complex amplitude of metasurface
        meta_complex = None if meta is None else meta(opt['meta_err'], opt['max_err'])    

        # propagation
        recon = forward_prop(slm_phase=slm_phase, meta=meta_complex)
        recon = utils.crop_image(recon, opt['roi_res'])
        recon_amp_pol = utils.incoherent_sum(recon, dim=0, method='avg') # avg for TM. Polarizatoin-dependent amplitude
        recon_amp = utils.incoherent_sum(recon_amp_pol, dim=0, method='sum') # sum for polarization

        # Loss
        s = utils.compute_scale(recon_amp, target, dim=(0,2,3))
        loss_val = loss_fn(s * recon_amp, target)

        # backprop
        loss_val.backward()
        optimizer.step()

        with torch.no_grad():
            # compute psnr
            psnr_val = utils.get_psnr(s * recon_amp, target)

            # report progress
            pbar.set_postfix({'psnr': psnr_val.item(), \
                                'scale': [f'{ss:.2f}' for ss in s[0,:,0,0].tolist()] })

            # update best results
            if loss_val < best_loss:
                best_phase = slm_phase
                best_field = s.unsqueeze(0).unsqueeze(0) * recon 
                best_amp = s * recon_amp # (D,C,H',W')
                best_amp_pol = math.sqrt(2) * s * recon_amp_pol # (P,D,C,H',W')
                best_psnr = utils.get_psnr(s * recon_amp, target).item()
                best_scale = s.squeeze().tolist()

    return {'best_phase': best_phase, # (N,C,H,W)
            'best_field': best_field, # (N,P,D,C,H',W')
            'best_amp': best_amp, # (D,C,H',W')
            'best_amp_pol': best_amp_pol, # (P,D,C,H',W')
            'best_psnr': best_psnr, # scalar
            'best_scale': best_scale}

 
def sgd_pol():
    pass


def joint(slm_loader, target_loader, forward_prop, meta=None, loss_fn=torch.nn.MSELoss(), **opt):
    '''
    Joint SGD for SLM & metasurface
        :param slm_loader: slm_phase and slm_optimizer loader in train_meta.py
        :param target_loader: target dataset loader, returns (target_image, target_name)
        :param forward_prop: forward propagation model from prop_models
        :param meta: metasurface model (nn.Module) defined in metasurface.py
        :param loss_fn: torch loss
        :return : dictionary of optimization results
    '''
    # tensorboard
    tb_writer = opt['writer'] if opt['tensorboard'] else None
    avg_loss = [0] * opt['num_iters']
    avg_psnr = [0] * opt['num_iters']

    '''optimize only slm'''
    pbar = tqdm(target_loader, desc='pre_iters', ascii=True, ncols=200)
    for idx, batch in enumerate(pbar):
        # target
        target_image, _ = batch
        # load optimizer & slm
        slm_phase, slm_optimizer = slm_loader.load(idx)
        
        for iter in range(opt['num_pre_iters']):
            slm_optimizer.zero_grad()

            # propagation
            recon = forward_prop(slm_phase=slm_phase, meta=meta(meta_err=False))
            recon = utils.crop_image(recon, opt['roi_res'])
            recon_amp = utils.incoherent_sum(recon, dim=0, method='avg') # avg for TM
            recon_amp = utils.incoherent_sum(recon_amp, dim=0, method='sum') # sum for polarization

            # Loss
            s = utils.compute_scale(recon_amp, target_image, dim=(0,2,3))
            loss_val = loss_fn(s * recon_amp, target_image)

            # backprop
            loss_val.backward()
            slm_optimizer.step()

            with torch.no_grad():
                # save loss and psnr
                psnr_val = utils.get_psnr(s * recon_amp, target_image)
                avg_loss[iter] += loss_val.item() / opt['data_size']
                avg_psnr[iter] += psnr_val.item() / opt['data_size']

        # save optimizer & slm
        slm_loader.save(idx)

    # tensorboard
    for i in range(opt['num_pre_iters']):
        if tb_writer is not None:
            tb_writer.add_scalar("Train Epoch/Loss", avg_loss[i], i)
            tb_writer.add_scalar("Train Epoch/PSNR", avg_psnr[i], i)


    '''joint optimization'''
    # meta optimizer
    meta_optimizer = torch.optim.Adam([{"params":meta.parameters(), "lr":opt['lr_meta']}])
    best_loss = 1e9

    pbar = tqdm(range(opt['num_pre_iters'], opt['num_iters']), desc='joint', ascii=True, ncols=200)
    for iter in pbar:
        # alternating update of SLM and metasurface
        slm_update = (iter % (opt['num_alt_iters'][0] + opt['num_alt_iters'][1])) < opt['num_alt_iters'][0]
        # metasurface is perfectly aligned during SLM update
        meta_err = False if slm_update else opt['meta_err']

        # single update for every target image
        for idx, batch in enumerate(target_loader):
            # load target
            target_image, _ = batch

            # load SLM
            slm_phase, slm_optimizer = slm_loader.load(idx)
            slm_optimizer.zero_grad()
            meta_optimizer.zero_grad()

            # propagation
            recon = forward_prop(slm_phase=slm_phase, meta=meta(meta_err=meta_err))
            recon = utils.crop_image(recon, opt['roi_res'])
            recon_amp = utils.incoherent_sum(recon, dim=0, method='avg') # avg for TM
            recon_amp = utils.incoherent_sum(recon_amp, dim=0, method='sum') # sum for polarization

            # Loss
            s = utils.compute_scale(recon_amp, target_image, dim=(0,2,3))
            loss_val = loss_fn(s * recon_amp, target_image)

            # backprop
            if slm_update:
                loss_val.backward()
                slm_optimizer.step()
            else:
                loss_val.backward() 
                meta_optimizer.step() 
            
            # save slm
            slm_loader.save(idx)

            with torch.no_grad():
                # save loss and psnr
                psnr_val = utils.get_psnr(s * recon_amp, target_image)
                avg_loss[iter] += loss_val.item() / opt['data_size']
                avg_psnr[iter] += psnr_val.item() / opt['data_size']

        with torch.no_grad():
            # update best results
            if best_loss > avg_loss[iter]:
                best_meta = meta

            # update tensorboard
            if tb_writer is not None:
                tb_writer.add_scalar("Train Epoch/Loss", avg_loss[iter], iter)
                tb_writer.add_scalar("Train Epoch/PSNR", avg_psnr[iter], iter)
            
    return best_meta
