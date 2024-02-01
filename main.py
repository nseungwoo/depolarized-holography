# package
import torch
import configargparse
import os

# codes
import params
import utils
import image_loader
import prop_models
import algorithms
import metasurface


def main():
    # Command line argument processing / Parameters
    torch.set_default_dtype(torch.float32)
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file')
    params.add_parameters(p)
    opt = p.parse_args()
    opt = params.set_configs(opt)

    # Models
    target_loader = image_loader.TargetDataset(**opt)
    forward_prop = prop_models.prop_model(**opt)
    meta = metasurface.get_meta(**opt)
    algorithm = algorithms.load_alg(opt.optim_method)

    opt.data_size = len(target_loader) if opt.data_size is None else opt.data_size

    # load optimized metasurface if needed
    meta = load_meta(meta, opt)

    # Iteration
    for idx, batch in enumerate(target_loader):
        # target from batch
        target_amp, target_name = batch
        print(f'- target: {target_name}')

        # initialization
        init_phase = (2 * torch.pi * opt['slm_init_range'] \
                    * (-0.5 + torch.rand(opt['num_frames'], opt['num_ch'], *opt['slm_res']))).to(opt['dev'])

        # optimization
        results = algorithm(init_phase, target_amp, forward_prop, meta, **opt)

        # save results
        visualize_and_save(results, target_name, **opt)


def load_meta(meta, opt):
    if opt.meta_path is not None:
        print(f'- Meta loaded from [{opt.meta_path}]')
        meta.load_state_dict(torch.load(os.path.join(opt.meta_path, 'best_meta.pt'), map_location=opt.dev))
        meta.eval()
    else:
        print('- CGH optimized witout meta')
        meta = None
    return meta


def visualize_and_save(results, target_name, **opt):
    ''' Visualize optimization results and save them '''
    batch_path = os.path.join(opt['out_path'], target_name)
    utils.cond_mkdir(batch_path)
    chan_str = opt['chan_str']

    # SLM
    for N in range(results['best_phase'].shape[0]):
        slm = (results['best_phase'][N, ...] + torch.pi) % (2 * torch.pi) / (2 * torch.pi)
        fname = os.path.join(batch_path, f'slm_{chan_str}_{N}.bmp')
        utils.imsave(fname, slm, invert=opt['invert_phase'])

    # recon amplitude
    for D in range(results['best_amp'].shape[0]):
        recon = results['best_amp'][D, ...]
        # save as image
        fname = os.path.join(batch_path, f'recon_{chan_str}_{D}.png')
        utils.imsave(fname, recon)

    # polarization-dependent amplitude
    if opt['meta_type'] is not None:
        for P in range(results['best_amp_pol'].shape[0]):
            for D in range(results['best_amp_pol'].shape[1]):
                recon = results['best_amp_pol'][P, D, ...]
                # save as image
                pol = ('x', 'y')[P]
                fname = os.path.join(batch_path, f'recon_{chan_str}_{D}_{pol}.png')
                utils.imsave(fname, recon)


if __name__=='__main__':
    main()