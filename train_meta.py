# package
import torch
import configargparse
import os
import shutil

# codes
import params
import utils
import image_loader
import prop_models
import algorithms
import metasurface


class slm_and_optimizer:
    '''
    slm_phase and slm_optimizer for each batch
    During joint optimization, slm phase profile and corresponding optimizer are saved and loaded for each batch
    '''
    def __init__(self, data_size, slm_res, num_frames, out_path, slm_init_range, lr_slm, dev, **opt):
        self.out_path = os.path.join(out_path, 'slm_and_optimizer')
        self.phase_params = (slm_res, num_frames, slm_init_range, dev)
        self.data_size = data_size
        self.lr_slm = lr_slm
        self.opt = opt

    def initialize(self):
        # initialize & save
        utils.cond_mkdir(self.out_path)
        for i in range(self.data_size):
            self.slm_phase = single_slm(*self.phase_params)
            self.slm_optimizer = torch.optim.Adam([{"params":self.slm_phase.parameters(), "lr":self.lr_slm}])
            self.save(i)
            
    def save(self, idx):
        # save current slm phase and optimizer
        torch.save(self.slm_phase.state_dict(), os.path.join(self.out_path, f'slm_{idx}.pt'))
        torch.save(self.slm_optimizer.state_dict(), os.path.join(self.out_path, f'optim_{idx}.pt'))

    def load(self, idx):
        # load slm phase and optimizer of previous iteration to continue optimization
        self.slm_phase.load_state_dict(torch.load(os.path.join(self.out_path, f'slm_{idx}.pt')))
        self.slm_optimizer.load_state_dict(torch.load(os.path.join(self.out_path, f'optim_{idx}.pt')))
        return self.slm_phase.phase_val, self.slm_optimizer
    
    def clear(self):
        # clear saved temporary file
        if os.path.exists(self.out_path):
            shutil.rmtree(self.out_path)
            print(f'- SLM & optimizer in [{self.out_path}] deleted')


class single_slm(torch.nn.Module):
    '''single slm_phase module'''
    def __init__(self, slm_res, num_frames, slm_init_range, dev):
        super(single_slm, self).__init__()
        init_phase = (2 * torch.pi * slm_init_range * (-0.5 + torch.rand(num_frames, 3, *slm_res))).to(dev)
        self.phase_val = torch.nn.Parameter(init_phase, requires_grad=True)


def main():
    # Command line argument processing of parameters
    torch.set_default_dtype(torch.float32)
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file')
    params.add_parameters(p)
    opt = p.parse_args()
    opt = params.set_configs(opt)

    # 3 color channels are simultaneoulsy optimzed in joint optmization pipeline
    assert opt.num_ch == 3, 'Always full-color for metasurface optimization'

    # Propagation models
    target_loader = image_loader.TargetDataset(**opt)
    SLMs = slm_and_optimizer(**opt)
    forward_prop = prop_models.prop_model(**opt)
    meta = metasurface.get_meta(**opt)
    algorithm = algorithms.load_alg(opt.optim_method)

    # clear intermediate results, if exists
    SLMs.clear()
    SLMs.initialize()
    target_loader.clear()

    best_meta = algorithm(SLMs, target_loader, forward_prop, meta, **opt)

    # save optimization results
    best_meta.save_as_image(opt['out_path'], 'best')
    torch.save(best_meta.state_dict(), os.path.join(opt['out_path'], 'best_meta.pt'))

    # clear saved temporary files
    SLMs.clear()
    target_loader.clear()


if __name__=='__main__':
    main()