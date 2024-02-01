# Depolarized Holography with Polarization-multiplexing Metasurface
### [Paper](https://doi.org/10.1145/3618395) | [arXiv](http://arxiv.org/abs/2309.14668)

[Seung-Woo Nam*](https://nseungwoo.github.io/), 
[Youngjin Kim*](https://youngjin94.github.io/), 
[Dongyeon Kim](https://dongyeon93.github.io/), 
and [Yoonchan Jeong](http://oeqelab.snu.ac.kr/) (\* denotes equal contribution)

Source code for the SIGGRAPH Asia 2023 paper titled "Depolarized Holography with Polarization-multiplexing Metasurface"

## 1. Environment setting
Create anaconda environment. 
Our code has been implemented and tested on Windows.
```
conda env create -f environment.yml
conda activate depolarized-holography
```

## 2. Prerequisites
### 2.1. Metasurface proxy model
We pre-calibrated the metasurface proxy model with electromagnetic response simulated by the [GPU-accelerated rigorous coupled-wave analysis (RCWA) method](https://github.com/kch3782/torcwa). While this method was chosen in our work, alternative methods could also be employed to calibrate the proxy model. The source code requires six coefficients of the model, which should be fitted with second-order polynomials.

To plot the metasurface proxy model, execute the following command:
```
python metasurface.py -c configs/plot_meta_proxy.txt
```
If the output successfully reproduces Figure S4 of the manuscript, it indicates that the proxy model data is consistent with our code implementation.

### 2.2. Image dataset
The joint optimization pipeline for the metasurface and Spatial Light Modulator (SLM) phase patterns requires a 2D image dataset. In our paper, we utilized the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/); however, any 2D image dataset with a sufficient number of images can be used.

Please place your image dataset in the `data` folder and update the `data_path` in the configuration files located in the `configs` directory.

### 2.3. Target image
Our code supports CGH optimization for 2D or focal stack targets. The incoherent focal stack is created using RGB and depth map images, with the depth map appearing darker for objects that are closer. The filenames for the RGB images and their corresponding depth map images should follow the format `{img_name}_rgb.jpg` and `{img_name}_depthmap.jpg`, respectively.

Please place the 2D target images in the `data/RGB` directory and the RGB + depth map images in the `data/RGBD` directory.

## 3. Joint optimization
Jointly optimize the metasurface and the SLM phase patterns by:
```
python train_meta.py -c configs/train_meta.txt
```

## 4. CGH optimization
Conventional CGH optimization using a stochastic gradient descent method:
```
python main.py -c configs/cgh_wo_meta.txt
```

CGH optimization incorporating the optimized polarization-multiplexing metasurface:
```
python main.py -c configs/cgh_w_meta.txt
```
The result includes reconstructed amplitudes for two single polarization states (x, y) as well as a depolarized state (45$\degree$).

## Acknowledgements
We acknowledge that the implemenation of our code was highly inspired by the [Neural Holography](https://github.com/computational-imaging/neural-holography
) and [Holotorch](https://github.com/facebookresearch/holotorch) implementation of the [Neural Etendue Expander](https://arxiv.org/abs/2109.08123). Additionnally, our method for generating the incoherent focal stack from RGB and depthmap images is based on the approach described in the work of [Lee et al.](https://doi.org/10.1038/s41598-022-06405-2).

## Citation
```
@article{Nam2023:DepolarizedHolography,
    author = {Nam, Seung-Woo and Kim, Youngjin and Kim, Dongyeon and Jeong, Yoonchan},
    title = {Depolarized Holography with Polarization-multiplexing Metasurface},
    journal = {ACM Trans. Graph.},
    year = {2023},
    volume = {42},
    number = {6},
    doi = {10.1145/3618395},
}
```