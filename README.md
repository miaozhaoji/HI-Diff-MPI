## Contents

1. [Installation](#Installation)
1. [Datasets](#Datasets)
1. [Models](#Models)
1. [Training](#Training)
1. [Testing](#Testing)
1. [Results](#Results)
1. [Citation](#Citation)
1. [Acknowledgements](Acknowledgements)

## Installation

- Python 3.9
- PyTorch 1.9.0
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

```bash
# Clone the github repo and go to the default directory 'HI-Diff'.
git clone https://github.com/zhengchen1999/HI-Diff.git
cd HI-Diff
conda create -n hi_diff python=3.9
conda activate hi_diff
pip install -r requirements.txt
```



## Training



  Stage-1 (S1) 
  ```shell

  python train.py -opt options/train/Simu_S1.yml
  ```shell

  Stage-2 (S2)
  
  python train.py -opt options/train/Simu_S2.yml

  


- The training experiment is in `experiments/`.

## Testing

- Download the pre-trained [models](https://drive.google.com/drive/folders/1X3oos6dmtDDo9IqC6SK5RiujMYE6Y22q?usp=drive_link) and place them in `experiments/pretrained_models/`.

- Download [test](https://drive.google.com/file/d/1pUFsJQleqCGTeeHnsSukJU0oSbjjWIJP/view?usp=drive_link) (GoPro, HIDE, RealBlur) datasets, place them in `datasets/`.

- Run the following scripts. The testing configuration is in `options/test/`.

  Synthetic, reproduces results in Table 2 of the main paper

  ```python
  # generate images
  python test.py -opt options/test/GoPro.yml
  # test PSNR/SSIM
  evaluate_gopro_hide.m
  python evaluate_realblur.py --dataset RealBlur_R --dir results/test_HI_Diff_GoPro
  python evaluate_realblur.py --dataset RealBlur_J --dir results/test_HI_Diff_GoPro
  ```

  Real-World, RealBlur-R, reproduces results in Table 3 of the main paper

  ```python
  # generate images
  python test.py -opt options/test/RealBlur_R.yml
  # test PSNR/SSIM
  python evaluate_realblur.py --dataset RealBlur_R --dir results/test_HI_Diff_RealBlur_R
  ```

  Real-World, RealBlur-J, reproduces results in Table 3 of the main paper

  ```python
  # generate images
  python test.py -opt options/test/RealBlur_J.yml
  # test PSNR/SSIM
  python evaluate_realblur.py --dataset RealBlur_J --dir results/test_HI_Diff_RealBlur_J
  ```

- The output is in `results/`.

## Results

We achieved state-of-the-art performance on synthetic and real-world blur dataset. Detailed results can be found in the paper.

<details>
<summary>Evaluation on Synthetic Datasets (click to expand)</summary>

- quantitative comparisons in Table 2 of the main paper

<p align="center">
  <img width="900" src="figs/Tab-1.png">
</p>

- visual comparison in Figure 4 of the main paper

<p align="center">
  <img width="900" src="figs/Fig-1.png">
</p>
</details>

<details>
<summary>Evaluation on Real-World Datasets (click to expand)</summary>


- quantitative comparisons in Table 3 of the main paper

<p align="center">
  <img width="900" src="figs/Tab-2.png">
</p>

- visual comparison in Figure 5 of the main paper

<p align="center">
  <img width="900" src="figs/Fig-2.png">
</p>

</details>

## Citation

If you find the code helpful in your resarch or work, please cite the following paper(s).

```
@inproceedings{chen2023hierarchical,
  title={Hierarchical Integration Diffusion Model for Realistic Image Deblurring}, 
  author={Chen, Zheng and Zhang, Yulun and Ding, Liu and Bin, Xia and Gu, Jinjin and Kong, Linghe and Yuan, Xin},
  booktitle={NeurIPS},
  year={2023}
}
```

## Acknowledgements

This code is built on  [BasicSR](https://github.com/XPixelGroup/BasicSR), [Restormer](https://github.com/swz30/Restormer), and [DiffIR](https://github.com/Zj-BinXia/DiffIR).
