# DiffRenderGAN

**Addressing Data Scarcity in Nanomaterial Segmentation Networks with Differentiable Rendering**

This repository contains the implementation of **DiffRenderGAN**, a framework developed to synthesize annotated microscopy images of nanoparticles using differentiable rendering integrated into a generative adversarial network.

The method and experiments are described in the study accepted in *npj Computational Materials*:

> **Title:** Addressing Data Scarcity in Nanomaterial Segmentation Networks with Differentiable Rendering  
> **Journal:** npj Computational Materials  
> **Authors:** [Dennis Possart, Leonid Mill, Florian Vollnhals, Tor Hildebrand, Peter Suter, Mathis Hoffmann,
Jonas Utz, Daniel Augsburger, Mareike Thies, Mingxuan Gu, Fabian Wagner, George Sarau, Silke Christiansen, Katharina Breininger]  
> **DOI:** https://doi.org/10.1038/s41524-025-01702-6

This implementation was tested on:

- **OS:** AlmaLinux 9.2 (Turquoise Kodkod)
- **GPU:** NVIDIA A40, Driver 545.23.08, CUDA 12.3

## Repository Overview

```
.
├── CycleGAN/             # CycleGAN components 
├── datasets/             # Dataset definitions and utilities
├── experiments/          # Pretrained weights and experimental outputs
├── model/                # Model architecture 
├── notebooks/            # Jupyter notebooks for training and testing
├── scripts/              # Scripts for training and testing
├── utils/                # Helper functions and utilities
├── train_gan.py          # Entry point for training
├── test_gan.py           # Entry point for evaluation
├── environment.yml       # Conda environment definition
├── .gitmodules
├── .gitignore
├── LICENSE               # MIT License
└── README.md
```

---

## Installation

We recommend using Conda to manage dependencies:

```
conda env create -f environment.yml
conda activate diffrendergan
```

---

## Data Preparation

Prior to training, microscopy images should be preprocessed into greyscale 256 × 256 pixel patches, each containing at least one nanoparticle instance. 

For details regarding dataset sources and preprocessing pipelines, please refer to the publication. Raw data used in the experiments can be obtained from the studies cited in the paper.


---

## Training

To train DiffRenderGAN, use `train_gan.py`.

Example training scripts illustrating parameter settings are provided in `scripts/train`. See the example scripts and adapt the parameter configuration. 

Additionally, a training notebook (`notebooks/train_diffrendergan.ipynb`) is included for convenience and demonstration purposes

For a complete list of configurable arguments, run:

```
python train_gan.py --help
```
---

## Evaluation and Reproduction of Results

After training, models can be evaluated using `test_gan.py`.

Example testing scripts illustrating are provided in `scripts/test`.

A testing notebook (`notebooks/test_diffrendergan.ipynb`) demonstrates how to load weights, generate synthetic images and masks, and visualize basic results. 

For details on all configurable arguments, run:

```
python test_gan.py --help
```
---

## Planned Releases

We are preparing additional materials that will be released in upcoming updates:

- [✔️] Example Jupyter notebooks demonstrating training, evaluation, and synthetic data generation workflows
- [ ] nnU-Net segmentation results and configuration files corresponding to the experiments in the paper

---

## License

This project is released under the **MIT License**. See the `LICENSE` file for details.

---

## Citation

If you use this code or pretrained models in your research, please cite our publication:

> @article{possart2025addressing,
>  title={Addressing data scarcity in nanomaterial segmentation networks with differentiable rendering and generative modeling},
>  author={Possart, Dennis and Mill, Leonid and Vollnhals, Florian and Hildebrand, Tor and Suter, Peter and Hoffmann, Mathis and Utz, Jonas and Augsburger, Daniel and Thies, Mareike and Gu, Mingxuan and others},
>  journal={npj Computational Materials},
>  volume={11},
>  number={1},
>  pages={197},
>  year={2025},
>  publisher={Nature Publishing Group UK London}}

---

## Contact

For questions regarding the code or experiments, please contact dennis.possart@gmail.com.