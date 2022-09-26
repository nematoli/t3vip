# T3VIP
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[<b>T3VIP: Transformation-based 3D Video Prediction</b>](https://arxiv.org/pdf/2209.11693.pdf)

[Iman Nematollahi](http://www2.informatik.uni-freiburg.de/~nematoli/), 
[Erick Rosete Beas](https://erickrosete.com/), 
[Seyed Mahdi B. Azad](), 
[Raghu Rajan](https://ml.informatik.uni-freiburg.de/profile/rajan/), 
[Frank Hutter](https://ml.informatik.uni-freiburg.de/profile/hutter/), 
[Wolfram Burgard](http://www2.informatik.uni-freiburg.de/~burgard)

We present **T3VIP**, a transformation-based 3D video prediction approach that explicitly models the 3D motion by decomposing a scene into its object parts and predicting their corresponding rigid transformations. Our model is fully unsupervised, captures the stochastic nature of the real world, and the observational cues in image and point cloud domains constitute its learning signals. To the best of our knowledge, our model is the first generative model that provides an RGB-D video prediction of the future for a static camera. Given the ability to learn long-range RGB-D predictions of the future from unlabeled experience, we hope that this repo paves the way of employing 3D world models for robots in the real world.

![](media/demo.gif)


## Installation
To begin, clone this repository locally
```bash
git clone https://github.com/nematoli/t3vip.git
export T3VIP_ROOT=$(pwd)/t3vip

```
Install requirements:
```bash
cd T3VIP_ROOT
conda create -n t3vip_venv python=3.7
conda activate t3vip_venv
sh install.sh
```

## Download
### Datasets
To download datasets used in this work, please follow [this guide](dataset/README.md).

### Pre-trained Models
To download our final T3VIP models on different datasets, please follow [this guide](checkpoints/README.md).


## Training
To train T3VIP models on each dataset you need to specify the corresponding datamodule and model configs. 
```
python train.py datamodule=dexhand model=T3VIP_dex 
python train.py datamodule=omnipush model=T3VIP_omni 
python train.py datamodule=calvin datamodule.dataset.env=env_c model=T3VIP_calvin 
```

If you want to skip frames during data loading (e.g. 2 frames), you need to set `datamodule.skip_frames=2`.

If you have access to slurm cluster, please follow [this guide](slurm_scripts/README.md).

### Ablations
we compare our stochastic RGB-D video prediction model to the following well-established RGB video prediction baselines:

CDNA: Unsupervised Learning for Physical Interaction through Video Prediction, (Finn et al., 2016):
```
python train.py datamodule=dexhand model=CDNA 
```

SNA: Self-supervised Visual Planning with Temporal Skip Connections, (Ebert et al., 2017):
```
python train.py datamodule=dexhand model=SNA 
```
SV2P: Stochastic Variational Video Prediction, (Babaeizadeh et al., 2018):
```
python train.py datamodule=dexhand model=SV2P 
```

You can change the datamodule variable to be either dexhand, calvin or omnipush.

## Evaluation
You can evaluate your final trained model on test sets. The `evaluate.py` script will fetch the latest checkpoint and run the evaluation. You need to specify the corresponding `datamodule` and `model` for your run. For example, to evaluate T3VIP on omnipush test set, we run:

```
python evaluate.py datamodule=omnipush model=T3VIP_omni
```
If you want to change the length of test videos sequences, you can modify `eval_seq_len`.

## Automated Hyperparameter Optimization
In order to automatically find the best set of hyperparameters of T3VIP for your own dataset, you need to set ray scheduler parameters such that it suits your resources and dataset. We used the following configs for finding hyperparameters with our datasets having access to 8 parallel gpus:

For DexHand dataset:
```
python train_hpo.py datamodule=dexhand model=T3VIP ray.reports_per_epoch=1 ray.scheduler.max_t=63 ray.scheduler.grace_period=1 ray.num_samples=200 trainer.limit_val_batches=0.5
```
For Calvin dataset:
```
python train_hpo.py datamodule=calvin datamodule.dataset.env=env_c model=T3VIP ray.reports_per_epoch=27 ray.scheduler.max_t=27 ray.scheduler.grace_period=1 ray.num_samples=200 trainer.limit_val_batches=0.1 
```

For Omnipush dataset:
```
python train_hpo.py datamodule=omnipush model=T3VIP ray.reports_per_epoch=1 ray.scheduler.max_t=252 ray.scheduler.grace_period=1 ray.num_samples=200 trainer.limit_val_batches=1.0
```


## Citation

If you find the code useful, please cite:

**T3VIP**
```bibtex
@inproceedings{nematollahi22iros,
    author  = {Iman Nematollahi and Erick Rosete-Beas and Seyed Mahdi B. Azad and Raghu Rajan and Frank Hutter and Wolfram Burgard}
    title   = {T3VIP: Transformation-based 3D Video Prediction},
    booktitle={Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    year = 2022,
    url={http://ais.informatik.uni-freiburg.de/publications/papers/nematollahi22iros.pdf}
    address = {Kyoto, Japan}
}
```

## License

MIT License
