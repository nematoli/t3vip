# T3VIP
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[<b>T3VIP: Transformation-based 3D Video Prediction</b>]()

[Iman Nematollahi](http://www2.informatik.uni-freiburg.de/~nematoli/), 
[Erick Rosete Beas](https://erickrosete.com/), 
[Seyed Mahdi B. Azad](), 
[Raghu Rajan](https://ml.informatik.uni-freiburg.de/profile/rajan/), 
[Frank Hutter](https://ml.informatik.uni-freiburg.de/profile/hutter/), 
[Wolfram Burgard](http://www2.informatik.uni-freiburg.de/~burgard)

We present **T3VIP**, 

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
### DexHand
If you want to train on the DexHand dataset:
```bash
cd $T3VIP_ROOT/dataset
sh download_data.sh 
```
### CALVIN

### Omnipush

### Pre-trained Models
We provide our final models 
```bash
cd $T3VIP_ROOT/checkpoints
sh download_model_weights.sh 
```


## Training
```
python 
```

### Ablations
CDNA: Unsupervised Learning for Physical Interaction through Video Prediction, (Finn et al., 2016):

SNA: Self-supervised Visual Planning with Temporal Skip Connections, (Ebert et al., 2017):

SV2P: Stochastic Variational Video Prediction, (Babaeizadeh et al., 2018):
```
python 
```

## Evaluation
```
python 
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