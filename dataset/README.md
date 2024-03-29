# Dataset

We evaluate our RGB-D video prediction model on three datasets: 1. DexHand 2. CALVIN 3. Omnipush

## DexHand

We created a synthetic RGB-D dataset of a Shadow Hand robot manipulating a cube towards arbitrary goal configurations. This dataset consists of about 10K videos, each video including 25 RGB-D frames.

To download the DexHand dataset:
```bash
cd $T3VIP_ROOT/dataset
sh download_data.sh dexhand
```

## CALVIN
We train and validate our models on CALVIN Env C and test them on Env D:

To download the CALVIN dataset:
```bash
cd $T3VIP_ROOT/dataset
sh download_data.sh calvin
```

## Omnipush

We use the first split of Omnipush Dataset, consisting of 70 objects without extra weight.

To download the Omnipush dataset:
```bash
cd $T3VIP_ROOT/dataset
sh download_data.sh omnipush
```