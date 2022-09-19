## DexHand

To download the final T3VIP model for DexHand dataset:
```bash
cd $T3VIP_ROOT/checkpoints
sh download_model_weights.sh dexhand
```

## CALVIN
We trained T3VIP on CALIVN dataset once without any skipping frames (calvin_s0) and once with always skipping two frames (calvin_s2):

To download the final T3VIP models for CALVIN dataset:
```bash
cd $T3VIP_ROOT/checkpoints
sh download_model_weights.sh calvin_s0
sh download_model_weights.sh calvin_s2
```

## Omnipush
We trained T3VIP on Omnipush dataset once without any skipping frames (omnipush_s0) and once with always skipping two frames (omnipush_s2):

To download the final T3VIP models for Omnipush dataset:
```bash
cd $T3VIP_ROOT/checkpoints
sh download_model_weights.sh omnipush_s0
sh download_model_weights.sh omnipush_s2
```