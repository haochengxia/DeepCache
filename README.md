# Accelerating Stable Diffusion Inference via Selective Caching

## Installation

```
pip install -e .
```

## How to run the experiments

For individual generation:

```bash
bash run.sh # using basic deepcache
export ENABLE_SMART_INTERVAL=1; bash run.sh  # enable smart-internval opt
export ENABLE_MASK=1; bash run.sh # enable mask opt
```

Run benchmarking

```bash
pip install diffusers transformers open_clip_torch
python generate.py --dataset coco2017 --layer 0 --block 0 --update_interval 10 --uniform --steps 50 --batch_size 16 
export ENABLE_SMART_INTERVAL=1; python generate.py --dataset coco2017  --original --steps 50 --batch_size 16 # For original pipeline

python score_is.py PATH_TO_SAVED_IMAGES
```


## Ackownledgement

- Based on [DeepCache](https://github.com/horseee/DeepCache)