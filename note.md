### 参数说明

- `cache_interval`
- `cache_layer_id`: 指定更新的具体层。
- `cache_block_id`: 指定更新的具体块。

---


1.	interval_seq: 决定在哪些时间步中复用缓存特征。若 cache_interval > 1，特征在指定间隔中被重复使用；否则，每步都更新特征。

2.	prv_features: 储存上一次计算的高层特征，用于在当前时间步复用。若需要更新特征，则将其设置为 None。

---

#### U-Net

1. quick_replicate: 表示是否启用快速复用机制。启用后，会使用 replicate_prv_feature 中的缓存特征，而不重新计算。

2. replicate_prv_feature: 表示存储的上一个时间步的中间特征，用于当前时间步的复用。

3. cache_layer_id 和 cache_block_id: 指定需要复用缓存的层和块。

```python
        quick_replicate: bool = False,
        replicate_prv_feature: Optional[List[torch.Tensor]] = None,
        cache_layer_id: Optional[int] = None,
        cache_block_id: Optional[int] = None,
```

### Res

```
PYTHONPATH=. python experiments/generate.py --dataset coco2017 --layer 0 --block 0 --update_interval 10 --steps 50
```

Make sure all parameters are the same:
50-block-0-layer-0-interval-10-uniform-False-pow-1.4-center-15

```
python score_is.py XX.pt
```

```
Ours:      Inception Score: 6.471504894826317 ± 0.7687108696958255
Deepcache: Inception Score: 6.2338975133210965 ± 0.7756914243550831
```