# F3RStar

**English** | [中文](README_zh.md)

**Feed-Forward 3D Reconstruction of Large-Scale Scenes Based on Adaptive Star Graphs**

More implementation details will be released after the paper is accepted.

F3RStar builds on [GLUEMAP](https://github.com/colmap/gluemap) and adds:

1. **Adaptive star-graph selection** — saturated coverage-gain selection with connectivity repair, reducing redundant local feed-forward inference.
2. **Sparse-geometry-guided dense reconstruction** — BA-calibrated per-star depth fusion with edge-guided anchor correction.

This repository does not include datasets or model weights; please download them separately.

## Repository layout

```
gluemap/                 # Modified GLUEMAP pipeline + F3RStar modules
  estimators/greedy_star.py          # coverage-gain star selection
  estimators/cross_star_consistency.py
  dense/                             # DenseStar: cache / fuse / edge_correct
configs/
  base.yaml                          # default F3RStar pipeline settings
  f3rstar/default.yaml               # compact method config
  example.yaml                       # demo IO template
dense_tools/             # export refined dense point clouds from a run
thirdparty/              # git submodules (π³, SALAD, Doppelgangers++, …)
```

## Installation

See [INSTALL.md](INSTALL.md) for C++/CUDA build dependencies. Short version:

```bash
git clone --recursive https://github.com/Mrr-Wang/F3RStar.git
cd F3RStar
# if you already cloned without --recursive:
# git submodule update --init --recursive

# create env, install Ceres/Eigen/Boost, then:
pip install -e .
```

Default feed-forward backbone is **π³**. Place checkpoints under `checkpoints/`:

| File | Source |
|------|--------|
| `checkpoints/pi3.safetensors` | [π³](https://github.com/yyfz/Pi3) |
| `checkpoints/dino_salad.ckpt` | [SALAD](https://github.com/serizba/salad) |
| `checkpoints/vggsfm_v2_0_0_track_predictor.bin` | VGGSfM tracker (see GLUEMAP INSTALL) |
| `checkpoints/checkpoint-dg+visym.pth` | Doppelgangers++ |

Optional backbones (`vggt`, `map_anything`) are submodule-gated and not required for the default F3RStar config.

## Quick start

```bash
# 1) Edit images_path / write_path in configs/f3rstar.yaml
# 2) Run

python demo.py --config configs/f3rstar.yaml --save_dense_depth
```

After a run with `--save_dense_depth`, export the dense point cloud from the
reconstruction directory (expects `gluemap_aba/`, `dense_cache/`, `global_scales.pth`):

```bash
PYTHONPATH=. python -m dense_tools.run run \
  --source results/scene \
  --run results/dense \
  --scenes scene \
  --no-score
```

`--no-score` skips the optional ETH3D official evaluator (not shipped here).

## Method defaults

Concrete implementation parameters will be released after the paper is accepted.

## License & acknowledgements

- F3RStar additions: MIT (see [LICENSE](LICENSE)).
- Pipeline code is derived from [GLUEMAP](https://github.com/colmap/gluemap) (BSD-3, ETH Zurich). See [NOTICE](NOTICE).
- Third-party models keep their own licenses under `thirdparty/`.

If you use this code, please also cite GLUEMAP and the feed-forward backbones you rely on.

## Citation

```bibtex
@article{wang2026f3rstar,
  title={Feed-Forward 3D Reconstruction of Large-Scale Scenes Based on Adaptive Star Graphs},
  author={Wang, Xun},
  year={2026}
}
```
