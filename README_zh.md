# F3RStar

[English](README.md) | **中文**

**基于自适应星图的大规模场景前馈式三维重建方法**

待论文接收后公布更多实现细节代码。

F3RStar 基于 [GLUEMAP](https://github.com/colmap/gluemap)，主要增加：

1. **自适应星图选择** — 饱和覆盖增益选星，并补齐连通性，减少冗余的局部前馈推理。
2. **稀疏几何引导的稠密重建** — 用 BA 标定各星深度后融合，再经边缘引导的锚点校正。

本仓库不包含数据集、模型权重，需额外自行下载。

## 目录结构

```
gluemap/                 # 修改后的 GLUEMAP 管线 + F3RStar 模块
  estimators/greedy_star.py          # 覆盖增益选星
  estimators/cross_star_consistency.py
  dense/                             # DenseStar：缓存 / 融合 / 边缘校正
configs/
  base.yaml                          # F3RStar 管线默认参数
  f3rstar/default.yaml               # 精简方法配置
  example.yaml                       # 演示用 IO 模板
dense_tools/             # 从一次运行结果导出稠密点云
thirdparty/              # git submodule（π³、SALAD、Doppelgangers++ 等）
```

## 安装

C++/CUDA 依赖见 [INSTALL.md](INSTALL.md)。简要步骤：

```bash
git clone --recursive https://github.com/Mrr-Wang/F3RStar.git
cd F3RStar
# 若克隆时未加 --recursive：
# git submodule update --init --recursive

# 创建环境并安装 Ceres / Eigen / Boost 后：
pip install -e .
```

默认前馈骨干为 **π³**。请将权重放到 `checkpoints/`：

| 文件 | 来源 |
|------|------|
| `checkpoints/pi3.safetensors` | [π³](https://github.com/yyfz/Pi3) |
| `checkpoints/dino_salad.ckpt` | [SALAD](https://github.com/serizba/salad) |
| `checkpoints/vggsfm_v2_0_0_track_predictor.bin` | VGGSfM tracker（见 GLUEMAP INSTALL） |
| `checkpoints/checkpoint-dg+visym.pth` | Doppelgangers++ |

可选骨干（`vggt`、`map_anything`）由 submodule 控制，默认 F3RStar 配置不需要它们。

## 快速开始

```bash
# 1）编辑 configs/f3rstar.yaml 中的 images_path / write_path
# 2）运行

python demo.py --config configs/f3rstar.yaml --save_dense_depth
```

若启用了 `--save_dense_depth`，可从重建目录导出稠密点云（需包含 `gluemap_aba/`、`dense_cache/`、`global_scales.pth`）：

```bash
PYTHONPATH=. python -m dense_tools.run run \
  --source results/scene \
  --run results/dense \
  --scenes scene \
  --no-score
```

`--no-score` 会跳过可选的 ETH3D 官方评测器（本仓库不附带该评测程序）。

## 方法默认设置

具体实现参数待论文接收后再公布。

## 许可与致谢

- F3RStar 新增部分：MIT（见 [LICENSE](LICENSE)）。
- 管线代码源自 [GLUEMAP](https://github.com/colmap/gluemap)（BSD-3，ETH Zurich），见 [NOTICE](NOTICE)。
- `thirdparty/` 下各模型保留各自上游许可。

若使用本代码，请同时引用 GLUEMAP 以及所用的前馈骨干网络。

## 引用

```bibtex
@article{wang2026f3rstar,
  title={Feed-Forward 3D Reconstruction of Large-Scale Scenes Based on Adaptive Star Graphs},
  author={Wang, Xun},
  year={2026}
}
```
