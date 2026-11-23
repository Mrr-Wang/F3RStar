"""Regenerate comparison tables exclusively from saved official scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from .common import RUN, atomic_json
from .run import DEVELOPMENT, summarize

ORDER = ["baseline", "maxconf", "lfc8", "edge_anchor", "edge_anchor_direct", "mv_depth",
         "local_mvs", "visibility_soft", "upsample_only", "local_mvs_hr"]
LABELS = {"baseline": "基础中值", "maxconf": "最高置信度", "lfc8": "原8×8 LFC",
          "edge_anchor": "边缘校正CG（未收敛）", "edge_anchor_direct": "可靠锚点+边缘校正",
          "mv_depth": "多视图深度基优化", "local_mvs": "局部MVS", "visibility_soft": "遮挡感知软融合",
          "upsample_only": "仅二倍采样对照", "local_mvs_hr": "二倍分辨率MVS"}


def collect(run):
    data = {}
    for path in (run/"tasks").glob("*/*/result.json"):
        result = json.loads(path.read_text())
        if result["scores"] is not None:
            data[(result["scene"], result["variant"])] = result
    return data


def group(data, scenes, variant):
    if not all((s, variant) in data for s in scenes):
        return None
    rr = [data[(s, variant)] for s in scenes]
    return {"variant": variant, "n_scenes": len(scenes), "scenes": scenes,
            "F2": float(np.mean([r["scores"]["metrics"]["fscore"][1]*100 for r in rr])),
            "A2": float(np.mean([r["scores"]["metrics"]["accuracy"][1]*100 for r in rr])),
            "C2": float(np.mean([r["scores"]["metrics"]["completeness"][1]*100 for r in rr])),
            "F1": float(np.mean([r["scores"]["metrics"]["fscore"][0]*100 for r in rr])),
            "F5": float(np.mean([r["scores"]["metrics"]["fscore"][2]*100 for r in rr])),
            "F10": float(np.mean([r["scores"]["metrics"]["fscore"][3]*100 for r in rr])),
            "algorithm_seconds": float(np.mean([r.get("stats", {}).get("algorithm_seconds", 0) for r in rr])),
            "selection_eligible": variant != "edge_anchor"}


def table(rows, baseline):
    out = ["| 配置 | F@2cm | ΔF@2cm | A@2cm | C@2cm | 精化均时 |",
           "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        out.append(f"| {LABELS.get(row['variant'],row['variant'])} | {row['F2']:.4f}% | "
                   f"{row['F2']-baseline['F2']:+.4f}pp | {row['A2']:.4f}% | {row['C2']:.4f}% | "
                   f"{row['algorithm_seconds']:.2f}s |")
    return out


def paired(data, scenes, variant):
    if not all((s, v) in data for s in scenes for v in ("baseline", variant)):
        return None
    delta = np.array([(data[(s, variant)]["scores"]["metrics"]["fscore"][1]
                       - data[(s, "baseline")]["scores"]["metrics"]["fscore"][1])*100 for s in scenes])
    return {"variant": variant, "scenes": scenes, "deltas_pp": dict(zip(scenes, delta.tolist())),
            "mean_delta_pp": float(delta.mean()), "median_delta_pp": float(np.median(delta)),
            "improved_scenes": int((delta > 0).sum()), "degraded_scenes": int((delta < 0).sum()),
            "mean_without_largest_gain_pp": float(np.delete(delta, delta.argmax()).mean()) if len(delta)>1 else None,
            "leave_one_scene_out_mean_range_pp": [float((delta.sum()-delta).min()/(len(delta)-1)),
                                                    float((delta.sum()-delta).max()/(len(delta)-1))] if len(delta)>1 else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--status", choices=["running", "completed"], default="running")
    args = parser.parse_args()
    run = args.run_dir
    summarize(run)
    data = collect(run)
    all_scenes = [x["scene"] for x in json.loads((run/"evaluation-set.json").read_text())["all_13"]]
    development = [g for v in ORDER if (g := group(data, DEVELOPMENT, v)) is not None]
    extended = [g for v in ORDER if (g := group(data, all_scenes, v)) is not None]
    outside = [s for s in all_scenes if s not in DEVELOPMENT]
    paired_all = [p for v in ORDER if v != "baseline" and (p := paired(data, all_scenes, v)) is not None]
    paired_outside = [p for v in ORDER if v != "baseline" and (p := paired(data, outside, v)) is not None]
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    output = {"status": args.status, "updated_beijing": stamp,
              "development": development, "all_13": extended, "n_saved_official_scores": len(data),
              "paired_all_13": paired_all, "paired_outside_development_10": paired_outside}
    atomic_json(run/"comparison.json", output)
    lines = ["# DenseStar 逐项改进结果", "", f"状态：{args.status}；更新：{stamp}。", "",
             "使用 2026-10-25 InfoStar q4 固定预测/最终相机，新版官方评测器协议；"
             "不能与历史代理分数直接相减。输入和原始指标保留。", "",
             "GT只用于评测对齐与官方扫描评分。原始预测不做未知物理尺度下的粗体素化；"
             "评测侧统一5mm采样，官方归一化体素1cm。所有分数按场景等权平均。", "",
             "## 三个开发场景", "", "pipes、kicker、office。仅列三个场景全部完成的配置。", ""]
    if development:
        lines += table(development, next(g for g in development if g["variant"] == "baseline"))
    lines += ["", "精化时间只统计算法模块，不含原模型推理、缓存准备、点云导出或官方评测，"
              "不是完整重建时间。基础/最高置信度的额外精化时间记0，并不表示整个流程零成本。", "",
              "本轮为质量探索：固定基线与官方CPU评测可和其他算法重叠执行，GPU冷启动/内核编译未拆分，"
              "因此记录的时间不能作为受控串行速度对比或加速比。运行版本见environment-and-timing-scope.json。", "",
              "初版边缘CG未达到指定容差，保留数值但不参与配置选择；直接求解版本对相同目标核验了残差。", "",
              "## 逐场景 F@2cm", "", "| 场景 | " + " | ".join(LABELS[v] for v in ORDER if any(g["variant"] == v for g in development)) + " |"]
    variants = [v for v in ORDER if any(g["variant"] == v for g in development)]
    lines.append("|---|" + "---:|"*len(variants))
    for scene in DEVELOPMENT:
        lines.append("| " + scene + " | " + " | ".join(f"{data[(scene,v)]['scores']['metrics']['fscore'][1]*100:.4f}%" for v in variants) + " |")
    lines += ["", "## 全部13个公开扫描场景", "", "仅列13场景全部完成的配置；不依据结果排除场景。", ""]
    if extended:
        lines += table(extended, next(g for g in extended if g["variant"] == "baseline"))
        lines += ["", "这些是全部13个公开训练扫描场景，包含用于选配置的三个开发场景；不是独立官方测试集。",
                  "只有基础版、开发集最高分对照和开发集最高分新增精化扩展至13场景；其余配置只在三个开发场景测试。", "",
                  "### 不同距离阈值", "", "| 配置 | F@1cm | F@2cm | F@5cm | F@10cm |",
                  "|---|---:|---:|---:|---:|"]
        for row in extended:
            lines.append(f"| {LABELS[row['variant']]} | {row['F1']:.4f}% | {row['F2']:.4f}% | {row['F5']:.4f}% | {row['F10']:.4f}% |")
        lines += ["", "### 跨场景稳定性", "",
                  "后10场景是在三个开发场景选完配置后评测，没有再调参数，但仍属于公开训练集。", "",
                  "| 配置 | 全13均值变化 | 全13中位变化 | 改善场景 | 去掉最大收益场景 | 后10均值变化 | 后10改善场景 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for p in paired_all:
            q = next(x for x in paired_outside if x["variant"] == p["variant"])
            lines.append(f"| {LABELS[p['variant']]} | {p['mean_delta_pp']:+.4f}pp | {p['median_delta_pp']:+.4f}pp | "
                         f"{p['improved_scenes']}/13 | {p['mean_without_largest_gain_pp']:+.4f}pp | "
                         f"{q['mean_delta_pp']:+.4f}pp | {q['improved_scenes']}/10 |")
        completed_variants = [r["variant"] for r in extended]
        lines += ["", "### 全13逐场景 F@2cm", "",
                  "| 场景 | " + " | ".join(LABELS[v] for v in completed_variants) + " | 基线相机中心误差中位数 |",
                  "|---|"+"---:|"*(len(completed_variants)+1)]
        for scene in all_scenes:
            baseline = data[(scene, "baseline")]
            values = " | ".join(f"{data[(scene,v)]['scores']['metrics']['fscore'][1]*100:.4f}%" for v in completed_variants)
            lines.append(f"| {scene} | {values} | {baseline['scores']['alignment']['median_camera_error_m']*100:.2f}cm |")
        lines += ["", "相机误差为评测侧全局相似变换后的诊断值，不是算法输入，也不能单独证明深度或相机哪个因素主导。", ""]
    else:
        lines += ["尚在运行。"]
    lines += ["", "## 实验边界与执行记录", "",
              "- 高分辨率MVS另有仅上采样对照；其收益需区分采样密度与深度重估。",
              "- office的GT相机替换是隔离保存的敏感性诊断，不作为可部署方法或严格上界。",
              "- office多视图优化曾因部分图像假设数较少出现通道数错误，已保留初始错误记录并补齐空通道后复跑；没有降输入。",
              "- 直接求解替代未收敛CG及MVS亚像素拟合/采样对照均有单独修订记录。",
              "- 原始日志见各task目录，完整指标见results.csv；不是独立重复实验。", ""]
    conclusions = run/"conclusions.md"
    if conclusions.exists():
        lines += [conclusions.read_text()]
    (run/"comparison.md").write_text("\n".join(lines))
    print(f"Saved report: {len(data)} official task scores, {len(development)} complete development configurations")


if __name__ == "__main__":
    main()
