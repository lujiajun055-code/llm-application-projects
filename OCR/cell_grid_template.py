#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cell_grid_template.py  (Level 1: Template-based grid, config-driven)

目标：
- 针对“版式固定/同类表格”的生产计划图片，使用模板配置（table bbox / rows / cols）生成单元格 bbox。
- 一次写好逻辑，后续只需要新增/修改模板配置，不改代码。

典型用法（推荐）：
1) 先为某一类表格手动标定 data 区域 bbox（像素坐标）
2) 配置行（rows）与列（cols）数量，按需要启用 sum 列
3) 输出每个 [row, col] 的 bbox，供后续单格 OCR 读取

坐标约定：
- bbox = [x1, y1, x2, y2]，像素坐标，左上为 (0,0)
- x2/y2 为右/下边界（PIL crop 使用 (x1,y1,x2,y2)）

注意：
- Level 1 的“通用”= 同类表格的版式稳定。遇到不同版式，只需新增一个 template，不需要改脚本。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import json


@dataclass(frozen=True)
class GridTemplate:
    """一个表格模板（Level 1）"""
    name: str
    # 仅包含“数据网格区域”的 bbox（通常：日期 1..31 + 可选 sum 列 + 行区）
    table_bbox: Tuple[int, int, int, int]
    # 行名列表（按 GT / 产出 JSON models 顺序）
    rows: List[str]
    # 日期列数（例如 31）
    cols: int
    # 是否包含最右侧 sum 列（不属于日期列，但也按网格输出一个 col 索引）
    has_sum_col: bool = True
    # 生成 bbox 时的内缩像素（减少线条干扰）
    inner_pad: int = 2

    def total_cols(self) -> int:
        return self.cols + (1 if self.has_sum_col else 0)

    def validate(self) -> None:
        x1, y1, x2, y2 = self.table_bbox
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError(f"Invalid table_bbox: {self.table_bbox}")
        if self.cols <= 0:
            raise ValueError("cols must be > 0")
        if not self.rows:
            raise ValueError("rows must be non-empty")
        if self.inner_pad < 0:
            raise ValueError("inner_pad must be >= 0")


def load_templates_from_json(path: str) -> Dict[str, GridTemplate]:
    """
    加载模板配置文件（JSON）。
    JSON 格式示例见 --init-config 输出。
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    templates: Dict[str, GridTemplate] = {}
    for name, t in cfg.get("templates", {}).items():
        templates[name] = GridTemplate(
            name=name,
            table_bbox=tuple(t["table_bbox"]),
            rows=list(t["rows"]),
            cols=int(t["cols"]),
            has_sum_col=bool(t.get("has_sum_col", True)),
            inner_pad=int(t.get("inner_pad", 2)),
        )
        templates[name].validate()
    if not templates:
        raise ValueError("No templates found in config.")
    return templates


def init_config_skeleton(path: str) -> None:
    """生成一个模板配置骨架文件（JSON）"""
    skeleton = {
        "templates": {
            "xpeng_monthly_v1": {
                "table_bbox": [100, 200, 1900, 1200],
                "rows": ["F30", "F30V", "F30R", "合计"],
                "cols": 31,
                "has_sum_col": True,
                "inner_pad": 2
            }
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(skeleton, f, ensure_ascii=False, indent=2)


def compute_edges(x1: int, x2: int, n: int) -> List[int]:
    """将 [x1,x2] 等分为 n 份，返回 n+1 条边界坐标（整数）"""
    w = x2 - x1
    return [x1 + int(round(i * w / n)) for i in range(n + 1)]


def build_cell_bboxes(template: GridTemplate) -> Dict[str, Any]:
    """
    返回结构：
    {
      "template": "...",
      "table_bbox": [x1,y1,x2,y2],
      "rows": [...],
      "cols": 31,
      "has_sum_col": true,
      "grid": {
         "<row_name>": {
            "1": [x1,y1,x2,y2],
            ...
            "31": [...],
            "sum": [...]   # 如果 has_sum_col
         },
         ...
      }
    }
    """
    template.validate()
    x1, y1, x2, y2 = template.table_bbox
    nrows = len(template.rows)
    ncols = template.total_cols()

    col_edges = compute_edges(x1, x2, ncols)
    row_edges = compute_edges(y1, y2, nrows)

    pad = template.inner_pad
    grid: Dict[str, Dict[str, List[int]]] = {}

    for ri, rname in enumerate(template.rows):
        ry1 = row_edges[ri]
        ry2 = row_edges[ri + 1]
        row_map: Dict[str, List[int]] = {}
        # 日期列 1..cols
        for ci in range(template.cols):
            cx1 = col_edges[ci]
            cx2 = col_edges[ci + 1]
            row_map[str(ci + 1)] = [
                cx1 + pad, ry1 + pad,
                max(cx1 + pad, cx2 - pad),
                max(ry1 + pad, ry2 - pad)
            ]
        # sum 列
        if template.has_sum_col:
            ci = template.cols
            cx1 = col_edges[ci]
            cx2 = col_edges[ci + 1]
            row_map["sum"] = [
                cx1 + pad, ry1 + pad,
                max(cx1 + pad, cx2 - pad),
                max(ry1 + pad, ry2 - pad)
            ]
        grid[rname] = row_map

    return {
        "template": template.name,
        "table_bbox": list(template.table_bbox),
        "rows": template.rows,
        "cols": template.cols,
        "has_sum_col": template.has_sum_col,
        "inner_pad": template.inner_pad,
        "grid": grid,
    }


def main():
    ap = argparse.ArgumentParser(description="Level-1 template-based cell grid bbox generator.")
    ap.add_argument("--config", type=str, default="grid_templates.json", help="模板配置 JSON 文件路径")
    ap.add_argument("--template", type=str, default=None, help="选择模板名（config 里 key）")
    ap.add_argument("--out", type=str, default="cell_grid.json", help="输出 bbox JSON 文件路径")
    ap.add_argument("--init-config", type=str, default=None, help="生成模板配置骨架到指定路径并退出")
    args = ap.parse_args()

    if args.init_config:
        init_config_skeleton(args.init_config)
        print(f"[OK] Wrote template config skeleton to: {args.init_config}")
        return

    templates = load_templates_from_json(args.config)
    name = args.template or next(iter(templates.keys()))
    if name not in templates:
        raise SystemExit(f"Template '{name}' not found. Available: {list(templates.keys())}")

    out_obj = build_cell_bboxes(templates[name])
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"[OK] Template: {name}")
    print(f"[OK] Output written to: {args.out}")


if __name__ == "__main__":
    main()
