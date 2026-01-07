#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
convert_pred.py

Purpose:
- Convert GT/Pred JSON in various formats into a unified production_plan structure.

Supported inputs:
1) notes/table/daily_values   (e.g. 小鹏 10月 GT)
2) production_plan/rows       (label/values/sum)
3) production_plan/models     (already normalized)

Output format:
{
  "production_plan": {
    "meta": {"month": null, "year": null, "plant": null},
    "headers": [...],
    "models": [
      {"name": "...", "values": [...], "sum": <int or null>}
    ],
    "total": null
  }
}
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return None
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
        try:
            return int(float(s))
        except Exception:
            return None
    return None


def convert_notes_to_production_plan(data: Dict[str, Any]) -> Dict[str, Any]:
    notes = data.get("notes")
    if not isinstance(notes, list) or not notes:
        raise ValueError("notes is missing or empty")

    table = notes[0].get("table")
    if not isinstance(table, list) or not table:
        raise ValueError("notes[0].table is missing or empty")

    day_keys = set()
    for row in table:
        dv = row.get("daily_values", {})
        if isinstance(dv, dict):
            for k in dv:
                if isinstance(k, int):
                    day_keys.add(k)
                elif isinstance(k, str) and k.strip().isdigit():
                    day_keys.add(int(k.strip()))

    if not day_keys:
        raise ValueError("no day keys found in daily_values")

    headers = [str(i) for i in sorted(day_keys)]
    n = len(headers)

    models = []

    for row in table:
        name = (
            row.get("车型")
            or row.get("model")
            or row.get("name")
            or row.get("label")
        )
        if not isinstance(name, str):
            continue
        name = name.strip()

        dv = row.get("daily_values", {})
        values = []
        for h in headers:
            v = dv.get(h, dv.get(int(h)) if h.isdigit() else None)
            values.append(parse_int_or_none(v))

        row_sum = parse_int_or_none(row.get("合计", row.get("sum")))

        if name not in ("SUM", "合计", "Total", "TOTAL"):
            models.append({
                "name": name,
                "values": values,
                "sum": row_sum
            })

    for m in models:
        if len(m["values"]) != n:
            raise ValueError(f"values length mismatch for model {m['name']}")

    return {
        "production_plan": {
            "meta": {"month": None, "year": None, "plant": None},
            "headers": headers,
            "models": models,
            "total": None
        }
    }


def convert_rows_to_models(data: Dict[str, Any]) -> Dict[str, Any]:
    pp = data.get("production_plan", {})
    headers = pp.get("headers", [])
    rows = pp.get("rows", [])

    if not isinstance(headers, list) or not isinstance(rows, list):
        raise ValueError("invalid production_plan.rows format")

    models = []
    for r in rows:
        label = r.get("label")
        if not isinstance(label, str):
            continue
        if label in ("SUM", "合计", "Total", "TOTAL"):
            continue
        models.append({
            "name": label,
            "values": r.get("values", []),
            "sum": r.get("sum")
        })

    return {
        "production_plan": {
            "meta": {"month": None, "year": None, "plant": None},
            "headers": headers,
            "models": models,
            "total": None
        }
    }


def convert_any(data: Dict[str, Any]) -> Dict[str, Any]:
    if "notes" in data:
        return convert_notes_to_production_plan(data)

    if "production_plan" in data:
        pp = data["production_plan"]
        if isinstance(pp.get("rows"), list):
            return convert_rows_to_models(data)
        if isinstance(pp.get("models"), list):
            pp.setdefault("meta", {"month": None, "year": None, "plant": None})
            pp.setdefault("total", None)
            return data

    raise ValueError("unsupported input format")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outfile", default="")
    args = parser.parse_args()

    in_path = Path(args.infile)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_path = Path(args.outfile) if args.outfile else in_path

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted = convert_any(data)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
