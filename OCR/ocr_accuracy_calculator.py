#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像OCR准确率统计程序
比较预测文件中的raw_content_preview与标签文件，计算准确率
"""

import json
import sys
import argparse
from typing import Any, Dict, List, Tuple
from pathlib import Path
import re

def normalize_value(value: Any) -> Any:
    """标准化值，用于比较"""
    if isinstance(value, str):
        # 去除首尾空格
        value = value.strip()
        # 空字符串统一处理
        if value == "":
            return None
    elif isinstance(value, (int, float)):
        # 数字类型保持不变
        pass
    elif value is None:
        return None
    return value

def is_all_null_row(values):
    return isinstance(values, list) and len(values) > 0 and all(v is None for v in values)

def should_skip_model_by_gt(gt_model: dict) -> bool:
    """GT 中 values 全为 null 且 sum 为 null => 结构性空行，不参与评估"""
    if not isinstance(gt_model, dict):
        return False
    vals = gt_model.get("values", [])
    s = gt_model.get("sum", None)
    if not isinstance(vals, list) or len(vals) == 0:
        return False
    return is_all_null_row(vals) and (s is None)


def clean_for_comparison(text: Any) -> Any:
    """去除字符串中的所有空格、中文括号和英文括号，并转为小写。非字符串类型不处理。"""
    if not isinstance(text, str):
        return text

        # --- 增强标准化 (新增) ---
    # 替换中文标点为英文标点
    text = text.replace('，', ',')
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace('：', ':')
    # -------------------------

    # 1. 移除所有空格 (包括普通空格、制表符、换行符等)
    text = re.sub(r'\s+', '', text)

    # 2. 移除中文和英文括号
    text = text.replace('（', '').replace('）', '')
    text = text.replace('(', '').replace(')', '')

    # 3. 统一转换为小写
    return text.lower()


def compare_values(pred: Any, label: Any) -> Tuple[bool, str]:
    """
    比较两个值是否匹配
    返回: (是否匹配, 说明)
    注意：实际内容一致即可判别为准确，比如10和"10"就是准确的
    """
    pred_norm = normalize_value(pred)
    label_norm = normalize_value(label)
    
    # 都为空，认为匹配
    if pred_norm is None and label_norm is None:
        return True, "both_empty"
    
    # 一个为空一个不为空
    if pred_norm is None or label_norm is None:
        return False, f"empty_mismatch: pred={pred_norm}, label={label_norm}"


    # 直接比较
    if pred_norm == label_norm:
        return True, "exact_match"


    # 尝试转换为相同类型比较（增强版）
    try:
        # 情况1: 两个都是字符串，但内容相同（去除空格后）
        if isinstance(pred_norm, str) and isinstance(label_norm, str):

            pred_cleaned = clean_for_comparison(pred_norm)
            label_cleaned = clean_for_comparison(label_norm)

            if pred_cleaned == label_cleaned:
                return True, "cleaned_string_match"

        # 情况2: 字符串 vs 数字
        elif isinstance(pred_norm, str) and isinstance(label_norm, (int, float)):
            # 去除字符串首尾空格后比较
            pred_str = pred_norm.strip()
            label_str = str(label_norm)
            if pred_str == label_str:
                return True, "type_converted_match"
            # 尝试将字符串转换为数字后比较
            try:
                pred_num = float(pred_str) if '.' in pred_str else int(pred_str)
                if abs(pred_num - label_norm) < 1e-10:  # 浮点数比较
                    return True, "type_converted_match"
            except ValueError:
                pass
        elif isinstance(label_norm, str) and isinstance(pred_norm, (int, float)):
            # 去除字符串首尾空格后比较
            label_str = label_norm.strip()
            pred_str = str(pred_norm)

            if label_str == pred_str:
                return True, "type_converted_match"
            # 尝试将字符串转换为数字后比较
            try:
                label_num = float(label_str) if '.' in label_str else int(label_str)
                if abs(label_num - pred_norm) < 1e-10:  # 浮点数比较
                    return True, "type_converted_match"
            except ValueError:
                pass
        # 情况3: 两个都是数字，但类型不同（int vs float）
        elif isinstance(pred_norm, (int, float)) and isinstance(label_norm, (int, float)):
            if abs(pred_norm - label_norm) < 1e-10:  # 浮点数比较
                return True, "type_converted_match"
    except Exception:
        pass
    
    return False, f"mismatch: pred={pred_norm}, label={label_norm}"


def count_and_mark_missing(pred: Dict, label: Dict, path: str = "") -> Dict[str, Any]:
    """
    当预测数据缺失时，统计标签数据中的所有字段
    """
    stats = {
        "total_fields": 0,
        "details": []
    }
    
    for key, label_val in label.items():
        current_path = f"{path}.{key}" if path else key
        
        if isinstance(label_val, dict):
            nested_stats = count_and_mark_missing({}, label_val, current_path)
            stats["total_fields"] += nested_stats["total_fields"]
            stats["details"].extend(nested_stats["details"])
        elif isinstance(label_val, list):
            nested_stats = count_and_mark_missing_list([], label_val, current_path)
            stats["total_fields"] += nested_stats["total_fields"]
            stats["details"].extend(nested_stats["details"])
        else:
            stats["total_fields"] += 1
            stats["details"].append({
                "path": current_path,
                "status": "missing_in_pred",
                "label_value": label_val
            })
    
    return stats


def count_and_mark_missing_list(pred: List, label: List, path: str = "") -> Dict[str, Any]:
    """
    当预测数据缺失时，统计标签数据列表中的所有字段
    """
    stats = {
        "total_fields": 0,
        "details": []
    }
    
    for i, label_item in enumerate(label):
        current_path = f"{path}[{i}]"
        
        if isinstance(label_item, dict):
            nested_stats = count_and_mark_missing({}, label_item, current_path)
            stats["total_fields"] += nested_stats["total_fields"]
            stats["details"].extend(nested_stats["details"])
        elif isinstance(label_item, list):
            nested_stats = count_and_mark_missing_list([], label_item, current_path)
            stats["total_fields"] += nested_stats["total_fields"]
            stats["details"].extend(nested_stats["details"])
        else:
            stats["total_fields"] += 1
            stats["details"].append({
                "path": current_path,
                "status": "missing_in_pred",
                "label_value": label_item
            })
    
    return stats


def count_extra_fields(pred: Dict, label: Dict, path: str = "") -> Dict[str, Any]:
    """
    当标签数据缺失时，统计预测数据中的所有字段（多余字段）
    """
    stats = {
        "total_fields": 0,
        "details": []
    }
    
    for key, pred_val in pred.items():
        current_path = f"{path}.{key}" if path else key
        
        if isinstance(pred_val, dict):
            nested_stats = count_extra_fields(pred_val, {}, current_path)
            stats["total_fields"] += nested_stats["total_fields"]
            stats["details"].extend(nested_stats["details"])
        elif isinstance(pred_val, list):
            nested_stats = count_extra_fields_list(pred_val, [], current_path)
            stats["total_fields"] += nested_stats["total_fields"]
            stats["details"].extend(nested_stats["details"])
        else:
            stats["total_fields"] += 1
            stats["details"].append({
                "path": current_path,
                "status": "extra_in_pred",
                "pred_value": pred_val
            })
    
    return stats


def count_extra_fields_list(pred: List, label: List, path: str = "") -> Dict[str, Any]:
    """
    当标签数据缺失时，统计预测数据列表中的所有字段（多余字段）
    """
    stats = {
        "total_fields": 0,
        "details": []
    }
    
    for i, pred_item in enumerate(pred):
        current_path = f"{path}[{i}]"
        
        if isinstance(pred_item, dict):
            nested_stats = count_extra_fields(pred_item, {}, current_path)
            stats["total_fields"] += nested_stats["total_fields"]
            stats["details"].extend(nested_stats["details"])
        elif isinstance(pred_item, list):
            nested_stats = count_extra_fields_list(pred_item, [], current_path)
            stats["total_fields"] += nested_stats["total_fields"]
            stats["details"].extend(nested_stats["details"])
        else:
            stats["total_fields"] += 1
            stats["details"].append({
                "path": current_path,
                "status": "extra_in_pred",
                "pred_value": pred_item
            })
    
    return stats


def compare_dict(pred: Dict, label: Dict, path: str = "") -> Dict[str, Any]:
    """
    递归比较两个字典
    返回统计信息
    """
    stats = {
        "total_fields": 0,
        "matched_fields": 0,
        "mismatched_fields": 0,
        "missing_in_pred": 0,
        "extra_in_pred": 0,
        "details": []
    }
    
    # 获取所有键
    all_keys = set(pred.keys()) | set(label.keys())
    
    for key in all_keys:
        current_path = f"{path}.{key}" if path else key
                # ===== A 路线：只评估数值字段 =====
        # 直接忽略这些结构字段（不进分母、不记 extra/missing）
        if current_path.endswith(".meta") or current_path.endswith(".headers") or current_path.endswith(".total") or current_path.endswith(".name"):
            continue

        # 只有这三类“数值叶子字段”才计入评估
            # ===== A 路线：只评估 values（推荐）+ 可选 sum；不评估 daily（避免重复扣分）=====
        is_value_leaf = (".values[" in current_path) or current_path.endswith(".sum")


        if key not in pred:
            # 预测数据中缺失该字段，需要递归统计标签数据中的所有字段
            label_val = label[key]
            if isinstance(label_val, dict):
                nested_stats = count_and_mark_missing({}, label_val, current_path)
                stats["total_fields"] += nested_stats["total_fields"]
                stats["missing_in_pred"] += nested_stats["total_fields"]
                stats["details"].extend(nested_stats["details"])
            elif isinstance(label_val, list):
                nested_stats = count_and_mark_missing_list([], label_val, current_path)
                stats["total_fields"] += nested_stats["total_fields"]
                stats["missing_in_pred"] += nested_stats["total_fields"]
                stats["details"].extend(nested_stats["details"])
            else:
                stats["total_fields"] += 1
                stats["missing_in_pred"] += 1
                stats["details"].append({
                    "path": current_path,
                    "status": "missing_in_pred",
                    "label_value": label_val
                })
        elif key not in label:
            # 标签数据中缺失该字段（预测数据中多余）
            pred_val = pred[key]
            if isinstance(pred_val, dict):
                nested_stats = count_extra_fields(pred_val, {}, current_path)
                stats["total_fields"] += nested_stats["total_fields"]
                stats["extra_in_pred"] += nested_stats["total_fields"]
                stats["details"].extend(nested_stats["details"])
            elif isinstance(pred_val, list):
                nested_stats = count_extra_fields_list(pred_val, [], current_path)
                stats["total_fields"] += nested_stats["total_fields"]
                stats["extra_in_pred"] += nested_stats["total_fields"]
                stats["details"].extend(nested_stats["details"])
            else:
                stats["total_fields"] += 1
                stats["extra_in_pred"] += 1
                stats["details"].append({
                    "path": current_path,
                    "status": "extra_in_pred",
                    "pred_value": pred_val
                })
        else:
            pred_val = pred[key]
            label_val = label[key]
            
            # 递归处理嵌套结构
            if isinstance(pred_val, dict) and isinstance(label_val, dict):
                nested_stats = compare_dict(pred_val, label_val, current_path)
                stats["total_fields"] += nested_stats["total_fields"]
                stats["matched_fields"] += nested_stats["matched_fields"]
                stats["mismatched_fields"] += nested_stats["mismatched_fields"]
                stats["missing_in_pred"] += nested_stats["missing_in_pred"]
                stats["extra_in_pred"] += nested_stats["extra_in_pred"]
                stats["details"].extend(nested_stats["details"])
            elif isinstance(pred_val, list) and isinstance(label_val, list):
                nested_stats = compare_list(pred_val, label_val, current_path)
                stats["total_fields"] += nested_stats["total_fields"]
                stats["matched_fields"] += nested_stats["matched_fields"]
                stats["mismatched_fields"] += nested_stats["mismatched_fields"]
                stats["missing_in_pred"] += nested_stats["missing_in_pred"]
                stats["extra_in_pred"] += nested_stats["extra_in_pred"]
                stats["details"].extend(nested_stats["details"])
            else:
                # ===== A 路线：只在数值叶子字段上计数 =====
                if not is_value_leaf:
                    continue

                # Step2：GT 为 null 的字段不参与评估
                if label_val is None:
                    continue

                stats["total_fields"] += 1
                is_match, reason = compare_values(pred_val, label_val)
                if is_match:
                    stats["matched_fields"] += 1
                else:
                    stats["mismatched_fields"] += 1
                    stats["details"].append({
                        "path": current_path,
                        "status": "mismatch",
                        "pred_value": pred_val,
                        "label_value": label_val,
                        "reason": reason
                    })

    return stats


def compare_list(pred: List, label: List, path: str = "") -> Dict[str, Any]:
    """
    比较两个列表
    返回统计信息
    """
    stats = {
        "total_fields": 0,
        "matched_fields": 0,
        "mismatched_fields": 0,
        "missing_in_pred": 0,
        "extra_in_pred": 0,
        "details": []
    }
    # ===== A 路线：headers 不参与评估 =====
    if path.endswith(".headers"):
        return stats

    max_len = max(len(pred), len(label))
    
    for i in range(max_len):
        current_path = f"{path}[{i}]"
        
        if i >= len(pred):
            # 预测数据中缺失该元素，需要递归统计标签数据中的所有字段
            label_item = label[i]
            if isinstance(label_item, dict):
                nested_stats = count_and_mark_missing({}, label_item, current_path)
                stats["total_fields"] += nested_stats["total_fields"]
                stats["missing_in_pred"] += nested_stats["total_fields"]
                stats["details"].extend(nested_stats["details"])
            elif isinstance(label_item, list):
                nested_stats = count_and_mark_missing_list([], label_item, current_path)
                stats["total_fields"] += nested_stats["total_fields"]
                stats["missing_in_pred"] += nested_stats["total_fields"]
                stats["details"].extend(nested_stats["details"])
            else:
                stats["total_fields"] += 1
                stats["missing_in_pred"] += 1
                stats["details"].append({
                    "path": current_path,
                    "status": "missing_in_pred",
                    "label_value": label_item
                })
        elif i >= len(label):
            # 标签数据中缺失该元素（预测数据中多余）
            pred_item = pred[i]
            if isinstance(pred_item, dict):
                nested_stats = count_extra_fields(pred_item, {}, current_path)
                stats["total_fields"] += nested_stats["total_fields"]
                stats["extra_in_pred"] += nested_stats["total_fields"]
                stats["details"].extend(nested_stats["details"])
            elif isinstance(pred_item, list):
                nested_stats = count_extra_fields_list(pred_item, [], current_path)
                stats["total_fields"] += nested_stats["total_fields"]
                stats["extra_in_pred"] += nested_stats["total_fields"]
                stats["details"].extend(nested_stats["details"])
            else:
                stats["total_fields"] += 1
                stats["extra_in_pred"] += 1
                stats["details"].append({
                    "path": current_path,
                    "status": "extra_in_pred",
                    "pred_value": pred_item
                })
        else:
            pred_item = pred[i]
            label_item = label[i]
            # ===== A 路线：list 元素级别，GT 为 null 不参与评估 =====
            # 主要用于 values[i] 这类路径
            if label_item is None:
                continue

            if isinstance(label_item, dict) and "values" in label_item and "sum" in label_item:
                if should_skip_model_by_gt(label_item):
                    continue           
            # 递归处理嵌套结构
            if isinstance(pred_item, dict) and isinstance(label_item, dict):
                nested_stats = compare_dict(pred_item, label_item, current_path)
                stats["total_fields"] += nested_stats["total_fields"]
                stats["matched_fields"] += nested_stats["matched_fields"]
                stats["mismatched_fields"] += nested_stats["mismatched_fields"]
                stats["missing_in_pred"] += nested_stats["missing_in_pred"]
                stats["extra_in_pred"] += nested_stats["extra_in_pred"]
                stats["details"].extend(nested_stats["details"])
            elif isinstance(pred_item, list) and isinstance(label_item, list):
                nested_stats = compare_list(pred_item, label_item, current_path)
                stats["total_fields"] += nested_stats["total_fields"]
                stats["matched_fields"] += nested_stats["matched_fields"]
                stats["mismatched_fields"] += nested_stats["mismatched_fields"]
                stats["missing_in_pred"] += nested_stats["missing_in_pred"]
                stats["extra_in_pred"] += nested_stats["extra_in_pred"]
                stats["details"].extend(nested_stats["details"])
            else:
                stats["total_fields"] += 1
                is_match, reason = compare_values(pred_item, label_item)
                if is_match:
                    stats["matched_fields"] += 1
                else:
                    stats["mismatched_fields"] += 1
                    stats["details"].append({
                        "path": current_path,
                        "status": "mismatch",
                        "pred_value": pred_item,
                        "label_value": label_item,
                        "reason": reason
                    })
    
    return stats


def calculate_accuracy(pred_data: Any, label_data: Any) -> Dict[str, Any]:
    """
    计算准确率
    """
    if isinstance(pred_data, dict) and isinstance(label_data, dict):
        stats = compare_dict(pred_data, label_data)
    elif isinstance(pred_data, list) and isinstance(label_data, list):
        stats = compare_list(pred_data, label_data)
    else:
        # 简单值比较
        is_match, reason = compare_values(pred_data, label_data)
        stats = {
            "total_fields": 1,
            "matched_fields": 1 if is_match else 0,
            "mismatched_fields": 0 if is_match else 1,
            "missing_in_pred": 0,
            "extra_in_pred": 0,
            "details": [] if is_match else [{
                "path": "root",
                "status": "mismatch",
                "pred_value": pred_data,
                "label_value": label_data,
                "reason": reason
            }]
        }
    
    # 计算准确率
    # 准确率 = 匹配字段数 / (匹配字段数 + 不匹配字段数 + 缺失字段数)
    # 这样计算可以反映预测数据相对于标签数据的完整性和准确性
    total_comparable = stats["matched_fields"] + stats["mismatched_fields"] + stats["missing_in_pred"]
    if total_comparable > 0:
        accuracy = (stats["matched_fields"] / total_comparable) * 100
    else:
        accuracy = 0.0
    
    stats["accuracy"] = accuracy
    stats["accuracy_percentage"] = f"{accuracy:.2f}%"
    stats["total_comparable_fields"] = total_comparable  # 可比较的字段总数（不包括多余字段）
    
    return stats

def is_all_null_row(values):
    return isinstance(values, list) and len(values) > 0 and all(v is None for v in values)

def should_skip_model_by_gt(gt_model: dict) -> bool:
    """GT 中 values 全为 null 且 sum 为 null => 结构性空行，不参与评估"""
    if not isinstance(gt_model, dict):
        return True
    vals = gt_model.get("values", [])
    s = gt_model.get("sum", None)
    # values 为空也视为不可评估
    if not isinstance(vals, list) or len(vals) == 0:
        return True
    return is_all_null_row(vals) and (s is None)

def load_prediction_file(pred_file_path: str) -> Dict[str, Any]:
    """
    从预测文件中加载JSON数据
    支持多种格式：
    A) 直接的JSON文件（新格式）：
       - 顶层包含 production_plan / table_data / notes 等
    B) wrapper格式：
       - {"parse_success": true, "parsed_json": {...}}
    C) 旧格式：
       - {"results": {...}, 其中包含 raw_content_preview 字符串}
    """
    with open(pred_file_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    # ---------------------------
    # A) 直接 JSON（新格式）
    # ---------------------------
    # 你的 pred 现在是 {"production_plan": {...}}
    if isinstance(pred_data, dict) and (
        "production_plan" in pred_data
        or "table_data" in pred_data
        or "notes" in pred_data
    ):
        print("检测到直接JSON格式，直接使用")
        print(f"解析成功: 类型={type(pred_data)}, 键={list(pred_data.keys())}")
        return pred_data

    # ---------------------------
    # B) wrapper 格式：parse_success / parsed_json
    # ---------------------------
    if isinstance(pred_data, dict) and pred_data.get("parse_success") is True and isinstance(pred_data.get("parsed_json"), dict):
        pj = pred_data["parsed_json"]
        # pj 也可能是 production_plan 顶层
        if isinstance(pj, dict):
            print("检测到 wrapper(JSON) 格式（parse_success + parsed_json），使用 parsed_json")
            print(f"解析成功: 类型={type(pj)}, 键={list(pj.keys())}")
            return pj

    # ---------------------------
    # C) 旧格式：results -> raw_content_preview
    # ---------------------------
    results = pred_data.get("results", {})
    if not results:
        raise ValueError("预测文件中没有results字段或results为空，且不是直接的JSON格式")

    # 获取第一个结果
    if isinstance(results, dict):
        first_result_key = list(results.keys())[0]
        first_result = results[first_result_key]
    elif isinstance(results, list) and results:
        # 有些实现会把 results 做成 list
        first_result = results[0]
    else:
        raise ValueError("预测文件中的results格式不正确或为空")

    raw_content = ""
    if isinstance(first_result, dict):
        raw_content = first_result.get("raw_content_preview", "") or first_result.get("raw_stream_content", "")
    if not raw_content:
        raise ValueError("预测文件中没有raw_content_preview字段（或raw_stream_content为空）")

    # 检查并去除Markdown代码块标记
    if raw_content.startswith("```json\n"):
        print("检测到Markdown代码块格式（```json），正在去除标记...")
        raw_content = raw_content[8:]
    elif raw_content.startswith("```\n"):
        print("检测到Markdown代码块格式（```），正在去除标记...")
        raw_content = raw_content[4:]
    elif raw_content.startswith("```json"):
        print("检测到Markdown代码块格式（```json），正在去除标记...")
        raw_content = raw_content[7:]
    elif raw_content.startswith("```"):
        print("检测到Markdown代码块格式（```），正在去除标记...")
        raw_content = raw_content[3:]

    if raw_content.endswith("\n```"):
        raw_content = raw_content[:-4]
    elif raw_content.endswith("```"):
        raw_content = raw_content[:-3]

    raw_content = raw_content.strip()

    # 检查是否被截断/不完整
    is_truncated = raw_content.endswith("...")
    open_braces = raw_content.count('{') - raw_content.count('}')
    open_brackets = raw_content.count('[') - raw_content.count(']')
    is_incomplete = open_braces > 0 or open_brackets > 0

    if is_truncated or is_incomplete:
        print(f"警告: raw_content 内容被截断或不完整（缺少{open_braces}个大括号和{open_brackets}个方括号），将尝试解析可用部分")
        if is_truncated:
            raw_content = raw_content.rstrip("...").rstrip()

    # 解析 JSON 字符串
    try:
        pred_json = json.loads(raw_content)
        if is_truncated:
            print("注意: 由于内容被截断，准确率计算可能不完整")
        if isinstance(pred_json, dict) and len(pred_json) == 0:
            print("警告: 解析出的JSON是空字典")
        elif isinstance(pred_json, list) and len(pred_json) == 0:
            print("警告: 解析出的JSON是空列表")
        print(f"解析成功: 类型={type(pred_json)}, 键={list(pred_json.keys()) if isinstance(pred_json, dict) else 'N/A'}")
        return pred_json

    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        print(f"错误位置: {getattr(e, 'pos', 'N/A')}")
        print("尝试修复JSON...")

        # 尝试修复：从后往前截断 + 补全括号
        best_result = None
        best_length = 0

        for i in range(len(raw_content), max(0, len(raw_content) - 2000), -10):
            try:
                test_content = raw_content[:i]
                ob = test_content.count('{') - test_content.count('}')
                ok = test_content.count('[') - test_content.count(']')

                if ob > 0:
                    test_content += '\n' + ('}' * ob)
                if ok > 0:
                    test_content += '\n' + (']' * ok)

                tmp = json.loads(test_content)
                if i > best_length:
                    best_result = tmp
                    best_length = i
                    print(f"成功解析部分JSON（前{i}个字符，补全了{ob}个大括号和{ok}个方括号）")
            except Exception:
                continue

        if best_result is not None:
            return best_result

        # 最简单修复：只补全最外层
        try:
            test_content = raw_content
            ob = test_content.count('{') - test_content.count('}')
            ok = test_content.count('[') - test_content.count(']')
            if ob > 0:
                test_content += '\n' + ('}' * ob)
            if ok > 0:
                test_content += '\n' + (']' * ok)
            pred_json = json.loads(test_content)
            print(f"使用简单修复成功解析JSON（补全了{ob}个大括号和{ok}个方括号）")
            return pred_json
        except Exception:
            raise ValueError(f"无法解析raw_content_preview中的JSON（即使尝试修复后）: {e}")


def load_label_file(label_file_path: str) -> Dict[str, Any]:
    """
    加载标签文件
    """
    with open(label_file_path, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    return label_data

def _to_int_or_none(x):
    x = normalize_value(x)
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None
        # 纯数字
        if re.fullmatch(r"-?\d+", s):
            return int(s)
    return x  # 保留字符串备注（例如节日）

def convert_notes_table_to_production_plan(pred_data: dict, headers=None) -> dict:
    """
    把 {"notes": {"table":[{"label":..., "1":..., ...,"SUM":...}, ...]}}
    转为 {"production_plan": {"headers":[...], "models":[...], "total":{...}}}
    """
    table = pred_data.get("notes", {}).get("table", [])
    if not isinstance(table, list) or not table:
        return pred_data

    # ✅ 优先使用 GT.headers
    if headers and isinstance(headers, list) and len(headers) > 0:
        headers = [str(h) for h in headers]
    else:
        # fallback：自己推断
        day_keys = [str(i) for i in range(1, 32)]
        headers = [k for k in day_keys if any((isinstance(r, dict) and k in r) for r in table)]
        if not headers:
            first = table[0] if isinstance(table[0], dict) else {}
            headers = [k for k in first.keys() if k.isdigit()]
            headers = sorted(headers, key=lambda x: int(x))
            print("[DEBUG] headers_from_gt =", bool(headers), "len=", (len(headers) if headers else None))


    models = []
    total_obj = None

    for row in table:
        if not isinstance(row, dict):
            continue
        name = row.get("label") or row.get("name")
        if not name:
            continue

        daily = {h: _to_int_or_none(row.get(h)) for h in headers}
        values = [_to_int_or_none(row.get(h)) for h in headers]
        s = _to_int_or_none(row.get("SUM") or row.get("sum"))

        item = {
            "name": name,
            "values": values,
            "daily": daily,
            "sum": s
        }

        # 约定：TJ 行是 total（你提示词里也是这么要求的）
        if str(name).strip().upper() == "TJ":
            total_obj = item
        else:
            models.append(item)

    production_plan = {
        "headers": headers,
        "models": models,
        "total": total_obj if total_obj is not None else {"name": "TJ", "values": [], "daily": {}, "sum": None}
    }

    return {"production_plan": production_plan}

def convert_toyota_notes_to_production_plan(data: dict, headers=None) -> dict:
    """
    兼容“丰田三个月内示”这类 schema：
    {"notes": {"table": [
        {
          "品番": "...", "纳入区分":"...", "背番号":"...", "收容数":"...",
          "N月差异":"...", "N月合计":"...", "N+1月合计":"...", "N+2月合计":"...",
          "车名":"...",
          "日期-订货个数": {"1":0, "2":0, ..., "31":0}
        }, ...
    ]}}

    转为评测器可用的 production_plan：
    {
      "production_plan": {
        "headers": ["1",...,"31"],
        "models": [{"name": "...", "values":[...], "daily":{}, "sum": <int|None>}],
        "total": {"name":"TJ", ...}   # 若源数据没有则给空占位
      }
    }

    说明：
    - name 优先：品番，其次 背番号，其次 车名，最后用行号兜底。
    - sum 优先使用 N月合计（能转 int 就转），否则 None。
    - values 按 headers 顺序填充；若某天缺失键则填 None（不做推断补全）。
    """
    notes = data.get("notes", {})
    table = notes.get("table", [])
    if not isinstance(notes, dict) or not isinstance(table, list) or not table:
        return data

    # headers：优先用外部传入（通常是 GT.headers），否则默认 1..31
    if headers and isinstance(headers, list) and len(headers) > 0:
        headers = [str(h) for h in headers]
    else:
        headers = [str(i) for i in range(1, 32)]

    def to_int_or_none(x):
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, float)):
            return int(x)
        if isinstance(x, str):
            s = x.strip()
            if s == "":
                return None
            # 去掉千分位/空格
            s2 = s.replace(",", "")
            if re.fullmatch(r"-?\d+", s2):
                try:
                    return int(s2)
                except Exception:
                    return None
        return None

    models = []
    for i, row in enumerate(table):
        if not isinstance(row, dict):
            continue

        daily_map = row.get("日期-订货个数")
        if not isinstance(daily_map, dict):
            # 不是该 schema 就跳过
            continue

        name = row.get("品番") or row.get("背番号") or row.get("车名") or f"ROW_{i+1}"
        name = str(name).strip()

        values = []
        for h in headers:
            v = daily_map.get(h, None)
            values.append(to_int_or_none(v))

        s = to_int_or_none(row.get("N月合计"))

        models.append({
            "name": name,
            "values": values,
            "daily": daily_map,  # 保留原 daily，便于调试（评测时会忽略）
            "sum": s
        })

    production_plan = {
        "headers": headers,
        "models": models,
        "total": {"name": "TJ", "values": [], "daily": {}, "sum": None}
    }
    # 可选：保留 meta（不会计分，但方便排查）
    meta = {}
    if isinstance(notes.get("metadata"), dict):
        meta.update(notes["metadata"])
    if isinstance(notes.get("右上角"), dict):
        meta.update({f"右上角_{k}": v for k, v in notes["右上角"].items()})
    if meta:
        production_plan["meta"] = meta

    return {"production_plan": production_plan}

def coerce_to_same_schema(pred_data, label_data):
    """
    将 pred/label 统一到 {"production_plan": {...}} 的 schema。
    - label 若是 notes:[{production_plan:{...}}] 或 [{production_plan:{...}}] -> 提取 production_plan
    - pred 若是 notes.table -> 转成 production_plan（headers 以 GT 为准）
    - pred 若缺 meta -> 用 GT.meta 补齐（仅内存，不写回文件）
    """

    # -------- 1) 规整 label 到 {"production_plan": {...}} --------
    if isinstance(label_data, dict) and isinstance(label_data.get("notes"), list):
        notes = label_data["notes"]
        if len(notes) == 1 and isinstance(notes[0], dict) and "production_plan" in notes[0]:
            label_data = {"production_plan": notes[0]["production_plan"]}

    if isinstance(label_data, list) and len(label_data) == 1 and isinstance(label_data[0], dict) and "production_plan" in label_data[0]:
        label_data = {"production_plan": label_data[0]["production_plan"]}

    gt_plan = None
    gt_headers = None
    gt_meta = None
    if isinstance(label_data, dict) and isinstance(label_data.get("production_plan"), dict):
        gt_plan = label_data["production_plan"]
        gt_headers = gt_plan.get("headers")
        gt_meta = gt_plan.get("meta")

    # -------- 1.5) label: 丰田 notes.table(日期-订货个数) -> production_plan --------
    # 让 GT 能被当前评测器统计（评测器仅统计 production_plan.values[*] 与 .sum）
    if isinstance(label_data, dict) and isinstance(label_data.get("notes"), dict) and "table" in label_data["notes"]:
        t = label_data["notes"].get("table")
        if isinstance(t, list) and (len(t) == 0 or isinstance(t[0], dict)):
            # 判断是否为“日期-订货个数”schema
            is_toyota = any(isinstance(r, dict) and isinstance(r.get("日期-订货个数"), dict) for r in t)
            if is_toyota:
                label_data = convert_toyota_notes_to_production_plan(label_data)
                if isinstance(label_data, dict) and isinstance(label_data.get("production_plan"), dict):
                    gt_plan = label_data["production_plan"]
                    gt_headers = gt_plan.get("headers")
                    gt_meta = gt_plan.get("meta")

        # -------- 2a) pred: 丰田 notes.table(日期-订货个数) -> production_plan（headers 用 GT）--------
    if isinstance(pred_data, dict) and isinstance(pred_data.get("notes"), dict) and "table" in pred_data["notes"]:
        t = pred_data["notes"].get("table")
        if isinstance(t, list) and (len(t) == 0 or isinstance(t[0], dict)):
            is_toyota = any(isinstance(r, dict) and isinstance(r.get("日期-订货个数"), dict) for r in t)
            if is_toyota:
                pred_data = convert_toyota_notes_to_production_plan(pred_data, headers=gt_headers)

# -------- 2) pred: notes.table -> production_plan（headers 用 GT）--------
    if isinstance(pred_data, dict) and isinstance(pred_data.get("notes"), dict) and "table" in pred_data["notes"]:
        t = pred_data["notes"].get("table")
        if isinstance(t, list) and (len(t) == 0 or isinstance(t[0], dict)):
            pred_data = convert_notes_table_to_production_plan(pred_data, headers=gt_headers)

    # -------- 3) pred 缺 meta -> 用 GT meta 补齐 --------
    if isinstance(pred_data, dict) and isinstance(pred_data.get("production_plan"), dict) and isinstance(gt_meta, dict):
        pp = pred_data["production_plan"]
        if "meta" not in pp or not isinstance(pp.get("meta"), dict):
            pp["meta"] = {}
        for k in ("month", "year", "plant"):
            if k in gt_meta and k not in pp["meta"]:
                pp["meta"][k] = gt_meta[k]

    return pred_data, label_data



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_dir", required=True, help="GT(JSON)目录，例如 gt")
    parser.add_argument("--pred_dir", required=True, help="预测(JSON)目录，例如 pred")
    parser.add_argument("--out", default="ocr_accuracy_report.json", help="输出报告文件名")
    args = parser.parse_args()

    label_dir = Path(args.label_dir)
    pred_dir = Path(args.pred_dir)

    if not label_dir.exists():
        raise FileNotFoundError(f"label_dir 不存在: {label_dir}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"pred_dir 不存在: {pred_dir}")

    report = {
        "summary": {},
        "files": []
    }

    # 逐个文件对比：默认以“预测文件名”去 label_dir 找同名 GT
    pred_files = sorted(pred_dir.glob("*.json"))
    if not pred_files:
        raise FileNotFoundError(f"pred_dir 下没有 .json: {pred_dir}")

    total_fields = matched_fields = mismatched_fields = missing_in_pred = extra_in_pred = 0

    for pred_path in pred_files:
        label_path = label_dir / pred_path.name
        if not label_path.exists():
            print(f"[跳过] 找不到对应GT: {label_path}")
            continue

        print(f"\n=== 对比: {pred_path.name} ===")
        pred_data = load_prediction_file(str(pred_path))
        label_data = load_label_file(str(label_path))

        # ✅ 加这一句：做结构归一（下面第二部分会实现）
        pred_data, label_data = coerce_to_same_schema(pred_data, label_data)

        stats = calculate_accuracy(pred_data, label_data)

        report["files"].append({
            "file": pred_path.name,
            "stats": stats
        })

        total_fields += stats["total_fields"]
        matched_fields += stats["matched_fields"]
        mismatched_fields += stats["mismatched_fields"]
        missing_in_pred += stats["missing_in_pred"]
        extra_in_pred += stats["extra_in_pred"]

    total_comparable = matched_fields + mismatched_fields + missing_in_pred
    accuracy = (matched_fields / total_comparable * 100) if total_comparable else 0.0

    report["summary"] = {
        "total_fields": total_fields,
        "matched_fields": matched_fields,
        "mismatched_fields": mismatched_fields,
        "missing_in_pred": missing_in_pred,
        "extra_in_pred": extra_in_pred,
        "accuracy": accuracy,
        "accuracy_percentage": f"{accuracy:.2f}%",
        "total_comparable_fields": total_comparable
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n============================================================")
    print("OCR准确率统计结果(汇总)")
    print("============================================================")
    print(f"总字段数: {report['summary']['total_fields']}")
    print(f"匹配字段数: {report['summary']['matched_fields']}")
    print(f"不匹配字段数: {report['summary']['mismatched_fields']}")
    print(f"预测中缺失字段数: {report['summary']['missing_in_pred']}")
    print(f"预测中多余字段数: {report['summary']['extra_in_pred']}")
    print(f"\n准确率: {report['summary']['accuracy_percentage']}")
    print("============================================================")
    print(f"\n详细报告已保存到: {args.out}")



if __name__ == "__main__":
    main()

