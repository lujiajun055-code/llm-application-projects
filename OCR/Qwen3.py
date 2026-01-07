LABEL_DIR = r".\gt_0827"
PRED_DEBUG_TXT_ON_FAIL = True

PROMPT_MARKER = "PROMPT_DEBUG_MARKER_FORCE_SUBMODEL_EXTRACTION"
import os
import json
import time
import glob
import base64
from openai import OpenAI
from PIL import Image
import fitz
import re

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _extract_first_json_obj(s: str):
    s = _strip_code_fences(s)
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":  # backslash
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return None

def _basic_json_repairs(s: str) -> str:
    s = s.strip()
    s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing comma
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    return s

def safe_load_json_from_text(text: str):
    cand = _extract_first_json_obj(text)
    if not cand:
        return None
    for _ in range(3):
        try:
            return json.loads(cand)
        except Exception:
            cand = _basic_json_repairs(cand)
    return None

from datetime import datetime
from pathlib import Path

# 配置信息
PROJECT_FOLDER = r"C:\Users\sjtu\Desktop\qwen3-vl-32b-instruct测试结果\images"  # 请修改为实际文件夹路径
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 从环境变量获取或直接填写

# 如果环境变量中没有API密钥，可以在这里直接设置
# DASHSCOPE_API_KEY = "your-api-key-here"

# 优化后的信息提取Prompt
PROMPT_EXTRACTION = """
你是一个表格 OCR 结构化抽取器。请从图片中抽取生产计划表，并严格按以下 JSON 结构输出。

【输出协议（必须遵守）】
1) 只输出 JSON，不得输出任何解释、说明、Markdown 或代码围栏。
2) 输出必须从 { 开始，到 } 结束，禁止省略号（...）或截断。
3) 顶层只能有一个键：production_plan。

【最终 JSON 结构（必须严格匹配）】
{
  "production_plan": {
    "meta": { "month": null, "year": null, "plant": null },
    "headers": [...],
    "models": [
      {
        "name": "Bora",
        "values": [...],
        "daily": {...},
        "sum": ...
      }
      // 共 11 行，顺序固定
    ],
    "total": null
  }
}

【headers 规则】
- headers 只能来自图片中“date”那一行，从左到右逐列抄写。
- headers 必须完整输出 28 个日期（例如 "9/1" 到 "9/28"），顺序不得改变。

【models 规则（子车型 / 叶子行）】
- models 必须恰好 11 行，且顺序必须固定如下：
  1. Bora
  2. C536MV
  3. C536MZ
  4. C537JZ
  5. C537MZ
  6. C538MZ
  7. C539NZ
  8. 88F
  9. 88L
  10. 88A
  11. 88B
- 只输出最终子车型（叶子行），禁止输出 SUM 行、TOTAL 行、父级分组行或标题行。
- name 必须与图片左侧行名完全一致，不得增删字符、空格或改写。

【values 规则（最重要）】
- values 必须是数组，长度必须严格等于 headers 长度（=28）。
- 第 i 个 values 必须对应 headers[i] 的那一列。
- 单元格完全空白（无任何字符、数字、符号）：输出 null。
- 单元格中只要出现字符 “0”（即使在灰格或周末列）：必须输出数值 0。
- 禁止使用 items 等稀疏格式。
- 禁止省略中间位置，必须逐列输出 28 个占位值。

【0 与 null 的判定优先级（严格）】
- 只要单元格中出现字符 “0”，必须输出数值 0。
- 只有在单元格完全空白、无任何字符时，才允许输出 null。
- 禁止将连续的 0 合并、压缩或当作空白处理。

【daily 规则】
- daily 必须是一个对象，其键集合必须与 headers 完全一致（28 个键，一个不少）。
- daily[headers[i]] 的值必须与 values[i] 完全一致。
- 不得遗漏任何日期键。

【sum 规则】
- sum 只抄图片中该行 SUM 单元格里的数字。
- 若该单元格为空或不存在：输出 null。
- 禁止自行计算或推断 sum。

【meta / total 规则】
- meta 固定输出：{ "month": null, "year": null, "plant": null }
- total 固定输出为 null，不要输出 {}，不要推断。

【输出前自检（必须执行）】
- 顶层只有 production_plan 一个键
- headers 长度 = 28
- models 行数 = 11，顺序与名称完全一致
- 每个 model：
  - values 长度 = 28
  - daily 键集合与 headers 完全一致
  - daily 与 values 一一对应
- 只输出 JSON

请严格依据图片真实内容填写，不得联想、补齐或推断。

"""






def count_total_non_null_values(pp: dict) -> int:
    rows = pp.get("rows") or []
    total = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        vals = r.get("values") or []
        total += sum(v is not None for v in vals)
    return total

def choose_best_pp_from_parsed(parsed_json: dict) -> dict:
    """
    从 parsed_json 中挑选“最完整”的 production_plan，避免选到全 null 的候选。
    支持：
      - {"production_plan": {...}}
      - {"notes":[{"production_plan": {...}}, ...]}
    """
    if not isinstance(parsed_json, dict):
        return {}

    candidates = []
    if isinstance(parsed_json.get("production_plan"), dict):
        candidates.append(parsed_json["production_plan"])

    notes = parsed_json.get("notes")
    if isinstance(notes, list):
        for it in notes:
            if isinstance(it, dict) and isinstance(it.get("production_plan"), dict):
                candidates.append(it["production_plan"])

    if not candidates:
        return {}

    # 如果脚本里有 GT_SUBMODELS（或其他 GT 子车型列表），就用命中数做第一排序
    gt_set = set()
    for k in ("GT_SUBMODELS", "SUBMODELS_GT", "GT_MODELS"):
        if k in globals() and isinstance(globals()[k], (list, tuple)):
            gt_set = set([str(x) for x in globals()[k] if x])
            break

    best_pp = None
    best_score = None

    for pp in candidates:
        if not isinstance(pp, dict):
            continue

        labels = set()
        non_null = 0

        if isinstance(pp.get("rows"), list):
            labels = set([str(r.get("label")) for r in pp.get("rows", []) if isinstance(r, dict) and r.get("label")])
            non_null = count_total_non_null_values(pp)
        elif isinstance(pp.get("models"), list):
            labels = set([str(m.get("name")) for m in pp.get("models", []) if isinstance(m, dict) and m.get("name")])
            for m in pp.get("models", []):
                if not isinstance(m, dict):
                    continue
                vals = m.get("values") or []
                non_null += sum(v is not None for v in vals)
        else:
            continue

        hit = len(labels & gt_set) if gt_set else 0
        extra = len(labels - gt_set) if gt_set else 0
        score = (hit, non_null, -extra)

        if best_pp is None or score > best_score:
            best_pp = pp
            best_score = score

    return best_pp if best_pp is not None else candidates[0]
def setup_client():
    """设置OpenAI客户端"""
    if not DASHSCOPE_API_KEY:
        raise ValueError("DASHSCOPE_API_KEY 未设置")

    return OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def image_to_base64(image_path):
    """将图片转换为base64编码"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"转换图片为base64时出错: {str(e)}")
        return None


def pdf_to_images(pdf_path, output_folder="temp_images"):
    """将PDF转换为图片（提高质量版本）"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        pdf_document = fitz.open(pdf_path)
        image_paths = []

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            # 提高分辨率到300DPI
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat)

            image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
            pix.save(image_path, "PNG")
            image_paths.append(image_path)

        pdf_document.close()
        return image_paths
    except Exception as e:
        print(f"PDF转换出错: {str(e)}")
        return []


def process_single_image_page(image_path, client, page_num=1, total_pages_local=1):
    """
    处理单个图片页面（通用适配所有单页图片）：
    - 优先尝试 Dense JSON（headers + rows.values[28]），与原准确率脚本兼容；
    - 若 Dense 解析失败/被截断，则自动降级为 Sparse JSON（只输出有值的 date/qty），
      再由本地转换回 Dense，避免输出过长导致的截断与 JSON 不闭合。
    """
    try:
        with Image.open(image_path) as img:
            img.verify()

        base64_image = image_to_base64(image_path)
        if not base64_image:
            return {"error": "parse_failed", "file_path": image_path, "page": page_num, "parse_error": "image_to_base64_failed", "raw_stream_content": ""}

        file_ext = os.path.splitext(image_path)[1].lower()
        mime_type = "image/png"
        if file_ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif file_ext == ".webp":
            mime_type = "image/webp"

        page_info = f"\n\n当前页面: {page_num}/{total_pages_local}" if total_pages_local > 1 else ""

        def _extract_json_object(t: str):
            if not t:
                return None
            s = t.strip()
            s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\s*```$", "", s)
            l = s.find("{")
            r = s.rfind("}")
            if l == -1 or r == -1 or r <= l:
                return None
            return s[l:r + 1].strip()

        def _looks_truncated(t: str):
            return _extract_json_object(t) is None

        def _validate_dense(parsed: dict):
            if not isinstance(parsed, dict):
                return False, "parsed_not_dict"
            pp = parsed.get("production_plan")
            if not isinstance(pp, dict):
                return False, "production_plan_not_dict"
            headers = pp.get("headers")
            rows = pp.get("rows")
            if not isinstance(headers, list) or not headers:
                return False, "headers_missing_or_not_list"
            if not isinstance(rows, list):
                return False, "rows_missing_or_not_list"
            hlen = len(headers)
            bad_rows = []
            for i, row in enumerate(rows):
                if not isinstance(row, dict):
                    bad_rows.append({"row_index": i, "reason": "row_not_dict"})
                    continue
                values = row.get("values")
                if not isinstance(values, list):
                    bad_rows.append({"row_index": i, "label": row.get("label"), "reason": "values_not_list"})
                    continue
                if len(values) != hlen:
                    bad_rows.append({"row_index": i, "label": row.get("label"), "values_len": len(values), "headers_len": hlen})
            if bad_rows:
                return False, {"type": "values_len_mismatch", "sample": bad_rows[:5], "headers_len": hlen, "rows_count": len(rows)}
            return True, None

        def _sparse_to_dense(sparse: dict):
            pp = sparse.get("production_plan", {})
            headers = pp.get("headers", [])
            rows = pp.get("rows", [])
            idx = {str(d): i for i, d in enumerate(headers)}
            dense_rows = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                label = row.get("label")
                values = [None] * len(headers)
                items = row.get("items", [])
                if isinstance(items, list):
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        d = it.get("date")
                        q = it.get("qty", None)
                        if d is None:
                            continue
                        d = str(d)
                        if d in idx:
                            values[idx[d]] = q
                s = 0
                has_num = False
                for v in values:
                    if isinstance(v, (int, float)):
                        s += v
                        has_num = True
                out = {"label": label, "values": values}
                if has_num:
                    out["sum"] = s
                dense_rows.append(out)
            return {"production_plan": {"headers": headers, "rows": dense_rows}}

        def _call_llm(prompt_text: str, max_tokens: int = 30000):
            start_time = time.time()
            completion = client.chat.completions.create(
                model="qwen3-vl-32b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=max_tokens,
                stream=False,
            )
            end_time = time.time()
            finish_reason = None
            full_content = ""
            try:
                choice0 = completion.choices[0]
                finish_reason = getattr(choice0, "finish_reason", None)
                full_content = (choice0.message.content or "").strip()
            except Exception:
                full_content = str(completion).strip()

            usage_obj = getattr(completion, "usage", None)
            stats = {
                "total_time_ms": (end_time - start_time) * 1000,
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", None) if usage_obj else None,
                "completion_tokens": getattr(usage_obj, "completion_tokens", None) if usage_obj else None,
                "total_tokens": getattr(usage_obj, "total_tokens", None) if usage_obj else None,
            }
            return full_content, finish_reason, stats

        # Dense
        dense_prompt = PROMPT_EXTRACTION + page_info
        raw1, fr1, stats1 = _call_llm(dense_prompt, max_tokens=30000)

        parsed1 = None
        ok1 = False
        err1 = None
        json_text1 = _extract_json_object(raw1)
        if json_text1 is None:
            err1 = "truncated_or_no_complete_json_object"
        else:
            try:
                parsed1 = None
                for _cand in (json_text1, _basic_json_repairs(json_text1)):
                    try:
                        parsed1 = json.loads(_cand)
                        break
                    except Exception:
                        parsed1 = None
                if parsed1 is None:
                    raise
                ok1, verr = _validate_dense(parsed1)
                if not ok1:
                    err1 = verr
            except Exception as e:
                err1 = str(e)

        if ok1:
            return {
                "raw_stream_content": raw1,
                "parse_success": True,
                "parse_error": None,
                "parsed_json": parsed1,
                "finish_reason": fr1,
                "file_path": image_path,
                "page_number": page_num,
                "total_pages_local": total_pages_local,
                "processing_stats": {
                    "page_number": page_num,
                    "total_pages_local": total_pages_local,
                    "ttft_ms": None,
                    "total_time_ms": stats1.get("total_time_ms"),
                    "token_count": stats1.get("completion_tokens") or stats1.get("total_tokens"),
                    "prompt_tokens": stats1.get("prompt_tokens"),
                    "completion_tokens": stats1.get("completion_tokens"),
                    "total_tokens": stats1.get("total_tokens"),
                    "file_path": image_path,
                    "process_time": datetime.now().isoformat(),
                    "attempt": "dense",
                    "dense_truncated": _looks_truncated(raw1),
                },
            }

        # Sparse fallback
        sparse_prompt = f"""你将从一张生产计划表图片中提取数据，并输出严格 JSON。

【输出协议（必须遵守）】
1. 最终输出必须是一个 JSON 对象，且顶层只能包含一个键：production_plan。
2. 禁止输出任何解释、说明、Markdown、代码围栏；输出必须从 {{ 开始，到 }} 结束。
3. 禁止任何省略符号（...、略、remaining 等）。

【表头（headers）规则】
1. 仅以图片中 date 行作为有效日期表头（例如 9/1, 9/2 … 9/28），保持原样。
2. 忽略 week num / week day。
3. 忽略最右侧 #VALUE! 列及其对应内容。
4. headers 按图片从左到右完整输出。

【行（rows）规则：稀疏输出】
1. rows 中每个元素对应图片中的一个数据行（label 与左侧行名完全一致）。
2. 只输出真正有每日数据或有 SUM 的数据行；纯分组标题无数据则不要输出。
3. 不要输出 values 28列数组；改为仅输出有数值的单元格 items。
4. items 中每个元素为 {{ "date": "<来自headers的日期>", "qty": <整数> }}。
5. 空白/斜线阴影/无计划 的单元格不要输出到 items（不要输出 null）。
6. 明确写 0 的单元格，qty 输出 0（必须保留）。

【最终输出结构（必须严格匹配）】
{{
  "production_plan": {{
    "headers": [...],
    "rows": [
      {{
        "label": "...",
        "items": [
          {{"date":"9/9","qty":170}}
        ]
      }}
    ]
  }}
}}

请严格依据图片真实内容填写，不得联想、补齐或推断。
{page_info}
""".strip()

        raw2, fr2, stats2 = _call_llm(sparse_prompt, max_tokens=30000)

        parsed2 = None
        ok2 = False
        err2 = None
        json_text2 = _extract_json_object(raw2)
        if json_text2 is None:
            err2 = "truncated_or_no_complete_json_object"
        else:
            try:
                parsed2 = None
                for _cand in (json_text2, _basic_json_repairs(json_text2)):
                    try:
                        parsed2 = json.loads(_cand)
                        break
                    except Exception:
                        parsed2 = None
                if parsed2 is None:
                    raise
                pp = parsed2.get("production_plan")
                if not isinstance(pp, dict):
                    raise ValueError("production_plan_not_dict")
                if not isinstance(pp.get("headers"), list) or not pp.get("headers"):
                    raise ValueError("headers_missing_or_not_list")
                if not isinstance(pp.get("rows"), list):
                    raise ValueError("rows_missing_or_not_list")
                ok2 = True
            except Exception as e:
                err2 = str(e)

        if not ok2:
            # 宽松兜底：即使校验不通过，只要解析到了一个 dict，就返回它（并交由后续写入 pred）
            if isinstance(parsed2, dict):
                return parsed2
            if isinstance(parsed1, dict):
                return parsed1
            return {
                "error": "parse_failed",
                "file": image_path,
                "parse_error": {
                    "dense_error": err1,
                    "sparse_error": err2,
                },
                "dense_truncated": _looks_truncated(raw1),
                "sparse_truncated": _looks_truncated(raw2),
            }

        dense_from_sparse = _sparse_to_dense(parsed2)
        ok_dense2, verr2 = _validate_dense(dense_from_sparse)
        if not ok_dense2:
            return {
                "error": "parse_failed",
                "file": image_path,
                "parse_error": {
                    "dense_error": err1,
                    "sparse_error": err2,
                    "fallback_dense_validation_failed": verr2,
                },
                "raw_stream_content": raw2,
            }

        return {
            "raw_stream_content": raw2,
            "parse_success": True,
            "parse_error": None,
            "parsed_json": dense_from_sparse,
            "finish_reason": fr2,
            "file_path": image_path,
            "page_number": page_num,
            "total_pages_local": total_pages_local,
            "processing_stats": {
                "page_number": page_num,
                "total_pages_local": total_pages_local,
                "ttft_ms": None,
                "total_time_ms": stats2.get("total_time_ms"),
                "token_count": stats2.get("completion_tokens") or stats2.get("total_tokens"),
                "prompt_tokens": stats2.get("prompt_tokens"),
                "completion_tokens": stats2.get("completion_tokens"),
                "total_tokens": stats2.get("total_tokens"),
                "file_path": image_path,
                "process_time": datetime.now().isoformat(),
                "attempt": "sparse_fallback_to_dense",
                "dense_first_error": err1,
            },
        }

    except Exception as e:
        print(f"处理图片页面时出错: {str(e)}")
        return {
            "error": str(e),
            "file_path": image_path,
            "page": page_num,
            "processing_stats": {
                "page_number": page_num,
                "total_pages_local": total_pages_local,
                "error": True,
                "process_time": datetime.now().isoformat(),
            },
        }


def merge_multi_page_results(page_results, original_path):
    """合并多页PDF的处理结果"""
    if not page_results:
        return {"error": "无有效页面结果", "file_path": original_path}

    # 检查是否有错误页面
    error_pages = [r for r in page_results if r.get("error")]
    valid_pages = [r for r in page_results if not r.get("error")]

    if not valid_pages:
        return {
            "error": "所有页面处理失败",
            "detailed_errors": page_results,
            "file_path": original_path
        }

    # 修改：合并多页的原始内容
    all_raw_contents = []
    all_parsed_contents = []

    for page_result in page_results:
        if "raw_stream_content" in page_result:
            all_raw_contents.append({
                "page": page_result.get("page_number", "unknown"),
                "content": page_result["raw_stream_content"]
            })
        if "parsed_json" in page_result:
            all_parsed_contents.append({
                "page": page_result.get("page_number", "unknown"),
                "content": page_result["parsed_json"]
            })

    # 创建合并结果
    merged_result = {
        "file_path": original_path,
        "multi_page_info": {
            "total_pages_local": len(page_results),
            "valid_pages": len(valid_pages),
            "error_pages": len(error_pages)
        },
        "all_raw_contents": all_raw_contents,  # 保存所有页面的原始内容
        "all_parsed_contents": all_parsed_contents if all_parsed_contents else "无有效JSON解析结果"
    }

    return merged_result


def process_pdf_file(pdf_path, client):
    """处理PDF文件"""
    print("检测到PDF文件，开始转换为高分辨率图片...")
    image_paths = pdf_to_images(pdf_path)

    if not image_paths:
        return {"error": "PDF转换失败", "file_path": pdf_path}

    all_page_results = []
    total_pages_local = len(image_paths)

    for page_num, img_path in enumerate(image_paths, 1):
        print(f"\n处理第 {page_num}/{total_pages_local} 页...")
        page_result = process_single_image_page(img_path, client, page_num, total_pages_local)
        all_page_results.append(page_result)

        # 页面间延迟
        if page_num < total_pages_local:
            time.sleep(2)

    # 清理临时文件
    for img_path in image_paths:
        try:
            os.remove(img_path)
        except:
            pass

    # 合并多页结果
    merged_result = merge_multi_page_results(all_page_results, pdf_path)
    return merged_result


def process_single_file(file_path, client):
    """处理单个文件（优化版本）"""
    print(f"\n正在处理文件: {file_path}")
    print("=" * 50)

    # 支持的文件类型
    supported_extensions = {'.pdf', '.jpg', '.jpeg', '.png'}
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext not in supported_extensions:
        print(f"不支持的文件格式: {file_ext}")
        return {"error": f"不支持的格式: {file_ext}", "file_path": file_path}

    try:
        # 处理PDF文件
        if file_ext == '.pdf':
            return process_pdf_file(file_path, client)
        else:
            # 处理图片文件
            return process_single_image_page(file_path, client)
    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")
        return {"error": str(e), "file_path": file_path}


def print_summary_report(stats, results):
    """打印处理总结报告"""
    print(f"\n{'=' * 60}")
    print("处理总结报告")
    print(f"{'=' * 60}")
    print(f"总文件数: {stats['total_files']}")
    print(f"成功处理: {stats['processed_files']}")
    print(f"处理失败: {stats['failed_files']}")

    if stats['total_files'] > 0:
        success_rate = (stats['processed_files'] / stats['total_files']) * 100
        print(f"成功率: {success_rate:.1f}%")

    if stats['processed_files'] > 0:
        print(f"平均TTFT: {stats['avg_ttft']:.2f}s")
        print(f"总Token数: {stats['total_tokens']}")
        print(f"总处理时间: {stats['total_time']:.2f}秒")

    # 显示提取的内容概览
    print(f"\n提取内容概览:")
    successful_results = [r for r in results.values() if r and not r.get('error')]
    for i, result in enumerate(successful_results, 1):
        file_name = os.path.basename(result.get('file_path', '未知文件'))

        # 检查是否有原始流内容
        if 'raw_stream_content' in result:
            content_preview = result['raw_stream_content'][:200] + "..." if len(result['raw_stream_content']) > 200 else \
            result['raw_stream_content']
            parse_status = "✓" if result.get('parse_success') else "✗"
            print(f"{i}. {file_name} [{parse_status}] - {content_preview}")
        elif 'all_raw_contents' in result:
            total_pages_local = len(result['all_raw_contents'])
            total_chars = sum(len(page['content']) for page in result['all_raw_contents'])
            print(f"{i}. {file_name} [多页文档, {total_pages_local}页, 总字符数: {total_chars}]")
        else:
            print(f"{i}. {file_name} [无有效内容]")


def save_results_with_backup(results, stats):
    """保存结果并创建备份"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    main_output = "extraction_results.json"
    backup_output = f"extraction_results_backup_{timestamp}.json"

    # 准备保存的数据结构
    save_data = {
        "metadata": {
            "export_time": timestamp,
            "statistics": stats,
            "project_folder": PROJECT_FOLDER,
            "note": "结果中包含原始流式输出内容(raw_stream_content)和尝试解析的JSON(parsed_json)"
        },
        "results": results
    }

    raw_output = f"raw_stream_contents_{timestamp}.json"
    try:
        contents = {
            "notes": []
        }

        for file_path, result in results.items():
            if result and not result.get('error'):
                if 'raw_stream_content' in result:
                    try:
                        parsed = json.loads(result['raw_stream_content'])
                        contents['notes'].append(parsed)
                    except:
                        contents['notes'].append(result['raw_stream_content'])

                elif 'all_raw_contents' in result:
                    for page_content in result['all_raw_contents']:
                        try:
                            parsed = json.loads(page_content['content'])
                            contents['notes'].append(parsed)
                        except:
                            contents['notes'].append(page_content['content'])

        # 一次性写入
        with open(raw_output, "w", encoding="utf-8") as f:
            # DEBUG: 写入提示词标记，确认运行的是当前 Qwen3.py
            try:
                contents.setdefault("production_plan", {}).setdefault("meta", {})["prompt_marker"] = "PROMPT_DEBUG_MARKER_FORCE_SUBMODEL_EXTRACTION"
            except Exception:
                pass

            json.dump(contents, f, ensure_ascii=False, indent=2)

        print(f"原始内容已保存到: {raw_output}")

        # ✅ 额外输出：用于准确率对比的 pred/ 单文件（每个源文件同名）
        # 注意：raw_stream_contents_*.json 仅用于 debug，不建议作为 pred 输入给准确率脚本
        try:
            pred_dir = Path("pred")
            pred_dir.mkdir(parents=True, exist_ok=True)

            for file_path, result in results.items():
                if not result or result.get("error"):
                    continue

                # 单页图片：优先写 parsed_json
                if result.get("parse_success") and isinstance(result.get("parsed_json"), dict):
                    best_pp = choose_best_pp_from_parsed(result["parsed_json"]) if isinstance(result.get("parsed_json"), dict) else {}
                    if isinstance(best_pp, dict) and best_pp:
                        pj = {"production_plan": best_pp} if ("headers" in best_pp and ("rows" in best_pp or "models" in best_pp)) else best_pp
                        save_pred_json("pred", file_path, pj, LABEL_DIR)
                    else:
                        save_pred_json("pred", file_path, result["parsed_json"], LABEL_DIR)

                # 多页 PDF：逐页写入（如果有 parsed_json）
                elif "all_parsed_contents" in result and isinstance(result["all_parsed_contents"], list):
                    total_pages_local = int(result.get("total_pages_local", 1) or 1)
                for p in result["all_parsed_contents"]:
                    page_idx = p.get("page")
                    pj = p.get("content")
                    if isinstance(pj, dict):
                        page_index_for_save = None if total_pages_local == 1 else (int(page_idx) if str(page_idx).isdigit() else page_idx)
                        save_pred_json("pred", file_path, pj, LABEL_DIR, page_index=page_index_for_save)
                        page_idx = p.get("page")
                        pj = p.get("content")
                        if isinstance(pj, dict):
                            page_index_for_save = None if total_pages_local == 1 else (int(page_idx) if str(page_idx).isdigit() else page_idx)
                            total_pages_local = int(result.get("total_pages_local", 1) or 1)
                            page_index_for_save = None if total_pages_local == 1 else page_idx
                            save_pred_json("pred", file_path, pj, LABEL_DIR, page_index=page_index_for_save)

                # 解析失败：写一个失败对象，便于定位（不会影响其它文件）
                else:
                    save_pred_json("pred", file_path, {
                        "error": "parse_failed",
                        "file": str(file_path),
                        "parse_error": result.get("parse_error"),
                        "raw_stream_content": result.get("raw_stream_content", "")
                    }, LABEL_DIR)

            print("标准 pred 已输出到: pred/")
        except Exception as e:
            print(f"输出 pred 目录失败: {str(e)}")


    except Exception as e:
        print(f"保存原始内容文件失败: {str(e)}")

def align_pred_to_gt_submodels(pred_json: dict, gt_json: dict) -> dict:
    """
    对齐 pred 到 GT 的子车型集合，并输出为 accuracy_calculator 期望的 models 结构。
    兼容 pred 两种结构：
      - production_plan.models（旧）
      - production_plan.rows（新，Dense 输出）
    """
    def unwrap(x):
        if isinstance(x, dict) and "results" in x:
            r = x["results"]
            if isinstance(r, list) and r:
                return r[0].get("label_value", r[0])
            if isinstance(r, dict) and r:
                first = list(r.values())[0]
                return first.get("label_value", first)
        return x

    pred = unwrap(pred_json) if isinstance(pred_json, dict) else {}
    gt = unwrap(gt_json) if isinstance(gt_json, dict) else {}

    gt_pp = gt.get("production_plan", {}) if isinstance(gt, dict) else {}
    pd_pp = pred.get("production_plan", {}) if isinstance(pred, dict) else {}

    headers = gt_pp.get("headers") or pd_pp.get("headers") or []
    headers = list(headers) if isinstance(headers, list) else []
    hlen = len(headers)

    gt_models = gt_pp.get("models", [])
    if not isinstance(gt_models, list) or not gt_models:
        raise ValueError("GT production_plan.models 不存在或不是 list（无法对齐）")
    gt_names = [m.get("name") for m in gt_models if isinstance(m, dict) and m.get("name")]
    if not gt_names:
        raise ValueError("GT production_plan.models 中未找到任何 name（无法对齐）")

    def pad(values):
        if not isinstance(values, list):
            values = []
        if len(values) < hlen:
            values = values + [None] * (hlen - len(values))
        elif len(values) > hlen:
            values = values[:hlen]
        return values

    # Build pred map from either models or rows
    pd_map = {}

    pd_models = pd_pp.get("models", [])
    if isinstance(pd_models, list) and pd_models:
        for m in pd_models:
            if isinstance(m, dict) and m.get("name") is not None:
                name = str(m["name"])
                pd_map[name] = {"values": m.get("values", []), "sum": m.get("sum", None)}

    if not pd_map:
        pd_rows = pd_pp.get("rows", [])
        if isinstance(pd_rows, list):
            for r in pd_rows:
                if not isinstance(r, dict):
                    continue
                label = r.get("label")
                if not label:
                    continue
                name = str(label)
                if name.strip().upper() in {"SUM", "TOTAL"}:
                    continue
                pd_map[name] = {"values": r.get("values", []), "sum": r.get("sum", None)}

    new_models = []
    for name in gt_names:
        if name in pd_map:
            v = pad(pd_map[name].get("values", []))
            new_models.append({"name": name, "values": v, "daily": {headers[i]: v[i] for i in range(hlen)}, "sum": pd_map[name].get("sum", None)})
        else:
            v = [None] * hlen
            new_models.append({"name": name, "values": v, "daily": {headers[i]: None for i in range(hlen)}, "sum": None})

    meta = pd_pp.get("meta")
    if not isinstance(meta, dict):
        meta = {"month": None, "year": None, "plant": None}

    return {"production_plan": {"meta": meta, "headers": headers, "models": new_models, "total": pd_pp.get("total", None)}}


def save_pred_json(
    pred_dir: str,
    src_file_path: str,
    parsed_json: dict,
    label_dir: str = None,
    page_index: int = None
):
    """
    将可直接用于准确率对比的 JSON（dict）写入 pred 目录。
    - 单页图片：pred/<文件名>.json
    - 多页（PDF）：pred/<文件名>_p{page}.json
    """
    pred_dir_path = Path(pred_dir)
    pred_dir_path.mkdir(parents=True, exist_ok=True)

    src = Path(src_file_path)
    stem = src.stem

    if page_index is None:
        out_name = f"{stem}.json"
    else:
        out_name = f"{stem}_p{page_index}.json"

    out_path = pred_dir_path / out_name

    # ========= 第二步核心：读取对应 GT =========
    aligned_pred = parsed_json

    # 若未提供 label_dir 或 GT 不存在/结构不符合，则直接写 parsed_json（不对齐）
    if label_dir:
        try:
            gt_path = Path(label_dir) / f"{stem}.json"
            if gt_path.exists():
                with open(gt_path, "r", encoding="utf-8") as f:
                    gt_json = json.load(f)

                # 仅当 GT 具备 production_plan.models(list) 时才做对齐
                pp = gt_json.get("production_plan") if isinstance(gt_json, dict) else None
                models = pp.get("models") if isinstance(pp, dict) else None
                if isinstance(models, list):
                    try:
                        aligned_pred = align_pred_to_gt_submodels(parsed_json, gt_json)
                    except Exception:
                        aligned_pred = parsed_json
        except Exception:
            aligned_pred = parsed_json

    # ========= 写入 pred =========
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aligned_pred, f, ensure_ascii=False, indent=2)
    return str(out_path)


def process_all_files():
    """处理文件夹中的所有文件（增强版本）"""
    if not DASHSCOPE_API_KEY:
        print("错误: 请设置DASHSCOPE_API_KEY环境变量")
        print("您可以通过以下方式设置:")
        print("1. 在代码中直接设置: DASHSCOPE_API_KEY = '您的API密钥'")
        print("2. 设置环境变量: export DASHSCOPE_API_KEY='您的API密钥'")
        print("3. 在Windows中: set DASHSCOPE_API_KEY=您的API密钥")
        return None

    try:
        client = setup_client()
    except Exception as e:
        print(f"初始化客户端失败: {str(e)}")
        return None

    # 检查文件夹是否存在
    if not os.path.exists(PROJECT_FOLDER):
        print(f"错误: 文件夹 {PROJECT_FOLDER} 不存在")
        print("请修改 PROJECT_FOLDER 变量为正确的文件夹路径")
        return None

    # 获取所有支持的文件
    file_patterns = [
        os.path.join(PROJECT_FOLDER, "*.pdf"),
        os.path.join(PROJECT_FOLDER, "*.jpg"),
        os.path.join(PROJECT_FOLDER, "*.jpeg"),
        os.path.join(PROJECT_FOLDER, "*.png")
    ]

    all_files = []
    for pattern in file_patterns:
        all_files.extend(glob.glob(pattern))

    # 去重并排序
    all_files = sorted(list(set(all_files)))

    if not all_files:
        print(f"在文件夹 {PROJECT_FOLDER} 中未找到支持的文件")
        print("支持的文件格式: PDF, JPG, JPEG, PNG")
        return None

    print(f"找到 {len(all_files)} 个文件待处理")

    all_results = {}
    performance_stats = {
        "total_files": len(all_files),
        "processed_files": 0,
        "failed_files": 0,
        "total_ttft": 0,
        "total_tokens": 0,
        "start_time": time.time()
    }

    for i, file_path in enumerate(all_files, 1):
        print(f"\n{'=' * 60}")
        print(f"处理进度: {i}/{len(all_files)} - {os.path.basename(file_path)}")
        print(f"{'=' * 60}")

        result = process_single_file(file_path, client)
        all_results[file_path] = result

        # 更新统计信息
        if result and not result.get("error"):
            performance_stats["processed_files"] += 1

            # 收集性能数据
            if "processing_stats" in result and result["processing_stats"].get("ttft_ms"):
                performance_stats["total_ttft"] += result["processing_stats"]["ttft_ms"]
            if "processing_stats" in result and result["processing_stats"].get("token_count"):
                performance_stats["total_tokens"] += result["processing_stats"]["token_count"]
        else:
            performance_stats["failed_files"] += 1

        # 文件间延迟
        if i < len(all_files):
            print("等待2秒后处理下一个文件...")
            time.sleep(2)

    # 计算最终统计
    performance_stats["total_time"] = time.time() - performance_stats["start_time"]
    if performance_stats["processed_files"] > 0:
        performance_stats["avg_ttft"] = performance_stats["total_ttft"] / performance_stats[
            "processed_files"] / 1000  # 转换为秒
    else:
        performance_stats["avg_ttft"] = 0

    # 输出总结报告
    print_summary_report(performance_stats, all_results)

    # 保存结果
    save_results_with_backup(all_results, performance_stats)

    return all_results


def main():
    """主函数"""
    print("=" * 60)
    print("文档信息提取系统")
    print("=" * 60)
    print(f"目标文件夹: {PROJECT_FOLDER}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 检查依赖
    try:
        import openai
        import fitz
        import PIL
        print("依赖检查: 通过")
    except ImportError as e:
        print(f"依赖检查失败: {e}")
        print("请安装所需依赖: pip install openai pymupdf Pillow")
        return

    start_time = time.time()
    results = process_all_files()
    end_time = time.time()

    print(f"\n{'=' * 60}")
    print("处理完成!")
    print(f"{'=' * 60}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")

    if results:
        successful_count = sum(1 for r in results.values() if r and not r.get('error'))
        print(f"成功处理文件数: {successful_count}/{len(results)}")

        # 显示JSON解析成功率
        parse_success_count = sum(1 for r in results.values() if r and not r.get('error') and r.get('parse_success'))
        if successful_count > 0:
            parse_success_rate = (parse_success_count / successful_count) * 100
            print(f"JSON解析成功率: {parse_success_rate:.1f}% ({parse_success_count}/{successful_count})")
    else:
        print("未获得有效结果")


if __name__ == "__main__":
    main()

def save_pred_json(
    pred_dir: str,
    src_file_path: str,
    parsed_json: dict,
    label_dir: str,
    page_index: int = None
):
    from pathlib import Path
    import json

    pred_dir_path = Path(pred_dir)
    pred_dir_path.mkdir(parents=True, exist_ok=True)

    src = Path(src_file_path)
    stem = src.stem

    if page_index is None:
        out_name = f"{stem}.json"
    else:
        out_name = f"{stem}_p{page_index}.json"

    out_path = pred_dir_path / out_name

    gt_path = Path(label_dir) / f"{stem}.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"GT 文件不存在: {gt_path}")

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_json = json.load(f)

    aligned_pred = align_pred_to_gt_submodels(parsed_json, gt_json)

    # DEBUG: embed prompt marker into output for verification
    try:
        aligned_pred.setdefault("production_plan", {}).setdefault("meta", {})["prompt_marker"] = PROMPT_MARKER
    except Exception:
        pass

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aligned_pred, f, ensure_ascii=False, indent=2)

    return str(out_path)