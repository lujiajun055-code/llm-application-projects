import json

gt_path = r".\gt_0827\青岛计划0823.json"
pd_path = r".\pred_0827\青岛计划0823.json"

gt = json.load(open(gt_path,'r',encoding='utf-8'))
pd = json.load(open(pd_path,'r',encoding='utf-8'))

def unwrap(x):
    # 兼容 results 包装
    if isinstance(x, dict) and "results" in x:
        r = x["results"]
        if isinstance(r, list) and r:
            return r[0].get("label_value", r[0])
        if isinstance(r, dict) and r:
            first = list(r.values())[0]
            return first.get("label_value", first)
    return x

gt = unwrap(gt); pd = unwrap(pd)

gt_pp = gt.get("production_plan", {})
pd_pp = pd.get("production_plan", {})

print("GT headers len =", len(gt_pp.get("headers", [])))
print("PD headers len =", len(pd_pp.get("headers", [])))

def get_models(pp):
    # 兼容 models 或 rows
    if "models" in pp and isinstance(pp["models"], list):
        return {m.get("name"): m for m in pp["models"] if isinstance(m, dict)}
    if "rows" in pp and isinstance(pp["rows"], list):
        return {r.get("label"): r for r in pp["rows"] if isinstance(r, dict)}
    return {}

gt_models = get_models(gt_pp)
pd_models = get_models(pd_pp)

print("GT model count =", len(gt_models))
print("PD model count =", len(pd_models))

missing_models = sorted(set(gt_models) - set(pd_models))
extra_models = sorted(set(pd_models) - set(gt_models))
print("Missing models in pred:", missing_models[:20], ("... total "+str(len(missing_models)) if len(missing_models)>20 else ""))
print("Extra models in pred:", extra_models)

# 检查每个模型 values 长度是否等于 headers
gt_headers = gt_pp.get("headers", [])
pd_headers = pd_pp.get("headers", [])
for name in sorted(set(gt_models) & set(pd_models))[:10]:
    gm = gt_models[name]; pm = pd_models[name]
    gv = gm.get("values"); pv = pm.get("values")
    if isinstance(gv, list) and len(gv) != len(gt_headers):
        print("[WARN] GT values len mismatch:", name, len(gv), "!= headers", len(gt_headers))
    if isinstance(pv, list) and len(pv) != len(pd_headers):
        print("[WARN] PD values len mismatch:", name, len(pv), "!= headers", len(pd_headers))
