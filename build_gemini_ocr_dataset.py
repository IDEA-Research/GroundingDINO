#!/usr/bin/env python3
"""
從 MongoDB 匯出以 gemini-3.1-flash 判讀結果為 GT 的螢幕 OCR 資料集，
用於評估 / 微調地端 VLM（GLM-4.6V 等）或較小的 OCR 模型。

資料來源：medical_monitor_db.screen_analysis（GroundingDINO 裁切的螢幕圖 + LLM JSON）

處理規則：
- 只取 llm_model 為 gemini-3.1-flash 系列且 success=True 的紀錄
- 排除送進 VLM 的是 sanitized（移除螢幕內容）圖片的紀錄，因為 GT 與原圖不對應
- 設備名稱正規化為固定類別，medical_values 依設備 schema 對齊（缺值補 null）
- 預設排除 ASUS 顯示器：任何模型曾辨識出 ASUS 的 session 整批排除（--exclude_devices / --exclude_keywords）
- 若有人工修正（corrected_model / corrected_medical_values）則覆蓋 Gemini 結果
- 依 device_templates 的合理範圍與邏輯規則標記 rule_violations，方便過濾
- 以影片 (video_analysis_id) 為單位切 train/val/test，避免相鄰畫面洩漏

輸出：
  <out>/images/<split>/<id>.jpg
  <out>/<split>.jsonl          每行一筆完整標注
  <out>/<split>_sharegpt.json  VLM 微調格式（LLaMA-Factory / ShareGPT）
  <out>/stats.json

使用方式：
  python build_gemini_ocr_dataset.py --out datasets/screen_ocr_gemini31
"""

import argparse
import collections
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

from pymongo import MongoClient

GEMINI_MODELS = [
    "google/gemini-3.1-flash-lite",
    "google/gemini-3.1-flash-lite-preview",
]

# 每種設備要輸出的欄位
SCHEMAS = {
    "philips": ["heart_rate", "respiration_rate", "bp_systolic", "bp_diastolic", "bp_map", "spo2"],
    "ventilator": ["pip", "peep", "map", "fio2"],
    "invos": ["rso2_left", "rso2_left_avg", "rso2_right", "rso2_right_avg"],
    "osypka": ["ci_meas", "svi_meas"],
    "unknown": [],
}

DEVICE_SCHEMA = {
    "Philips IntelliVue MP30": "philips",
    "Philips IntelliVue MP60": "philips",
    "ASUS Display (Philips UI)": "philips",
    "Dräger Infinity C500": "ventilator",
    "Dräger Evita V600": "ventilator",
    "SERVO-U Ventilator": "ventilator",
    "Medtronic INVOS 5100C": "invos",
    "Osypka Aesculon Hemodynamic Monitor": "osypka",
    "unknown": "unknown",
}

# 欄位別名 -> 標準欄位
KEY_ALIASES = {
    "pmean": "map",
    "ppeak": "pip",
    "o2_conc": "fio2",
}

# 依 device_templates 的合理範圍
VALUE_RANGES = {
    "heart_rate": (0, 350),
    "respiration_rate": (0, 200),
    "spo2": (0, 100),
    "bp_systolic": (10, 350),
    "bp_diastolic": (0, 250),
    "bp_map": (5, 300),
    "pip": (0, 120),
    "peep": (0, 50),
    "map": (0, 80),
    "fio2": (21, 100),
    "rso2_left": (15, 100),
    "rso2_left_avg": (15, 100),
    "rso2_right": (15, 100),
    "rso2_right_avg": (15, 100),
}


def normalize_device(name):
    """將 Gemini 回傳的各種設備名稱變體正規化。"""
    if not isinstance(name, str) or not name.strip():
        return "unknown"
    n = " ".join(name.replace("_", " ").replace("-", " ").lower().split())
    if n == "unknown":
        return "unknown"
    if "asus" in n:
        return "ASUS Display (Philips UI)"
    if "aesculon" in n or "osypka" in n:
        return "Osypka Aesculon Hemodynamic Monitor"
    if "servo" in n:
        return "SERVO-U Ventilator"
    if "v600" in n:
        return "Dräger Evita V600"
    if "c500" in n:
        return "Dräger Infinity C500"
    if "invos" in n:
        return "Medtronic INVOS 5100C"
    if "philips" in n or "intel" in n:
        # MP60 單獨出現時才視為 MP60；MP20/30/60 合併寫法與 MP300（誤讀）歸為 MP30
        if "mp60" in n and "30" not in n:
            return "Philips IntelliVue MP60"
        return "Philips IntelliVue MP30"
    return "unknown"


def to_number(v):
    """將數值或字串轉為 int/float，無法解析則回傳 None。"""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v) if float(v).is_integer() else float(v)
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in ("", "null", "none", "--", "---", "n/a"):
            return None
        try:
            f = float(s)
        except ValueError:
            return None
        return int(f) if f.is_integer() else f
    return None


def align_values(raw_values, schema_keys):
    """依 schema 對齊欄位，回傳 (values, extra_keys)。"""
    lowered = {}
    for k, v in (raw_values or {}).items():
        key = KEY_ALIASES.get(k.lower(), k.lower())
        lowered.setdefault(key, v)
    values = {k: to_number(lowered.get(k)) for k in schema_keys}
    extra = sorted(k for k in lowered if k not in schema_keys)
    return values, extra


def check_rules(values, schema):
    """回傳違反 device_templates 規則的描述列表。"""
    issues = []
    for k, v in values.items():
        if v is None or k not in VALUE_RANGES:
            continue
        lo, hi = VALUE_RANGES[k]
        if not lo <= v <= hi:
            issues.append(f"{k}_out_of_range")
    g = values.get
    if schema == "philips":
        s, m, d = g("bp_systolic"), g("bp_map"), g("bp_diastolic")
        if None not in (s, m, d) and not s > m > d:
            issues.append("bp_order")
        if None not in (g("heart_rate"), g("respiration_rate")) and g("heart_rate") <= g("respiration_rate"):
            issues.append("hr_le_rr")
    elif schema == "ventilator":
        p, m, e = g("pip"), g("map"), g("peep")
        if None not in (p, m, e) and not p > m > e:
            issues.append("pip_map_peep_order")
    return issues


def group_split(group_sizes, seed, ratios):
    """以影片為單位分配 split，貪婪地讓各 split 樣本數接近目標比例。

    串流影片長度差異極大（數筆到數千筆），單純依影片數切分會嚴重失衡。
    """
    names = ["train", "val", "test"]
    ids = sorted(group_sizes, key=str)
    random.Random(seed).shuffle(ids)
    ids.sort(key=lambda g: -group_sizes[g])  # 大影片先分配，同大小者保持隨機順序
    total = sum(group_sizes.values())
    counts = dict.fromkeys(names, 0)
    out = {}
    for gid in ids:
        split = max(names, key=lambda s: ratios[names.index(s)] * total - counts[s])
        out[gid] = split
        counts[split] += group_sizes[gid]
    return out


def build_prompt(schema_keys_by_device):
    """產生給學生模型的精簡提示詞（訓練與評估共用）。"""
    lines = [
        "Identify the medical device on this screen and read its displayed values.",
        "Return ONLY a JSON object: {\"device\": <device>, \"values\": {...}}.",
        "Use null for any value that is blank, '--', occluded, or obscured by glare. Do not guess.",
        "Devices and their value keys:",
    ]
    for device, keys in schema_keys_by_device.items():
        lines.append(f"- {device}: {', '.join(keys) if keys else '(no values)'}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="以 gemini-3.1-flash 結果為 GT 匯出螢幕 OCR 資料集")
    parser.add_argument("--out", default="datasets/screen_ocr_gemini31")
    parser.add_argument("--mongo_uri", default="mongodb://localhost:27017")
    parser.add_argument("--db", default="medical_monitor_db")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="0.8,0.1,0.1", help="train,val,test 比例（以影片為單位）")
    parser.add_argument("--include_unknown", action="store_true", help="保留 device=unknown 的負樣本")
    parser.add_argument("--exclude_devices", default="ASUS Display (Philips UI)",
                        help="以逗號分隔、要排除的正規化設備名稱（預設排除 ASUS 顯示器）")
    parser.add_argument("--exclude_keywords", default="asus,華碩",
                        help="任一模型的設備名稱或描述出現這些關鍵字（不分大小寫）時，"
                             "排除該 session（圖片資料夾）的所有裁切圖；ASUS 顯示器常被標成 MP30 或 unknown")
    parser.add_argument("--dry_run", action="store_true", help="只統計，不複製圖片")
    args = parser.parse_args()

    ratios = [float(x) for x in args.split.split(",")]
    excluded_keywords = [x.strip().lower() for x in args.exclude_keywords.split(",") if x.strip()]
    excluded_devices = {x.strip() for x in args.exclude_devices.split(",") if x.strip()}
    db = MongoClient(args.mongo_uri)[args.db]
    out_dir = Path(args.out)

    # 找出出現過 ASUS 等關鍵字的 session：同一串流 session 的畫面都是同一台顯示器，
    # 但 LLM 不一定每次都提到品牌，因此以 session 為單位排除（參考所有模型的判讀）
    excluded_sessions = set()
    if excluded_keywords:
        for d in db.screen_analysis.find(
            {}, {"screen_image_path": 1, "detected_model": 1, "raw_analysis.description": 1}
        ):
            text = f"{d.get('detected_model')} {(d.get('raw_analysis') or {}).get('description')}".lower()
            if any(kw in text for kw in excluded_keywords) and d.get("screen_image_path"):
                excluded_sessions.add(Path(d["screen_image_path"]).parent.parent.name)

    query = {"llm_model": {"$in": GEMINI_MODELS}, "success": True}
    docs = list(db.screen_analysis.find(query))

    # frame -> video 對應，用於分組切分與追溯
    frame_ids = list({d["frame_result_id"] for d in docs})
    frame_map = {
        f["_id"]: f for f in db.frame_results.find(
            {"_id": {"$in": frame_ids}}, {"video_analysis_id": 1, "minute": 1}
        )
    }
    video_ids = list({f["video_analysis_id"] for f in frame_map.values()})
    video_map = {
        v["_id"]: v for v in db.video_analysis.find(
            {"_id": {"$in": video_ids}}, {"video_name": 1, "source": 1, "camera_name": 1}
        )
    }

    stats = collections.Counter()
    records = []
    for d in docs:
        stats["total_success"] += 1
        img_path = d["screen_image_path"]
        vlm_path = d.get("screen_image_path_for_vlm") or img_path
        if vlm_path != img_path:
            stats["skip_sanitized_image"] += 1
            continue
        if not os.path.exists(img_path):
            stats["skip_missing_image"] += 1
            continue

        raw = d.get("raw_analysis") or {}
        device = normalize_device(d.get("corrected_model") or d.get("detected_model"))
        if device == "unknown" and not args.include_unknown:
            stats["skip_unknown_device"] += 1
            continue
        if device in excluded_devices:
            stats["skip_excluded_device"] += 1
            continue
        if Path(img_path).parent.parent.name in excluded_sessions:
            stats["skip_excluded_session"] += 1
            continue
        schema = DEVICE_SCHEMA[device]

        values, extra_keys = align_values(d.get("medical_values"), SCHEMAS[schema])
        corrected = d.get("corrected_medical_values") or {}
        for k, v in corrected.items():
            key = KEY_ALIASES.get(k.lower(), k.lower())
            if key in values:
                values[key] = to_number(v)
        gt_source = "human_corrected" if (corrected or d.get("is_model_corrected")) else "gemini"

        frame = frame_map.get(d["frame_result_id"], {})
        video_id = frame.get("video_analysis_id")
        video = video_map.get(video_id, {})
        sample_id = hashlib.sha1(str(d["_id"]).encode()).hexdigest()[:16]

        records.append({
            "id": sample_id,
            "source_image": img_path,
            "device": device,
            "schema": schema,
            "values": values,
            "gt_source": gt_source,
            "gt_model": d["llm_model"],
            "rule_violations": check_rules(values, schema),
            "has_glare": bool(d.get("has_glare")),
            "has_occlusion": bool(d.get("has_occlusion")),
            "extra_keys": extra_keys,
            "raw_model_name": d.get("detected_model"),
            "raw_numbers": raw.get("numbers"),
            "description": raw.get("description"),
            "mongo_screen_analysis_id": str(d["_id"]),
            "video_analysis_id": str(video_id) if video_id else None,
            "video_name": video.get("video_name"),
            "video_source": video.get("source"),
            "camera_name": video.get("camera_name"),
            "frame_time_seconds": frame.get("minute"),
            "analyzed_at": d.get("analyzed_at").isoformat() if d.get("analyzed_at") else None,
        })

    # 依各影片的主要設備分層後再切分，避免稀有設備全部落在 val/test
    device_counts = collections.defaultdict(collections.Counter)
    for r in records:
        if r["device"] != "unknown":
            device_counts[r["video_analysis_id"]][r["device"]] += 1
    strata = collections.defaultdict(collections.Counter)
    for r in records:
        vid = r["video_analysis_id"]
        dominant = device_counts[vid].most_common(1)[0][0] if device_counts[vid] else "unknown"
        strata[dominant][vid] += 1
    split_of = {}
    for stratum in sorted(strata):
        split_of.update(group_split(strata[stratum], args.seed, ratios))
    for r in records:
        r["split"] = split_of[r["video_analysis_id"]]

    devices_in_data = sorted({r["device"] for r in records})
    prompt = build_prompt({dev: SCHEMAS[DEVICE_SCHEMA[dev]] for dev in devices_in_data})

    by_split = collections.defaultdict(list)
    for r in records:
        by_split[r["split"]].append(r)

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, rows in by_split.items():
            img_dir = out_dir / "images" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            sharegpt = []
            with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
                for r in sorted(rows, key=lambda x: x["id"]):
                    rel = f"images/{split}/{r['id']}.jpg"
                    shutil.copy2(r["source_image"], out_dir / rel)
                    r["image"] = rel
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    target = json.dumps({"device": r["device"], "values": r["values"]}, ensure_ascii=False)
                    sharegpt.append({
                        "id": r["id"],
                        "images": [rel],
                        "messages": [
                            {"role": "user", "content": "<image>" + prompt},
                            {"role": "assistant", "content": target},
                        ],
                    })
            with open(out_dir / f"{split}_sharegpt.json", "w", encoding="utf-8") as f:
                json.dump(sharegpt, f, ensure_ascii=False, indent=1)
        (out_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    summary = {
        "gt_models": GEMINI_MODELS,
        "filter": dict(stats),
        "kept": len(records),
        "splits": {s: len(v) for s, v in sorted(by_split.items())},
        "videos_per_split": {
            s: len({r["video_analysis_id"] for r in v}) for s, v in sorted(by_split.items())
        },
        "devices_per_split": {
            s: dict(collections.Counter(r["device"] for r in v).most_common())
            for s, v in sorted(by_split.items())
        },
        "gt_source": dict(collections.Counter(r["gt_source"] for r in records)),
        "with_rule_violations": sum(1 for r in records if r["rule_violations"]),
        "rule_violation_types": dict(collections.Counter(
            i for r in records for i in r["rule_violations"]).most_common()),
        "with_glare_or_occlusion": sum(1 for r in records if r["has_glare"] or r["has_occlusion"]),
        "null_ratio": round(
            sum(v is None for r in records for v in r["values"].values())
            / max(1, sum(len(r["values"]) for r in records)), 4),
    }
    if not args.dry_run:
        with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
