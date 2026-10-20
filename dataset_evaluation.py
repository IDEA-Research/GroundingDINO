"""以人工訂正過的 Session 作為標準答案，比對資料集影片的分析結果。

先以（擷取秒數, 標準設備型號）分組，同組有多台設備時再以 bbox 位置一對一配對。
不使用螢幕編號，因為 GroundingDINO 的偵測順序與數量每次都可能不同。
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path

_QUERY_ENGINE_DIR = Path(__file__).resolve().parent / "query_engine"
if str(_QUERY_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_QUERY_ENGINE_DIR))

from nicu_qe.aliases import (  # noqa: E402
    UNKNOWN_FAMILY,
    classify_device,
    get_registry,
    normalize_key_text,
)
from nicu_qe.models import Q_NON_NUMERIC  # noqa: E402
from nicu_qe.normalize import normalize_medical_values  # noqa: E402
from screen_bbox import iou, same_screen, valid_bbox  # noqa: E402

UNMAPPED_KEY_PREFIX = "~"
MAX_DETAIL_ROWS = 200


def _timestamp_value(value):
    if isinstance(value, datetime):
        return value.timestamp()
    return float("-inf")


def _extract_values(medical_values, corrected_medical_values, family):
    """將醫療數值正規化為 {標準欄位: 數值或文字}。"""
    normalized = normalize_medical_values(
        medical_values, corrected_medical_values, family
    )
    values = dict(normalized.m)
    for key, flags in normalized.q.items():
        if Q_NON_NUMERIC in flags and key in normalized.raw:
            values[key] = normalized.raw[key]
    for raw_key, value in normalized.unmapped.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        values[_unmapped_field(raw_key)] = value
    return values


def _unmapped_field(raw_key):
    # MongoDB 欄位名稱不可包含 "." 或以 "$" 開頭。
    safe_key = normalize_key_text(raw_key).replace(".", "_").replace("$", "_")
    return f"{UNMAPPED_KEY_PREFIX}{safe_key}"


def build_entries(screens, use_corrections):
    """將 screen_analysis 文件整理成 {(秒數, 設備): [各台設備的比對項目]}。

    同組內位置重疊（或缺少 bbox）的結果視為同一台設備，只保留最佳的一筆；
    清單依最佳程度排序。
    use_corrections=True 用於標準答案（套用人工訂正）；
    False 用於待測結果（只看模型原始輸出）。
    回傳 (groups, duplicate_count)。
    """
    grouped = {}
    for screen in screens:
        if not screen.get("success"):
            continue
        detected_model = screen.get("detected_model")
        corrected_model = screen.get("corrected_model") if use_corrections else None
        family, device_key = classify_device(detected_model, corrected_model)
        if family == UNKNOWN_FAMILY:
            continue

        values = _extract_values(
            screen.get("medical_values") or {},
            screen.get("corrected_medical_values") if use_corrections else None,
            family,
        )
        device_name = (
            corrected_model if isinstance(corrected_model, str) and corrected_model.strip()
            else detected_model
        )
        candidate = {
            "minute": screen.get("minute"),
            "device_key": device_key,
            "device_name": device_name or device_key,
            "values": values,
            "screen_id": str(screen.get("_id", "")),
            "bbox": valid_bbox(screen.get("bbox")),
            "_rank": (
                bool(use_corrections and (
                    screen.get("is_corrected") or screen.get("is_model_corrected")
                )),
                len(values),
                _timestamp_value(screen.get("analyzed_at")),
                str(screen.get("_id", "")),
            ),
        }
        grouped.setdefault((candidate["minute"], device_key), []).append(candidate)

    groups = {}
    duplicate_count = 0
    for key, candidates in grouped.items():
        kept = []
        for candidate in sorted(candidates, key=lambda item: item["_rank"], reverse=True):
            if any(same_screen(candidate["bbox"], other["bbox"]) for other in kept):
                duplicate_count += 1
                continue
            kept.append(candidate)
        for entry in kept:
            entry.pop("_rank")
        groups[key] = kept
    return groups, duplicate_count


def _sort_key(key):
    return (key[0] if key[0] is not None else -1, key[1])


def count_entries(groups):
    return sum(len(entries) for entries in groups.values())


def serialize_entries(groups):
    """轉成可存入 MongoDB 的清單（tuple 鍵無法直接存）。"""
    return [entry for key in sorted(groups, key=_sort_key) for entry in groups[key]]


def deserialize_entries(entry_list):
    groups = {}
    for entry in entry_list or []:
        groups.setdefault((entry.get("minute"), entry.get("device_key")), []).append(entry)
    return groups


def _pair_group(reference, candidate):
    """同一（秒數, 設備）內的一對一配對，回傳 [(reference_index, candidate_index)]。"""
    if len(reference) == 1 and len(candidate) == 1:
        return [(0, 0)]
    has_all_boxes = all(entry.get("bbox") for entry in reference + candidate)
    if not has_all_boxes:
        return [(0, 0)] if reference and candidate else []
    scored = sorted(
        (
            (iou(ref["bbox"], cand["bbox"]), i, j)
            for i, ref in enumerate(reference)
            for j, cand in enumerate(candidate)
            if same_screen(ref["bbox"], cand["bbox"])
        ),
        reverse=True,
    )
    pairs, used_ref, used_cand = [], set(), set()
    for _, i, j in scored:
        if i in used_ref or j in used_cand:
            continue
        pairs.append((i, j))
        used_ref.add(i)
        used_cand.add(j)
    return pairs


def pair_entries(reference_groups, candidate_groups):
    """配對標準答案與待測結果，回傳兩個以（秒數, 設備, 序號）為鍵的扁平字典。

    配對成功的兩邊使用相同鍵；未配對的待測結果使用 "extra-N" 序號。
    """
    reference, candidate = {}, {}
    for key in set(reference_groups) | set(candidate_groups):
        ref_list = reference_groups.get(key, [])
        cand_list = candidate_groups.get(key, [])
        for index, entry in enumerate(ref_list):
            reference[(*key, index)] = entry
        matched = set()
        for ref_index, cand_index in _pair_group(ref_list, cand_list):
            candidate[(*key, ref_index)] = cand_list[cand_index]
            matched.add(cand_index)
        for cand_index, entry in enumerate(cand_list):
            if cand_index not in matched:
                candidate[(*key, f"extra-{cand_index}")] = entry
    return reference, candidate


def _comparable_text(value):
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    else:
        text = str(value)
    return "".join(text.split()).lower()


def values_match(expected, actual):
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not (math.isfinite(expected) and math.isfinite(actual)):
            return False
        return abs(float(expected) - float(actual)) <= 1e-6
    return _comparable_text(expected) == _comparable_text(actual)


def _display_value(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _ratio(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def compare_entries(reference, candidate):
    """以標準答案為準逐欄比對，回傳統計結果。"""
    by_field = {}
    by_device = {}
    mismatches = []
    extra_devices = []
    totals = {
        "device_total": len(reference),
        "device_matched": 0,
        "device_missed": 0,
        "device_extra": 0,
        "field_total": 0,
        "field_correct": 0,
        "field_wrong": 0,
        "field_missing": 0,
    }

    def add_mismatch(row):
        if len(mismatches) < MAX_DETAIL_ROWS:
            mismatches.append(row)

    for key in sorted(reference, key=lambda k: (k[0] if k[0] is not None else -1, k[1])):
        expected_entry = reference[key]
        actual_entry = candidate.get(key)
        device_key = expected_entry["device_key"]
        device_stats = by_device.setdefault(device_key, {
            "device_name": expected_entry.get("device_name") or device_key,
            "device_total": 0,
            "device_matched": 0,
            "field_total": 0,
            "field_correct": 0,
        })
        device_stats["device_total"] += 1
        if actual_entry is None:
            totals["device_missed"] += 1
        else:
            totals["device_matched"] += 1
            device_stats["device_matched"] += 1

        actual_values = (actual_entry or {}).get("values", {})
        for field, expected in expected_entry.get("values", {}).items():
            field_stats = by_field.setdefault(field, {
                "total": 0, "correct": 0, "wrong": 0, "missing": 0,
            })
            field_stats["total"] += 1
            totals["field_total"] += 1
            device_stats["field_total"] += 1

            if actual_entry is not None and field in actual_values:
                actual = actual_values[field]
                if values_match(expected, actual):
                    status = "correct"
                else:
                    status = "wrong"
            else:
                actual = None
                status = "missing"

            field_stats[status] += 1
            totals[f"field_{status}"] += 1
            if status == "correct":
                device_stats["field_correct"] += 1
            else:
                add_mismatch({
                    "minute": expected_entry.get("minute"),
                    "device_key": device_key,
                    "device_name": expected_entry.get("device_name"),
                    "field": field,
                    "expected": _display_value(expected),
                    "actual": _display_value(actual),
                    "status": status if actual_entry is not None else "device_missed",
                })

    for key in sorted(candidate, key=lambda k: (k[0] if k[0] is not None else -1, k[1])):
        if key in reference:
            continue
        totals["device_extra"] += 1
        if len(extra_devices) < MAX_DETAIL_ROWS:
            entry = candidate[key]
            extra_devices.append({
                "minute": entry.get("minute"),
                "device_key": entry.get("device_key"),
                "device_name": entry.get("device_name"),
            })

    for stats in by_field.values():
        stats["accuracy"] = _ratio(stats["correct"], stats["total"])
    for stats in by_device.values():
        stats["accuracy"] = _ratio(stats["field_correct"], stats["field_total"])

    return {
        **totals,
        "accuracy": _ratio(totals["field_correct"], totals["field_total"]),
        "matched_accuracy": _ratio(
            totals["field_correct"], totals["field_correct"] + totals["field_wrong"]
        ),
        "device_detection_rate": _ratio(totals["device_matched"], totals["device_total"]),
        "by_field": by_field,
        "by_device": by_device,
        "mismatches": mismatches,
        "mismatches_truncated": (
            totals["field_wrong"] + totals["field_missing"] > len(mismatches)
        ),
        "extra_devices": extra_devices,
    }


def _raw_key_fields(raw_key, family, values, registry):
    """原始欄位名稱對應到哪些標準欄位（血壓等複合欄位會拆成多個）。"""
    if not isinstance(raw_key, str) or registry.is_ignored(raw_key):
        return []
    canon = registry.normalize_key(family, raw_key)
    if canon is None:
        return [_unmapped_field(raw_key)]
    if registry.is_composite(canon):
        parts = [part for part in registry.composites[canon].parts if part in values]
        return parts or [_unmapped_field(raw_key)]
    return [canon]


def _is_empty_value(value):
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in ("", "null", "none")


def _format_expected(values, composite=False):
    texts = [str(_display_value(value)) for value in values]
    if composite and len(texts) == 3:
        return f"{texts[0]}/{texts[1]} ({texts[2]})"
    return ("/" if composite else " / ").join(texts)


def annotate_screens(screens, reference, candidate_entries):
    """逐一標記待測 Session 每個螢幕的每個原始欄位是否與標準答案一致。

    reference 與 candidate_entries 為 pair_entries 的輸出。
    回傳 {screen_id: {status, device_key, fields, missing_fields}}，
    fields 以原始欄位名稱為鍵，供前端在數值旁直接標示。
    """
    registry = get_registry()
    missed_by_minute = {}
    for key, entry in reference.items():
        if key not in candidate_entries:
            missed_by_minute.setdefault(key[0], []).append(entry)

    def expected_devices_at(screen):
        """此螢幕位置上，標準答案有但待測結果漏掉的設備名稱。"""
        bbox = valid_bbox(screen.get("bbox"))
        return [
            entry.get("device_name") or entry.get("device_key")
            for entry in missed_by_minute.get(screen.get("minute"), [])
            if same_screen(bbox, entry.get("bbox"))
        ]

    key_by_screen = {
        entry.get("screen_id"): key for key, entry in candidate_entries.items()
    }
    annotations = {}
    for screen in screens:
        screen_id = str(screen.get("_id", ""))
        if not screen_id:
            continue
        if not screen.get("success"):
            annotations[screen_id] = {"status": "failed"}
            continue
        family, device_key = classify_device(screen.get("detected_model"), None)
        if family == UNKNOWN_FAMILY:
            expected_devices = expected_devices_at(screen)
            if expected_devices:
                annotations[screen_id] = {
                    "status": "unknown_device",
                    "expected_devices": expected_devices,
                }
            else:
                # 看不清楚或被遮擋時回傳 unknown / null，標準答案此處也沒有可辨識設備即視為正確。
                annotations[screen_id] = {
                    "status": "unknown_ok",
                    "fields": {
                        raw_key: {"status": "correct" if _is_empty_value(value) else "not_in_ground_truth"}
                        for raw_key, value in (screen.get("medical_values") or {}).items()
                    },
                }
            continue

        key = key_by_screen.get(screen_id)
        if key is None:
            annotations[screen_id] = {"status": "duplicate", "device_key": device_key}
            continue
        candidate_entry = candidate_entries[key]
        reference_entry = reference.get(key)
        if reference_entry is None:
            annotations[screen_id] = {
                "status": "extra_device",
                "device_key": device_key,
                "expected_devices": expected_devices_at(screen),
            }
            continue

        expected_values = reference_entry.get("values", {})
        actual_values = candidate_entry.get("values", {})
        covered = set()
        fields = {}
        for raw_key, raw_value in (screen.get("medical_values") or {}).items():
            canon_fields = _raw_key_fields(raw_key, family, actual_values, registry)
            if not canon_fields:
                continue
            covered.update(canon_fields)
            compared = [field for field in canon_fields if field in expected_values]
            if not compared:
                fields[raw_key] = {
                    "status": "correct" if _is_empty_value(raw_value) else "not_in_ground_truth"
                }
                continue
            all_match = all(
                field in actual_values
                and values_match(expected_values[field], actual_values[field])
                for field in compared
            )
            fields[raw_key] = {
                "status": "correct" if all_match else "wrong",
                "expected": _format_expected(
                    [expected_values[field] for field in compared],
                    composite=len(canon_fields) > 1,
                ),
            }

        missing_fields = [
            {"field": field, "expected": _display_value(value)}
            for field, value in expected_values.items()
            if field not in covered
        ]
        annotations[screen_id] = {
            "status": "compared",
            "device_key": device_key,
            "expected_device_name": reference_entry.get("device_name") or device_key,
            "fields": fields,
            "missing_fields": missing_fields,
        }
    return annotations


SUMMED_FIELDS = (
    "device_total", "device_matched", "device_missed", "device_extra",
    "field_total", "field_correct", "field_wrong", "field_missing",
)


def aggregate_evaluations(evaluations):
    """將多部影片的比對結果加總為整體正確率。"""
    totals = {name: 0 for name in SUMMED_FIELDS}
    by_field = {}
    for evaluation in evaluations:
        for name in SUMMED_FIELDS:
            totals[name] += int(evaluation.get(name) or 0)
        for field, stats in (evaluation.get("by_field") or {}).items():
            merged = by_field.setdefault(field, {
                "total": 0, "correct": 0, "wrong": 0, "missing": 0,
            })
            for name in ("total", "correct", "wrong", "missing"):
                merged[name] += int(stats.get(name) or 0)
    for stats in by_field.values():
        stats["accuracy"] = _ratio(stats["correct"], stats["total"])
    return {
        **totals,
        "evaluated_videos": len(evaluations),
        "accuracy": _ratio(totals["field_correct"], totals["field_total"]),
        "matched_accuracy": _ratio(
            totals["field_correct"], totals["field_correct"] + totals["field_wrong"]
        ),
        "device_detection_rate": _ratio(totals["device_matched"], totals["device_total"]),
        "by_field": by_field,
    }
