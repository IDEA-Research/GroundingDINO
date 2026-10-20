"""
本機微調螢幕 OCR 模型的推論服務（OpenAI 相容 API）。

提供三個模型 ID，供 web_app 的 provider="ondevice" 使用：
- ondevice-glm46v-fix4-vote : GLM-4.6V-Flash + fix4 三個 seed adapter，逐欄位多數決（最準，最慢）
- ondevice-glm46v-r9        : GLM-4.6V-Flash + r9 單一 adapter
- ondevice-paddleocr-vl     : PaddleOCR-VL + 單一 adapter（最快）

推論設定必須與訓練一致，所以伺服器一律使用 prompt_with_rules.txt，忽略用戶端送來的文字提示；
回傳的 content 是 {"device": ..., "values": {...}, "per_adapter": [...]} 的 JSON 字串。

啟動：
    python ocr_server.py --host 127.0.0.1 --port 18765 [--preload glm paddle]
"""
import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import uuid

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(HERE, "screen-ocr-models-2026-10-12")
R9_DIR = os.path.join(HERE, "glm46v-flash-screen-ocr-r9")
sys.path.insert(0, PKG_DIR)
from common import (disable_peft_moe_conversion, input_device, load_image,  # noqa: E402
                    load_model_and_processor, parse_output, prompt_text)
from predict import vote  # noqa: E402

MIN_PIXELS = 800_000
MAX_PIXELS = 1280 * 28 * 28  # 1,003,520
MAX_SIDE = 2048
MAX_NEW_TOKENS = 256

BASES = {
    "glm": os.environ.get("ONDEVICE_GLM_PATH", os.path.join(HERE, "base_models", "GLM-4.6V-Flash")),
    "paddle": os.environ.get("ONDEVICE_PADDLE_PATH", os.path.join(HERE, "base_models", "PaddleOCR-VL")),
}

# adapter 名稱 -> (base, 路徑)
ADAPTERS = {
    "fix4_s42": ("glm", os.path.join(PKG_DIR, "glm46v-flash-fix4", "adapter_s42")),
    "fix4_s1": ("glm", os.path.join(PKG_DIR, "glm46v-flash-fix4", "adapter_s1")),
    "fix4_s7": ("glm", os.path.join(PKG_DIR, "glm46v-flash-fix4", "adapter_s7")),
    "r9": ("glm", os.path.join(R9_DIR, "adapter")),
    "paddle": ("paddle", os.path.join(PKG_DIR, "paddleocr-vl", "adapter")),
}

# 對外模型 ID -> (base, 依序要跑的 adapter)
MODELS = {
    "ondevice-glm46v-fix4-vote": ("glm", ["fix4_s42", "fix4_s1", "fix4_s7"]),
    "ondevice-glm46v-r9": ("glm", ["r9"]),
    "ondevice-paddleocr-vl": ("paddle", ["paddle"]),
}

with open(os.path.join(PKG_DIR, "prompt_with_rules.txt"), encoding="utf-8") as f:
    PROMPT = f.read().strip()


class BaseRunner:
    """一個基底模型 + 掛在上面的所有 adapter；GPU 一次只跑一個請求。"""

    def __init__(self, base):
        self.base = base
        self.model = None
        self.processor = None
        self.text = None
        self.load_lock = threading.Lock()
        self.gen_lock = threading.Lock()

    def ensure_loaded(self):
        with self.load_lock:
            if self.model is not None:
                return
            path = BASES[self.base]
            if not os.path.isdir(path):
                raise RuntimeError(f"找不到基底模型目錄: {path}")
            t0 = time.time()
            print(f"[ocr_server] 載入基底模型 {self.base}: {path}", flush=True)
            model, processor = load_model_and_processor(path, MAX_PIXELS)
            from peft import PeftModel
            disable_peft_moe_conversion()
            names = [n for n, (b, _) in ADAPTERS.items() if b == self.base]
            model = PeftModel.from_pretrained(model, ADAPTERS[names[0]][1], adapter_name=names[0])
            for n in names[1:]:
                model.load_adapter(ADAPTERS[n][1], adapter_name=n)
            model.eval()
            processor.tokenizer.padding_side = "left"
            self.text = prompt_text(processor, PROMPT)
            self.processor = processor
            self.model = model
            print(f"[ocr_server] {self.base} 載入完成（adapters={names}，{time.time() - t0:.1f}s）", flush=True)

    def run(self, image, adapter_names):
        self.ensure_loaded()
        preds, prompt_tokens, completion_tokens = [], 0, 0
        with self.gen_lock:
            enc = self.processor(text=[self.text], images=[image], padding=True,
                                 return_tensors="pt").to(input_device(self.model))
            for name in adapter_names:
                self.model.set_adapter(name)
                with torch.no_grad():
                    # PaddleOCR-VL 的 generation_config 預設 use_cache=false，解碼會慢 7 倍以上
                    out = self.model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                              use_cache=True)
                gen = out[:, enc["input_ids"].shape[1]:]
                text = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
                preds.append(parse_output(text))
                prompt_tokens += int(enc["input_ids"].shape[1])
                completion_tokens += int(gen.shape[1])
        return preds, prompt_tokens, completion_tokens


RUNNERS = {base: BaseRunner(base) for base in BASES}
app = FastAPI(title="On-device screen OCR")


def _decode_image(messages):
    """從 OpenAI 格式的 messages 取出第一張 data URI 圖片。"""
    for msg in reversed(messages or []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "image_url":
                continue
            url = (part.get("image_url") or {}).get("url", "")
            if not url.startswith("data:") or "," not in url:
                raise HTTPException(400, "image_url 必須是 base64 data URI")
            raw = base64.b64decode(url.split(",", 1)[1])
            return load_image(io.BytesIO(raw), MAX_SIDE, MIN_PIXELS)
    raise HTTPException(400, "請求中沒有圖片")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "loaded": {b: r.model is not None for b, r in RUNNERS.items()},
        "models": list(MODELS),
    }


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "ondevice"} for m in MODELS]}


@app.post("/v1/chat/completions")
def chat_completions(body: dict):
    model_id = body.get("model")
    if model_id not in MODELS:
        raise HTTPException(400, f"不支援的模型: {model_id}（可用: {', '.join(MODELS)}）")
    image = _decode_image(body.get("messages"))
    base, adapter_names = MODELS[model_id]

    t0 = time.time()
    try:
        preds, p_tok, c_tok = RUNNERS[base].run(image, adapter_names)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    pred = preds[0] if len(preds) == 1 else vote(preds)
    if pred is None:
        pred = {"device": "unknown", "values": {}}
    pred = {**pred, "per_adapter": preds}
    elapsed = time.time() - t0
    print(f"[ocr_server] {model_id} {elapsed:.2f}s -> {json.dumps(pred, ensure_ascii=False)}", flush=True)

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": json.dumps(pred, ensure_ascii=False)},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": p_tok + c_tok},
        "elapsed_seconds": round(elapsed, 3),
    })


def main():
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("ONDEVICE_OCR_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("ONDEVICE_OCR_PORT", "18765")))
    p.add_argument("--preload", nargs="*", default=[], choices=list(BASES),
                   help="啟動時先載入的基底模型（預設在第一次請求時才載入）")
    args = p.parse_args()
    for base in args.preload:
        RUNNERS[base].ensure_loaded()
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
