from __future__ import annotations

import math
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

# Optional dependency. The YouTube branch is a side feature; the rest
# of the orchestrator must keep working when the package is missing
# (CI, environments that don't need it). The actual import error is
# only raised when `build_outline_from_youtube_url` is called.
try:
    from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised via missing dep
    YouTubeTranscriptApi = None  # type: ignore[assignment]


YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


@dataclass
class YoutubeOutlineResult:
    video_id: str
    language: str
    duration_sec: float
    outline_text: str


def extract_youtube_url(text: str) -> str | None:
    m = re.search(r"https?://\S+", text)
    if not m:
        return None
    url = m.group(0).rstrip(").,]}>\"")
    p = urlparse(url)
    host = (p.netloc or "").lower()
    if host not in YOUTUBE_HOSTS:
        return None
    return url


def parse_video_id(url: str) -> str | None:
    p = urlparse(url)
    host = (p.netloc or "").lower()
    vid = None
    if host == "youtu.be":
        vid = p.path.strip("/")
    else:
        qs = parse_qs(p.query)
        vid = (qs.get("v") or [None])[0]
        if not vid and p.path.startswith("/shorts/"):
            vid = p.path.split("/")[2] if len(p.path.split("/")) > 2 else None
    if not vid or not YT_ID_RE.match(vid):
        return None
    return vid


def build_outline_from_youtube_url(url: str) -> YoutubeOutlineResult:
    if YouTubeTranscriptApi is None:
        raise RuntimeError(
            "youtube-transcript-api is not installed; install the optional "
            "dependency (`pip install youtube-transcript-api`) to use the "
            "YouTube outline feature"
        )

    video_id = parse_video_id(url)
    if not video_id:
        raise ValueError("invalid youtube url")

    preferred = ["zh-TW", "zh-Hant", "zh", "en"]
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=preferred)
    lang = str(transcript[0].get("language_code") or "unknown") if transcript else "unknown"

    duration = 0.0
    chunks: list[str] = []
    for seg in transcript:
        st = float(seg.get("start") or 0.0)
        du = float(seg.get("duration") or 0.0)
        duration = max(duration, st + du)
        tx = str(seg.get("text") or "").replace("\n", " ").strip()
        if tx:
            chunks.append(tx)
    clean_text = re.sub(r"\s+", " ", " ".join(chunks)).strip()
    if not clean_text:
        raise ValueError("no usable transcript")

    slide_count = max(4, min(16, math.ceil(duration / 120)))
    sentences = re.split(r"(?<=[。！？.!?])\s+", clean_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        sentences = [clean_text]

    bucket = max(1, math.ceil(len(sentences) / slide_count))
    lines: list[str] = ["影片大綱（純文字）", f"語言: {lang}", f"預估段數: {slide_count}", ""]
    for i in range(slide_count):
        part = sentences[i * bucket : (i + 1) * bucket]
        if not part:
            break
        title = part[0][:36]
        lines.append(f"{i+1}. {title}")
        for b in part[:3]:
            lines.append(f"- {b[:120]}")
        lines.append("")

    return YoutubeOutlineResult(
        video_id=video_id,
        language=lang,
        duration_sec=duration,
        outline_text="\n".join(lines).strip(),
    )

