"""Uygulama yapilandirmasi: .env'den secret okuma + dogrulama."""
from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(
            f"Ortam degiskeni eksik: {name}. "
            f"Lutfen .env dosyasi olusturun (ornek: .env.example)."
        )
    return (value or "").strip()


TELEGRAM_BOT_TOKEN: str = _get_env("TELEGRAM_BOT_TOKEN", required=True)
GEMINI_API_KEY: str = _get_env("GEMINI_API_KEY", required=True)
GEMINI_MODEL: str = _get_env("GEMINI_MODEL", default="gemini-2.5-flash") or "gemini-2.5-flash"

# --- NVIDIA NIM yedek saglayici (opsiyonel) ---
NVIDIA_API_KEY: str = _get_env("NVIDIA_API_KEY", default="") or ""
NVIDIA_MODEL: str = (
    _get_env("NVIDIA_MODEL", default="meta/llama-3.2-11b-vision-instruct")
    or "meta/llama-3.2-11b-vision-instruct"
)
NVIDIA_TIMEOUT_S = int(_get_env("NVIDIA_TIMEOUT_S", default="120") or "120")
# NVIDIA'ya gonderilen gorsel kucultulur (hiz icin; Gemini yolu etkilenmez)
NVIDIA_IMAGE_MAX_LONG_EDGE = int(_get_env("NVIDIA_IMAGE_MAX_LONG_EDGE", default="1280") or "1280")
NVIDIA_IMAGE_JPEG_QUALITY = int(_get_env("NVIDIA_IMAGE_JPEG_QUALITY", default="80") or "80")

# NVIDIA yedegi aktif mi? AI_PROVIDER_ORDER icinde "nvidia" gecmeli ve key dolu olmali.
# Ornek: AI_PROVIDER_ORDER=gemini,nvidia
def _get_provider_order() -> list[str]:
    raw = (_get_env("AI_PROVIDER_ORDER", default="gemini,nvidia") or "gemini,nvidia").lower()
    order = [p.strip() for p in raw.split(",") if p.strip() in ("gemini", "nvidia")]
    return order or ["gemini"]


AI_PROVIDER_ORDER: list[str] = _get_provider_order()

# Model basarisi ogrenilir: hangisi son calistiysa bir dahaki sefere ilk o denenir.
# Kayit: BASE_DIR/.model_priority.json (temp disinda, silinmez)
def _priority_file() -> str:
    return os.path.join(BASE_DIR, ".model_priority.json")


def _load_priority() -> list[str]:
    try:
        with open(_priority_file(), "r", encoding="utf-8") as f:
            data = json.load(f)
        order = [m for m in data.get("gemini_order", []) if isinstance(m, str)]
        if order:
            return order
    except Exception:
        pass
    return list(GEMINI_FALLBACK_MODELS)


def _save_priority_first(model: str) -> None:
    try:
        current = _load_priority()
        order = [model] + [m for m in current if m != model]
        # config'te bilinmeyen model kalmasin diye bilinenleri ekle
        for m in [GEMINI_MODEL, *GEMINI_FALLBACK_MODELS]:
            if m not in order:
                order.append(m)
        with open(_priority_file(), "w", encoding="utf-8") as f:
            json.dump({"gemini_order": order}, f)
    except Exception:
        logger.warning("Model onceligi kaydedilemedi.", exc_info=True)


def _gemini_try_order() -> list[str]:
    prio = _load_priority()
    base = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]
    ordered = [m for m in prio if m in base] + [m for m in base if m not in prio]
    return ordered or base

# Birincil model + kota dostu yedek havuzu.
# Tier1 oranlariniza gore secildi: birincil dar bogaz (RPD 10K), yedekler genis.
# .env'de GEMINI_FALLBACK_MODELS virgulle ayrilmis liste olarak ezilebilir.
# Varsayilan sira: 2.5-flash (RPD 10K) -> 3.5-flash-lite (RPD 150K) -> 3.1-flash-lite (RPD 150K)
def _get_fallbacks() -> list[str]:
    raw = _get_env("GEMINI_FALLBACK_MODELS", default="") or ""
    if raw.strip():
        return [m.strip() for m in raw.split(",") if m.strip()]
    return ["gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]


GEMINI_FALLBACK_MODELS: list[str] = _get_fallbacks()

# Klasorler (BASE_DIR dosya basinda tanimli)
TEMP_DIR = os.path.join(BASE_DIR, "temp")

# AI / HTTP ayarlari
GEMINI_TIMEOUT_S = 30
GEMINI_MAX_RETRIES = 2
GEMINI_TEMPERATURE = 0.0

# Album (media group) toplama bekleme suresi (sn)
ALBUM_COLLECT_DELAY_S = 2.0

# Gorsel isleme
IMAGE_MAX_LONG_EDGE = 2000
IMAGE_JPEG_QUALITY = 85
