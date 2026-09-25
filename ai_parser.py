"""Gemini Vision ile sayac formu okuma (structured output + timeout + retry)."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from typing import Any

from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field

import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Sen uzman bir veri sayisallastirma asistanisin. Gonderilen gorseldeki "
    "(el yazisi veya matbu) sayac listesini milimetrik cetvel hizasina dikkat ederek oku "
    "ve asagidaki is kurallarina gore yapilandirilmis JSON verisi uret. Gorseldeki sag sutun "
    "(51-100), sol sutundaki dairelerin not alani olarak kullanilmis olabilir, dikkat et.\n"
    "Kritik Is Kurallari:\n"
    "1. Durum Belirleme - Tamamlandi: Bir satirda 8 haneli yepyeni bir seri numarasi (barkod) varsa, "
    "o dairenin durumu KESINLIKLE 'Tamamlandi'dir. (Yaninda 'Evde yok', 'Sok tak' vs. yazsa bile "
    "seri no varsa iptal edilmez).\n"
    "2. Durum Belirleme - faruk: Seri numarasi YOKSA ve dairenin karsisina sadece 'f' harfi veya "
    "isareti konulmusssa, Durum sutununa 'Iptal' yerine 'faruk' yaz.\n"
    "3. Durum Belirleme - Iptal: Seri numarasi yoksa ve 'Evde yok', 'Bos', 'Dolap kesilecek' gibi net "
    "olumsuz notlar varsa Durum sutununa 'Iptal' yaz. Bos satirlari atla.\n"
    "4. Sayac Turu: Yeni seri numarasi '30' ile basliyorsa 'ULTRASONIK', farkliysa ('80', '23' vb.) "
    "'SICAK SU' yaz.\n"
    "5. Eski Endeks Sifirlama: 'EEY', 'E.Y.', 'Ek yok' veya 'Sifir' notlari varsa Eski Endeks'i sayisal "
    "olarak 0 yap.\n"
    "6. Seri Numarasi Tamamlama: '3050' yazilmayip '// 2681' veya '26A' gibi sadece son haneler yazilarak "
    "kisaltma yapilmissa, ustteki seri bloklarina bakarak bunu mantiksal olarak tam 8 haneli barkoda "
    "(orn: 30502681) donustur.\n"
    "7. COK FOTOGRAFLI BIRLESTIRME: Sana birden fazla fotograf gonderildiyse bunlar AYNI listenin "
    "devami/parcasi olabilir. Tum fotograflardaki satirlari TEK 'data' listesinde birlestir. "
    "Ayni DAIRE numarasi birden fazla fotoda geciyorsa: seri numarali (dolu) olani tut, bos olani at; "
    "ikisi de doluysa en net/uzun seri numarali olani tut. Sonucu DAIRE numarasina gore kucukten buyuge "
    "sirala. Ayni daireyi iki kez yazma.\n"
    "\nBeklenen JSON Cikti Formati:\n"
    '{"data": [{"BINA": "", "DAIRE": "1", "MARKA": "CALMET", "SAYAC TURU": "ULTRASONIK", '
    '"DURUM": "Tamamlandi", "TUTAR (TL)": "", "ESKI ENDEKS": 0, '
    '"YENI SERI NO (BARKOD)": "30502681", "NOTLAR": "EEY", "ENLEM": "", "BOYLAM": ""}]}'
)

# NOTE: Model ciktisinda asagidaki Turkce basliklar beklenir:
# BINA, DAIRE, MARKA, SAYAC TURU, DURUM, TUTAR (TL), ESKI ENDEKS,
# YENI SERI NO (BARKOD), NOTLAR, ENLEM, BOYLAM

def resize_image_for_api(src_path: str) -> bytes:
    """Gorseli uzun kenar max 2000px olacak sekilde resize edip JPEG bytes doner."""
    with Image.open(src_path) as img:
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        w, h = img.size
        long_edge = max(w, h)
        if long_edge > config.IMAGE_MAX_LONG_EDGE:
            scale = config.IMAGE_MAX_LONG_EDGE / float(long_edge)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=config.IMAGE_JPEG_QUALITY, optimize=True)
        return buf.getvalue()


def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


def _get_client() -> genai.Client:
    return genai.Client(api_key=config.GEMINI_API_KEY)


def _parse_response(resp: Any) -> dict:
    text = getattr(resp, "text", "") or ""
    if text.strip():
        return json.loads(text)
    parsed = getattr(resp, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, BaseModel):
            return parsed.model_dump(by_alias=True)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Model bos yanit dondu.")


def _call_gemini_with_config(image_bytes_list: list[bytes], model: str) -> dict:
    client = _get_client()
    parts: list[types.Part] = [
        types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in image_bytes_list
    ]
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=MeterResult,
        temperature=config.GEMINI_TEMPERATURE,
    )
    resp = client.models.generate_content(
        model=model,
        contents=[types.Content(
            role="user",
            parts=[*parts, types.Part.from_text(text=SYSTEM_PROMPT)],
        )],
        config=cfg,
    )
    return _parse_response(resp)


def _is_quota_or_not_found(exc: Exception) -> bool:
    msg = str(exc)
    return "404" in msg or "NOT_FOUND" in msg or "429" in msg or "RESOURCE_EXHAUSTED" in msg


# --- 429 sogutma (circuit breaker): kota yiyen model bir sure atlanir ---
import time as _time

_quota_cooldown_until: dict[str, float] = {}
QUOTA_COOLDOWN_S = 120.0


def _note_quota_miss(model: str) -> None:
    _quota_cooldown_until[model] = _time.monotonic() + QUOTA_COOLDOWN_S
    logger.info("%s 429/404 yedi, %.0f sn sogutmaya alindi.", model, QUOTA_COOLDOWN_S)


def _is_cooled_down(model: str) -> bool:
    return _time.monotonic() < _quota_cooldown_until.get(model, 0.0)


NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def _extract_json_object(text: str) -> dict:
    """Model ciktisindaki ilk gecerli {"data": ...} JSON objesini bulup parse eder.

    Llama-Vision aciklama + ```json fence ekleyebilir, ayrica stream
    birlesiminde birden fazla JSON blogu olabilir (bu durumda
    json.loads "Extra data" hatasi verirdi). JSONDecoder.raw_decode ile
    metni tarayip icinde "data" anahtari olan ilk objeyi secer.
    """
    cleaned = text.strip()
    decoder = json.JSONDecoder()
    # 1) fence iclerini once dene
    candidates: list[str] = []
    if "```" in cleaned:
        for part in cleaned.split("```"):
            cand = part.strip()
            if cand.lower().startswith("json"):
                cand = cand[4:].strip()
            if "{" in cand:
                candidates.append(cand)
    candidates.append(cleaned)
    for cand in candidates:
        idx = 0
        while True:
            start = cand.find("{", idx)
            if start < 0:
                break
            try:
                obj, end = decoder.raw_decode(cand[start:])
                if isinstance(obj, dict) and "data" in obj:
                    return obj
                idx = start + 1  # data yoksa sonraki { dene
            except json.JSONDecodeError:
                idx = start + 1
        # data'li obje bulunamadiysa: bu adaydaki ilk gecerli dict'i kabul et
        idx = 0
        while True:
            start = cand.find("{", idx)
            if start < 0:
                break
            try:
                obj, end = decoder.raw_decode(cand[start:])
                if isinstance(obj, dict):
                    return obj
                idx = start + 1
            except json.JSONDecodeError:
                idx = start + 1
    raise ValueError("NVIDIA ciktisi JSON icermiyor.")


def _shrink_for_nvidia(image_bytes: bytes) -> bytes:
    """NVIDIA'ya gondermeden once gorseli kucult (gecikmeyi dusurur)."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            w, h = img.size
            long_edge = max(w, h)
            limit = config.NVIDIA_IMAGE_MAX_LONG_EDGE
            if long_edge > limit:
                scale = limit / float(long_edge)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=config.NVIDIA_IMAGE_JPEG_QUALITY,
                     optimize=True)
            return buf.getvalue()
    except Exception:
        logger.warning("NVIDIA icin kucultme basarisiz, orijinal gonderiliyor.",
                       exc_info=True)
        return image_bytes


def _call_nvidia(image_bytes_list: list[bytes],
                 on_first_token: Any | None = None) -> dict:
    import requests  # yerel import: sadece nvidia yolu kullanildiginda gerekir

    if not config.NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY bos (.env).")
    shrunk = [_shrink_for_nvidia(b) for b in image_bytes_list]
    total_kb = sum(len(b) for b in shrunk) // 1024
    logger.info("NVIDIA'ya %d gorsel gonderiliyor (~%d KB, model=%s).",
                len(shrunk), total_kb, config.NVIDIA_MODEL)
    content: list[dict] = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT
            + "\n\nSADECE gecerli JSON dondur, baska aciklama yazma. Sema: {\"data\": [...]}",
        }
    ]
    for b in shrunk:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(b)}"},
            }
        )
    # Stream: ilk token gelince on_first_token() cagrilir (kullaniciya "uretiyor" bilgisi),
    # toplam sure NVIDIA_TIMEOUT_S ile sinirlanir. Stream kapalisi kadar guvenilir,
    # ayrica ilk belirti erken gorunur ve sessiz timeout riski azalir.
    resp = requests.post(
        NVIDIA_API_URL,
        headers={"Authorization": f"Bearer {config.NVIDIA_API_KEY}",
                 "Accept": "text/event-stream"},
        json={
            "model": config.NVIDIA_MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 2048,
            "temperature": 0.0,
            "top_p": 0.95,
            "stream": True,
        },
        timeout=(30, config.NVIDIA_TIMEOUT_S),
        stream=True,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"NVIDIA API {resp.status_code}: {resp.text[:500]}")
    chunks: list[str] = []
    first_seen = False
    try:
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            try:
                delta = obj["choices"][0]["delta"].get("content") or ""
            except (KeyError, IndexError, TypeError):
                delta = ""
            if delta:
                chunks.append(delta)
                if not first_seen:
                    first_seen = True
                    if on_first_token is not None:
                        try:
                            on_first_token()
                        except Exception:
                            pass
    finally:
        resp.close()
    text = "".join(chunks)
    if not text.strip():
        raise ValueError("NVIDIA bos yanit dondu.")
    return _extract_json_object(text)


async def _parse_with_gemini(image_bytes_list: list[bytes]) -> list[dict]:
    # Onceki basarili model ilk denenir (ogrenilmis sira).
    models_to_try = config._gemini_try_order()
    logger.info("Gemini deneme sirasi: %s", " -> ".join(models_to_try))
    last_err: Exception | None = None
    for model in models_to_try:
        if _is_cooled_down(model):
            logger.info("%s sogutmada, atlaniyor.", model)
            continue
        for attempt in range(config.GEMINI_MAX_RETRIES + 1):
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(_call_gemini_with_config, image_bytes_list, model),
                    timeout=config.GEMINI_TIMEOUT_S,
                )
                rows = result.get("data", []) if isinstance(result, dict) else []
                if not rows:
                    raise ValueError("Model veri uretemedi (bos liste).")
                validated = MeterResult.model_validate(result)
                normalized = validated.model_dump(by_alias=True)
                logger.info(
                    "Gemini %d satir dondu (model=%s).", len(normalized.get("data", [])), model
                )
                await asyncio.to_thread(config._save_priority_first, model)
                return normalized["data"]
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                logger.warning(
                    "Gemini cagrisi basarisiz (model=%s, deneme %d): %s | %s",
                    model, attempt + 1, type(exc).__name__, repr(exc),
                    exc_info=True,
                )
                if _is_quota_or_not_found(exc):
                    _note_quota_miss(model)  # kota yiyeni sogut (circuit breaker)
                    break  # siradaki modele gec
                if attempt < config.GEMINI_MAX_RETRIES:
                    await asyncio.sleep(2 ** (attempt + 1))
    raise (last_err or RuntimeError("Gemini bilinmeyen hata"))


async def _parse_with_nvidia(image_bytes_list: list[bytes],
                           on_progress: Any | None = None) -> list[dict]:
    last_err: Exception | None = None
    for attempt in range(2):  # 1 deneme + 1 retry
        try:
            loop = asyncio.get_running_loop()
            if on_progress is not None:
                def _cb() -> None:
                    loop.call_soon_threadsafe(
                        lambda: asyncio.ensure_future(on_progress()))
            else:
                _cb = None
            result = await asyncio.wait_for(
                asyncio.to_thread(_call_nvidia, image_bytes_list, _cb),
                timeout=config.NVIDIA_TIMEOUT_S + 30,
            )
            rows = result.get("data", []) if isinstance(result, dict) else []
            if not rows:
                raise ValueError("Model veri uretemedi (bos liste).")
            validated = MeterResult.model_validate(result)
            normalized = validated.model_dump(by_alias=True)
            logger.info("NVIDIA %d satir dondu (model=%s).", len(normalized.get("data", [])),
                        config.NVIDIA_MODEL)
            return normalized["data"]
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("NVIDIA cagrisi basarisiz (deneme %d): %s | %s",
                           attempt + 1, type(exc).__name__, repr(exc), exc_info=True)
            if "Extra data" in str(exc) or "JSON" in str(type(exc).__name__):
                break  # model YANIT VERDI ama parse edilemedi; retry faydasiz
            if attempt == 0:
                await asyncio.sleep(2)
    raise (last_err or RuntimeError("NVIDIA bilinmeyen hata"))


async def parse_meter_photos(image_paths: list[str], provider: str | None = None,
                           on_progress: Any | None = None) -> list[dict]:
    """Foto listesini okuyup satir listesi doner.

    provider verilirse ("gemini"/"nvidia") sadece o kullanilir;
    verilmezse AI_PROVIDER_ORDER sirasinda failover yapilir.
    Basarisizlikta ValueError yukseltir.
    """
    if not image_paths:
        raise ValueError("Islenecek fotograf bulunamadi.")
    image_bytes_list = [await asyncio.to_thread(resize_image_for_api, p) for p in image_paths]
    total_kb = sum(len(b) for b in image_bytes_list) // 1024
    logger.info("AI'ye %d gorsel gonderiliyor (~%d KB) [sira=%s].",
                len(image_bytes_list), total_kb, "+".join(config.AI_PROVIDER_ORDER))
    last_err: Exception | None = None
    order = [provider] if provider in ("gemini", "nvidia") else list(config.AI_PROVIDER_ORDER)
    for prov in order:
        try:
            if prov == "nvidia":
                if not config.NVIDIA_API_KEY:
                    logger.warning("NVIDIA atlandi: NVIDIA_API_KEY bos.")
                    continue
                return await _parse_with_nvidia(image_bytes_list, on_progress=on_progress)
            return await _parse_with_gemini(image_bytes_list)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("%s saglayicisi basarisiz, siradakine geciliyor: %s | %s",
                           prov, type(exc).__name__, repr(exc), exc_info=True)
    logger.error("Tum AI saglayicilari basarisiz.", exc_info=True)
    raise ValueError(
        "Fotoğraf okunamadı, lütfen daha net bir fotoğraf gönderin."
    ) from last_err

# (Orjinal spec'teki aksanli halleriyle birebir eslesir; Gemini'ye gonderilen
# prompt yukarida ASCII'ye sadelestirilmistir, model alias'lari tolere eder.)


class MeterRow(BaseModel):
    bina: str = Field(default="", alias="BİNA")
    daire: str = Field(default="", alias="DAİRE")
    marka: str = Field(default="", alias="MARKA")
    sayac_turu: str = Field(default="", alias="SAYAÇ TÜRÜ")
    durum: str = Field(default="", alias="DURUM")
    tutar: str = Field(default="", alias="TUTAR (TL)")
    eski_endeks: Any = Field(default="", alias="ESKİ ENDEKS")
    yeni_seri_no: str = Field(default="", alias="YENİ SERİ NO (BARKOD)")
    notlar: str = Field(default="", alias="NOTLAR")
    enlem: str = Field(default="", alias="ENLEM")
    boylam: str = Field(default="", alias="BOYLAM")

    model_config = {"populate_by_name": True}


class MeterResult(BaseModel):
    data: list[MeterRow] = Field(default_factory=list)
