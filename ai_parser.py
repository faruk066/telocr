"""Gemini Vision ile sayac formu okuma (structured output + timeout + retry)."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
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

# NVIDIA (Llama-3.2-11B-Vision) uzun/talimat yuklu sistem istemini takip etmekte
# zorlanir: "Beklenen JSON Cikti Formati" blogunu tekrar tekrar eko eder ve 3 kat
# yavas doner (olculdu: 20.7 sn vs 7.3 sn). Bu yuzden NVIDIA icin KISA ve net bir
# istem kullanilir; is kurallari aynen korunur.
NVIDIA_SYSTEM_PROMPT = (
    "CEVABIN ilk karakteri '{' olsun. '###' basligi, adim adim aciklama, kod blogu, "
    "gorsel analizi, madde listesi veya dusunce akisi YAZMA; sadece tek bir JSON objesi yaz.\n"
    "JSON'u TEK SATIRDA compact yaz: girintisiz, ```json blogu kullanma, yorum ekleme.\n"
    "Sana verilen sayac formu fotografini oku ve su semaya uygun TEK obje dondur:\n"
    '{"data":[{"BINA":"","DAIRE":"","MARKA":"","SAYAC TURU":"","DURUM":"","TUTAR (TL)":"",'
    '"ESKI ENDEKS":0,"YENI SERI NO (BARKOD)":"","NOTLAR":"","ENLEM":"","BOYLAM":""}]}\n'
    "Kurallar:\n"
    "1. 8 haneli yeni seri numarasi (barkod) varsa DURUM='Tamamlandi'.\n"
    "2. Seri numarasi yoksa ve daire karsisinda sadece 'f' isareti varsa DURUM='faruk'.\n"
    "3. Seri numarasi yoksa ve 'Evde yok', 'Bos', 'Dolap kesilecek' gibi not varsa DURUM='Iptal'.\n"
    "4. Seri no '30' ile basliyorsa SAYAC TURU='ULTRASONIK', degilse 'SICAK SU'.\n"
    "5. 'EEY', 'E.Y.', 'Ek yok', 'Sifir' notlari varsa ESKI ENDEKS=0.\n"
    "6. Seri no kisaltilmis yazildiysa ('// 2681' gibi) ustteki seri bloklarina bakip 8 haneye tamamla.\n"
    "7. Fotograftaki tum satirlari TEK 'data' listesinde ver; ayni daireyi iki kez yazma "
    "(seri numarali olani tut).\n"
    "8. Seri numarasi okunamiyorsa o alani BOS string ('') birak; notu veya aciklamayi ASLA "
    "'YENI SERI NO (BARKOD)' alanina yazma. Bos satirlari listeye ekleme."
)

# Kimi-K3 (reasoning model) icin AYRI ve KISA istem. Olculmus davranis:
#   - Uzun, kural yogun istemle model 34 karakter reasoning uretip hic icerik
#     vermeden duruyor (finish='stop', bos icerik).
#   - Kisa istemle (asagida) 117 sn'de tek satir compact JSON veriyor.
KIMI_SYSTEM_PROMPT = (
    "Bu sayac formu fotografini oku ve her satiri JSON'a cikar.\n"
    'Cevap SADICE su bicimde tek satir bir JSON olsun: {"data":[{"BINA":"","DAIRE":"",'
    '"MARKA":"","SAYAC TURU":"","DURUM":"","TUTAR (TL)":"","ESKI ENDEKS":0,'
    '"YENI SERI NO (BARKOD)":"","NOTLAR":"","ENLEM":"","BOYLAM":""}]}\n'
    "Kurallar: 8 haneli yeni seri numarasi varsa DURUM='Tamamlandi'. Seri yoksa ve daire "
    "karsisinda 'f' isareti varsa DURUM='faruk'; 'Evde yok'/'Bos' notu varsa "
    "DURUM='Iptal'. Seri no '30' ile basliyorsa SAYAC TURU='ULTRASONIK', degilse "
    "'SICAK SU'. 'EEY' notu varsa ESKI ENDEKS=0. Bos satirlari listeye ekleme."
)


def _nvidia_prompt(strict: bool = False) -> str:
    """Aktif NVIDIA modeline uygun istemi secer (kimi vs digerleri)."""
    base = (KIMI_SYSTEM_PROMPT if "kimi" in config.NVIDIA_MODEL.lower()
            else NVIDIA_SYSTEM_PROMPT)
    return base + (NVIDIA_STRICT_SUFFIX if strict else "")


# NVIDIA kucuk modeli bazen basliklari Turkce karakterler olmadan uretir
# (DAIRE, YENI SERI NO (BARKOD) ...). Pydantic alias'lari aksanli oldugu icin
# bu degerler dogrulamada sessizce bosalir; asagidaki harita ile kanonik
# basliklara cevrilir.
_KEY_ALIASES: dict[str, str] = {
    "bina": "BİNA",
    "daire": "DAİRE",
    "marka": "MARKA",
    "sayac turu": "SAYAÇ TÜRÜ",
    "sayac türü": "SAYAÇ TÜRÜ",
    "durum": "DURUM",
    "tutar": "TUTAR (TL)",
    "tutar (tl)": "TUTAR (TL)",
    "eski endeks": "ESKİ ENDEKS",
    "yeni seri no": "YENİ SERİ NO (BARKOD)",
    "yeni seri no (barkod)": "YENİ SERİ NO (BARKOD)",
    "yeni seri": "YENİ SERİ NO (BARKOD)",
    "seri no": "YENİ SERİ NO (BARKOD)",
    "barkod": "YENİ SERİ NO (BARKOD)",
    "notlar": "NOTLAR",
    "not": "NOTLAR",
    "enlem": "ENLEM",
    "boylam": "BOYLAM",
}


def _normalize_row_keys(row: dict) -> dict:
    """Satir anahtarlarini kanonik (aksanli) basliklara cevirir.

    Hem aksansiz varyantlari (DAIRE -> DAİRE) hem de Turkce buyuk/kucuk harf
    farklarini tolere eder; bilinmeyen alanlar aynen korunur.
    """
    out: dict = {}
    for k, v in row.items():
        key = str(k).strip()
        canonical = _KEY_ALIASES.get(key.lower())
        if canonical is None:
            # Aksanli eslesme: buyuk harf duyarsiz karsilastirma (DAİRE/daire)
            for alias_value in set(_KEY_ALIASES.values()):
                if alias_value.casefold() == key.casefold():
                    canonical = alias_value
                    break
        out[canonical or key] = v
    return out


def _get_field(row: dict, *names: str) -> str:
    """Satirdan ilk bulunan alani doner (kanonik + ASCII varyantlar)."""
    for name in names:
        val = row.get(name)
        if val not in (None, ""):
            return str(val)
    return ""


def _looks_like_row(row: dict) -> bool:
    """NVIDIA bazen aciklama/bos obje uretir; gercek satirda daire veya seri vardir."""
    daire = _get_field(row, "DAİRE", "DAIRE").strip()
    seri = _get_field(row, "YENİ SERİ NO (BARKOD)", "YENI SERI NO (BARKOD)").strip()
    return bool(daire or seri)


def _dedupe_exact_rows(rows: list[dict]) -> list[dict]:
    """Birebir ayni satirlari teke indirir.

    Model tekrara dustugunde ayni satiri onlarca kez uretebilir; icerigi
    tamamen ayni olan satirlar bilgi kaybi olmadan atilabilir.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        fingerprint = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append(row)
    return out


def _clean_rows(raw: list) -> list[dict]:
    """Ham model satirlarini normalize edip gercek satirlari secer."""
    rows: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        row = _normalize_row_keys(item)
        # Model bazen JSON yerine duz metni seri alanina koyar ("Evde yok" gibi);
        # bu degeri seri alaninda tutmak seri eslesmesini/kontrolunu bozar.
        seri = _get_field(row, "YENİ SERİ NO (BARKOD)")
        digits = "".join(ch for ch in seri if ch.isdigit())
        if seri.strip() and len(digits) < 4:
            logger.info("Model seri alaninda sayisal olmayan deger dondurdu: %r -> bosaltildi.",
                        seri[:60])
            row["YENİ SERİ NO (BARKOD)"] = ""
        if not _looks_like_row(row):
            continue
        rows.append(row)
    deduped = _dedupe_exact_rows(rows)
    if len(deduped) != len(rows):
        logger.info("Tekrar eden %d satir atildi (%d -> %d).",
                    len(rows) - len(deduped), len(rows), len(deduped))
    return deduped


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

# Tekrar denemede (2. girisim) kullanilan ek talimat: 11B model bazen cevaba
# basliga/yoruma girip donguye dusuyor; bu eki alinca dogrudan JSON'a basliyor.
NVIDIA_STRICT_SUFFIX = (
    "\n\nONEMLI: Simdi sadece tek satir compact JSON uret. Ilk karakter '{' olsun; "
    "baslik, aciklama, ```json blogu, madde listesi yazma."
)


def _merge_json_data_objects(text: str) -> dict | None:
    """Metindeki TUM {"data": [...]} objelerini bulup birlestirir.

    Llama-Vision bazen sema ornegini eko edip birden fazla data blogu uretir
    (json.loads "Extra data" hatasi verirdi). Bloklar birlestirilir, tamamen
    ayni satirlar tekillestirilir; hic blok yoksa None doner.
    """
    cleaned = text.strip()
    decoder = json.JSONDecoder()
    objects: list[dict] = []
    for cand in ([*_fenced_candidates(cleaned), cleaned]):
        idx = 0
        while True:
            start = cand.find("{", idx)
            if start < 0:
                break
            try:
                obj, _end = decoder.raw_decode(cand[start:])
            except json.JSONDecodeError:
                idx = start + 1
                continue
            if isinstance(obj, dict) and isinstance(obj.get("data"), list):
                objects.append(obj)
            idx = start + 1
        if objects:
            break
    if not objects:
        return None
    if len(objects) > 1:
        counts = [len(o.get("data") or []) for o in objects]
        logger.info("Model %d adet JSON blogu dondurdu (satir sayilari=%s), birlestiriliyor.",
                    len(objects), counts)
    rows: list[dict] = []
    for obj in objects:
        for row in obj.get("data") or []:
            if isinstance(row, dict):
                rows.append(row)
    # Model ayni satiri dongu icinde tekrar uretebilir -> tekillestir
    return {"data": _dedupe_exact_rows(rows)}


def _fenced_candidates(text: str) -> list[str]:
    """```json ... ``` bloklarinin iceriklerini dondurur."""
    out: list[str] = []
    if "```" not in text:
        return out
    for part in text.split("```"):
        cand = part.strip()
        if cand.lower().startswith("json"):
            cand = cand[4:].strip()
        if "{" in cand:
            out.append(cand)
    return out


def _salvage_partial_rows(text: str) -> list[dict]:
    """Yarim kalmis (token limiti) {"data": [ {...}, {...}  ciktisindan
    kapanmis satir objelerini kurtarir; hicbiri yoksa bos liste doner."""
    idx = text.find('"data"')
    if idx < 0:
        return []
    bracket = text.find("[", idx)
    if bracket < 0:
        return []
    decoder = json.JSONDecoder()
    rows: list[dict] = []
    pos = bracket + 1
    while True:
        start = text.find("{", pos)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            # Yarim/bozuk blok: sonraki '{' adayini dene (basdaki tekrarli
            # '{"data": [' kalibi boylece atlanir ve gercek satirlar kurtarilir).
            pos = start + 1
            continue
        if isinstance(obj, dict):
            if isinstance(obj.get("data"), list):
                # Model ic ice "data" blogu urettiyse satirlari duzlestir
                rows.extend(x for x in obj["data"] if isinstance(x, dict))
            else:
                rows.append(obj)
        pos = start + end
    return _dedupe_exact_rows(rows)


# Son care: model JSON yerine aciklama/madde listesi yazdiysa metinden satir
# cikarma. Ornek satir: "*   Daire 1: 30502527 - DURUM='Tamamlandi'"
_PROSE_ROW_RE = re.compile(r"daire\s*[:#]?\s*([0-9]{1,3})\s*[:\-]\s*([0-9]{8})?", re.IGNORECASE)
_PROSE_DURUM_RE = re.compile(r"durum\s*=\s*['\"]?([A-Za-zçÇğĞıİöÖşŞüÜ ]+)['\"]?", re.IGNORECASE)


def _rows_from_prose(text: str) -> list[dict]:
    """Model aciklama metninden 'Daire N: 8haneliSeri' satirlarini kurtarir.

    11B model bazen JSON yerine tabloyu madde listesi olarak yazip token
    limitine takilir; orada veri zaten mevcuttur. En az 2 satir bulunursa
    kullanilir (tek satirlik yanilgi karisimasi olmasin diye).
    """
    rows: list[dict] = []
    for line in text.splitlines():
        m = _PROSE_ROW_RE.search(line)
        if not m or not m.group(2):
            continue
        durum = ""
        dm = _PROSE_DURUM_RE.search(line)
        if dm:
            durum = dm.group(1).strip()
        rows.append({"DAIRE": m.group(1), "YENI SERI NO (BARKOD)": m.group(2),
                     "DURUM": durum})
    return _dedupe_exact_rows(rows)


def _extract_json_object(text: str) -> dict:
    """Model ciktisindaki gecerli {"data": ...} JSON objesini bulup parse eder.

    Llama-Vision aciklama + ```json fence ekleyebilir, ayrica stream
    birlesiminde birden fazla JSON blogu olabilir (bu durumda
    json.loads "Extra data" hatasi verirdi). JSONDecoder.raw_decode ile
    metni tarayip icinde "data" anahtari olan objeleri toplar.
    """
    cleaned = text.strip()
    merged = _merge_json_data_objects(cleaned)
    if merged is not None:
        return merged
    salvaged = _salvage_partial_rows(cleaned)
    if salvaged:
        logger.warning("Parcalanmis JSON'dan %d satir kurtarildi.", len(salvaged))
        return {"data": salvaged}
    decoder = json.JSONDecoder()
    for cand in [*_fenced_candidates(cleaned), cleaned]:
        # data'li obje bulunamadiysa: bu adaydaki ilk gecerli dict'i kabul et
        idx = 0
        while True:
            start = cand.find("{", idx)
            if start < 0:
                break
            try:
                obj, _end = decoder.raw_decode(cand[start:])
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
                 on_first_token: Any | None = None,
                 strict: bool = False) -> dict:
    """Coklu fotografi tek tek okur ve satirlari birlestirir.

    ONEMLI: NVIDIA NIM 11B-vision ucu tek istekte EN FAZLA 1 gorsel kabul eder;
    fazlasi "400 At most 1 image(s) may be provided in one prompt" hatasi verir
    (album islemlerinin canlida patlamasinin nedeni buydu). Bu yuzden her gorsel
    ayri cagrilir, satirlar burada birlestirilir; ayni daireye ait mukerrer
    satirlarin tekillestirilmesi excel_maker.merge_duplicate_daires'te yapilir.
    """
    if not image_bytes_list:
        raise ValueError("Islenecek gorsel yok.")
    prompt = _nvidia_prompt(strict)
    # 2. deneme: gateway'in "max" effort icin 504 verdigi olculdu -> emniyet degerine dus
    effort = (config.NVIDIA_FALLBACK_REASONING_EFFORT if strict
              else config.NVIDIA_REASONING_EFFORT)
    logger.info("NVIDIA istemi: model=%s, reasoning_effort=%s, max_tokens=%d, strict=%s",
                config.NVIDIA_MODEL, effort, config.NVIDIA_MAX_TOKENS, strict)
    merged_rows: list[dict] = []
    for i, image_bytes in enumerate(image_bytes_list):
        payload = _call_nvidia_single(image_bytes, prompt,
                                      on_first_token if i == 0 else None, effort)
        rows = payload.get("data") or []
        if not isinstance(rows, list):
            rows = []
        logger.info("NVIDIA gorsel %d/%d -> %d ham satir.",
                    i + 1, len(image_bytes_list), len(rows))
        merged_rows.extend(r for r in rows if isinstance(r, dict))
    return {"data": merged_rows}


def _call_nvidia_single(image_bytes: bytes, prompt: str,
                        on_first_token: Any | None = None,
                        effort: str | None = None) -> dict:
    import requests  # yerel import: sadece nvidia yolu kullanildiginda gerekir

    if not config.NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY bos (.env).")
    effort = effort or config.NVIDIA_REASONING_EFFORT
    shrunk = _shrink_for_nvidia(image_bytes)
    logger.info("NVIDIA'ya 1 gorsel gonderiliyor (~%d KB, model=%s).",
                len(shrunk) // 1024, config.NVIDIA_MODEL)
    content: list[dict] = [
        {
            "type": "text",
            "text": prompt,
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(shrunk)}"},
        },
    ]
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
            "max_tokens": config.NVIDIA_MAX_TOKENS,
            "seed": config.NVIDIA_SEED,
            "stream": True,
            "temperature": config.NVIDIA_TEMPERATURE,
            # Kimi-K3 gibi reasoning modeller icin; desteklemeyen modellerde
            # sunucu bu alani yok sayar.
            "reasoning_effort": effort,
        },
        timeout=(30, config.NVIDIA_TIMEOUT_S),
        stream=True,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"NVIDIA API {resp.status_code}: {resp.text[:500]}")
    chunks: list[str] = []
    first_seen = False
    finish_reason = ""
    reasoning_chars = 0
    buf = ""             # tam cevap metni (aciklama + JSON) burada toplanir
    depth = 0            # suslu parantez derinligi ("{" -> +1, "}" -> -1)
    saw_json_start = False
    closed_at = -1       # tam JSON objesinin kapandigi andaki buf uzunlugu
    looping = False
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
            choice = (obj.get("choices") or [{}])[0]
            if isinstance(choice, dict) and choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
            delta_obj = choice.get("delta") if isinstance(choice, dict) else None
            if not isinstance(delta_obj, dict):
                delta_obj = {}
            # Reasoning modelde dusunce delta.reasoning_content ile gelir; JSON'a
            # girmez ama kullaniciya "okunuyor" bilgisi vermek icin sayilir.
            reasoning = delta_obj.get("reasoning_content") or ""
            reasoning_chars += len(reasoning)
            try:
                delta = delta_obj.get("content") or ""
            except AttributeError:
                delta = ""
            if not delta and not reasoning:
                continue
            if reasoning and not first_seen:
                first_seen = True
                if on_first_token is not None:
                    try:
                        on_first_token()
                    except Exception:
                        pass
            if not delta:
                continue
            chunks.append(delta)
            buf += delta
            if not first_seen:
                first_seen = True
                if on_first_token is not None:
                    try:
                        on_first_token()
                    except Exception:
                        pass
            # 1) Tam JSON objesi kapandi mi? Model sonrasinda tekrara girse bile
            #    beklemeden cik (olculdu: 116 sn -> ~10 sn). Yeni bir "data" blogu
            #    baslarsa (cok fotolu birlesim) okumaya devam et.
            for ch in delta:
                if ch == "{":
                    depth += 1
                    saw_json_start = True
                elif ch == "}":
                    depth -= 1
            if saw_json_start and depth <= 0 and '"data"' in buf:
                new_block = '"data"' in delta
                if new_block:
                    closed_at = -1
                elif closed_at < 0:
                    closed_at = len(buf)
                elif len(buf) - closed_at > 200:
                    logger.info("JSON kapandi; kuyruktaki fazlalik atlandi (%d -> %d chr).",
                                len(buf), closed_at)
                    break
            # 2) Tekrar dongusu: ayni 200 karakterlik blok 3+ kez gectiyse kes
            if len(buf) > 600 and not looping:
                tail = buf[-200:]
                if buf.count(tail) >= 3:
                    looping = True
                    logger.warning("NVIDIA cikti tekrara dustu, akis kesildi (%.600s...)", buf[:200])
                    break
    finally:
        resp.close()
    if finish_reason == "length":
        # Cikti token limitine takildi -> JSON yarim kalabilir; parse tarafi
        # kapanmis bloklari kurtarmaya calisir, logda gorunur olsun.
        logger.warning("NVIDIA cikti token limitine takildi (max_tokens=%d, %d parca).",
                       config.NVIDIA_MAX_TOKENS, len(chunks))
    text = buf
    if not text.strip():
        logger.warning("NVIDIA icerik uretmedi (reasoning=%d karakter, finish=%r, "
                       "max_tokens=%d). reasoning_effort/max_tokens degerlerini gözden geçir.",
                       reasoning_chars, finish_reason, config.NVIDIA_MAX_TOKENS)
        raise ValueError("NVIDIA bos yanit dondu.")
    try:
        result = _extract_json_object(text)
    except Exception:
        # Son care: model JSON yerine aciklama/madde listesi yazdiysa
        # ("* Daire 1: 30502527 - DURUM='Tamamlandi'") satirlari metinden cikar.
        salvaged = _rows_from_prose(text)
        if len(salvaged) >= 2:
            logger.warning("NVIDIA JSON uretmedi; aciklama metninden %d satir kurtarildi.",
                           len(salvaged))
            return {"data": salvaged}
        logger.warning("NVIDIA JSON parse edilemedi (finish=%r, %d chr, reasoning=%d). "
                       "Ham yanit (ilk 1000 chr): %.1000s",
                       finish_reason, len(text), reasoning_chars, text)
        raise
    n_rows = len(result.get("data") or []) if isinstance(result, dict) else 0
    if n_rows < 3:
        logger.info("NVIDIA az satir uretti (%d) (finish=%r). "
                    "Ham yanit (ilk 600 chr): %.600s", n_rows, finish_reason, text)
    return result


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
                raw_rows = result.get("data", []) if isinstance(result, dict) else []
                rows = _clean_rows(raw_rows) if isinstance(raw_rows, list) else []
                if not rows:
                    raise ValueError("Model veri uretemedi (bos liste).")
                validated = MeterResult.model_validate({"data": rows})
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
            # Her gorsel ayri istek oldugu icin toplam sure gorsel sayisiyla olceklenir.
            batch_timeout = (config.NVIDIA_TIMEOUT_S + 30) * max(1, len(image_bytes_list))
            # 2. deneme strict istemle: model basliga/yoruma girip donguye dusmusse
            # JSON'a dogrudan baslamasi saglanir.
            result = await asyncio.wait_for(
                asyncio.to_thread(_call_nvidia, image_bytes_list, _cb,
                                  strict=(attempt == 1)),
                timeout=batch_timeout,
            )
            rows = result.get("data", []) if isinstance(result, dict) else []
            if not isinstance(rows, list):
                raise ValueError("Model 'data' listesi uretemedi.")
            # Anahtarlari kanonik basliklara cevir (DAIRE -> DAİRE ...) ve bos
            # satirlari at. Aksi halde pydantic alias'lari eslesmez ve tum
            # degerler bosalir -> "sadece baslikli bos Excel".
            rows = _clean_rows(rows)
            logger.info("NVIDIA %d satir dondu (model=%s).", len(rows), config.NVIDIA_MODEL)
            if not rows:
                raise ValueError("Model veri uretemedi (bos liste).")
            validated = MeterResult.model_validate({"data": rows})
            normalized = validated.model_dump(by_alias=True)
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
                           on_progress: Any | None = None,
                           used_provider: list[str] | None = None) -> list[dict]:
    """Foto listesini okuyup satir listesi doner.

    provider verilirse ("gemini"/"nvidia") o once denenir; basarisiz olursa
    diger saglayicilarla yedeklenir (kullanici bos Excel yerine sonuc alir).
    Verilmezse AI_PROVIDER_ORDER sirasinda failover yapilir.
    used_provider verilirse gercekten sonucu ureten saglayicinin adi yazilir
    (bot mesajini ona gore etiketlemek icin).
    Basarisizlikta ValueError yukseltir.
    """
    if not image_paths:
        raise ValueError("Islenecek fotograf bulunamadi.")
    image_bytes_list = [await asyncio.to_thread(resize_image_for_api, p) for p in image_paths]
    total_kb = sum(len(b) for b in image_bytes_list) // 1024
    if provider in ("gemini", "nvidia"):
        order = [provider] + [p for p in config.AI_PROVIDER_ORDER if p != provider]
    else:
        order = list(config.AI_PROVIDER_ORDER)
    logger.info("AI'ye %d gorsel gonderiliyor (~%d KB) [sira=%s].",
                len(image_bytes_list), total_kb, "+".join(order))
    last_err: Exception | None = None
    for prov in order:
        try:
            if prov == "nvidia":
                if not config.NVIDIA_API_KEY:
                    logger.warning("NVIDIA atlandi: NVIDIA_API_KEY bos.")
                    continue
                rows = await _parse_with_nvidia(image_bytes_list, on_progress=on_progress)
            else:
                rows = await _parse_with_gemini(image_bytes_list)
            if used_provider is not None:
                used_provider.clear()
                used_provider.append(prov)
            return rows
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
