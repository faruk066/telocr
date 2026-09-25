"""JSON satirlari -> dogrulamali, bicimlendirilmis .xlsx uretimi."""
from __future__ import annotations

import logging
import os
import re
import time
from collections import Counter

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import config

logger = logging.getLogger(__name__)

COLUMNS = [
    "BİNA",
    "DAİRE",
    "MARKA",
    "SAYAÇ TÜRÜ",
    "DURUM",
    "TUTAR (TL)",
    "ESKİ ENDEKS",
    "YENİ SERİ NO (BARKOD)",
    "NOTLAR",
    "ENLEM",
    "BOYLAM",
    "KONTROL NOTU",
]

HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(bold=True, color="FFFFFF")
GREEN_FILL = PatternFill("solid", fgColor="C6EFCE")
RED_FILL = PatternFill("solid", fgColor="FFC7CE")
YELLOW_FILL = PatternFill("solid", fgColor="FFEB9C")
GREEN_FONT = Font(color="006100")
RED_FONT = Font(color="9C0006")

# Mukerrer seri gruplari icin palet: ayni seri -> ayni renk.
# Acik pastel zemin + koyu yazi (okunabilirlik icin).
DUP_PALETTE: list[tuple[str, str]] = [
    ("FFD9E1", "9C0A2E"),  # pembe
    ("D9E8FF", "1A3FAA"),  # mavi
    ("FFE8C2", "9A5B00"),  # turuncu
    ("D9F2DF", "0E6B2E"),  # yesil
    ("E8D9FF", "5B21B6"),  # mor
    ("FFF3B0", "7A6200"),  # sari
    ("D6F0F0", "0B6E6E"),  # turkuaz
    ("F2D9D9", "8C1D1D"),  # bordo
    ("E2E8F0", "334155"),  # gri-mavi
    ("DCFCE7", "166534"),  # acik yesil
    ("FEF3C7", "92400E"),  # amber
    ("E0E7FF", "3730A3"),  # indigo
]

WARN_TEXT = "⚠️ Kontrol Edilmeli"


def _digits(s: object) -> str:
    return re.sub(r"\D", "", str(s or ""))


def validate_rows(rows: list[dict]) -> list[dict]:
    """8-hane + mukerrer kontrolu. Sorunlu satira KONTROL NOTU ekler, islemi durdurmaz."""
    serials = [_digits(r.get("YENİ SERİ NO (BARKOD)", "")) for r in rows]
    counts = Counter(s for s in serials if s)
    daire_counts = Counter(_daire_key(r.get("DAİRE", "")) for r in rows)
    out: list[dict] = []
    for r, serial in zip(rows, serials):
        warnings: list[str] = []
        if serial:
            if len(serial) != 8:
                warnings.append(f"Seri no 8 haneli degil ({serial})")
            if counts[serial] > 1:
                warnings.append(f"Mukerrer seri no ({serial})")
        if daire_counts[_daire_key(r.get("DAİRE", ""))] > 1:
            warnings.append(f"Ayni daire birden fazla satirda ({r.get('DAİRE', '')})")
        # Mevcut NOTLAR'i koru, kontrol notunu ayri sutuna yaz
        row = {c: r.get(c, "") for c in COLUMNS if c != "KONTROL NOTU"}
        row["KONTROL NOTU"] = "; ".join(warnings) if warnings else ""
        if warnings:
            row["KONTROL NOTU"] = f"{WARN_TEXT}: " + row["KONTROL NOTU"]
        out.append(row)
    return out


def _building_prefix(rows: list[dict]) -> str:
    """Dosya adi oneki: en sik gecen BINA degeri, guvenli dosya adiyla.

    Ornek: BINA='Mega Park B Blok' -> 'Mega_Park_B_Blok_'.
    Bina bos ya da hepsi farkliysa bos string doner (eski ad korunur).
    """
    names = [str(r.get("BİNA", "") or "").strip() for r in rows]
    names = [n for n in names if n]
    if not names:
        return ""
    top, count = Counter(names).most_common(1)[0]
    if count < 2 and len(set(names)) > 1:
        return ""  # tutarli bir bina adi yok -> tahmin yurutme
    safe = re.sub(r"[^\w\-]+", "_", top, flags=re.UNICODE).strip("_")
    safe = re.sub(r"_+", "_", safe)
    return f"{safe}_" if safe else ""


def _daire_key(v: object) -> str:
    """Daire karsilastirmasi icin normalize anahtar: 'Daire 7' -> '7'."""
    s = str(v or "").strip().lower()
    m = re.search(r"\d+", s)
    return m.group(0) if m else s


def _serial_score(r: dict) -> tuple[int, int]:
    """Ayni dairenin iki kaydi capisirsa hangisi kazanir: (seri_var_mi, seri_uzunlugu)."""
    s = _digits(r.get("YENİ SERİ NO (BARKOD)", ""))
    return (1 if s else 0, len(s))


def make_excel(rows: list[dict], out_dir: str | None = None) -> str:
    """Satirlari .xlsx'e yazar, dosya yolunu doner.

    Model cok fotolu birlestirmeyi atlamissa (ayni daire 2x), kod duzeltir:
    her daireden en iyi kayit tutulur, daire sirasina gore siralanir.
    """
    out_dir = out_dir or config.TEMP_DIR
    os.makedirs(out_dir, exist_ok=True)

    merged = merge_duplicate_daires(rows)
    if len(merged) != len(rows):
        logger.info("Cok-fotolu birlestirme: %d -> %d satir (mukerrer daireler tekillendi).",
                    len(rows), len(merged))
    validated = validate_rows(merged)
    df = pd.DataFrame(validated, columns=COLUMNS)

    fname = f"{_building_prefix(validated)}sayac_listesi_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
    path = os.path.join(out_dir, fname)
    df.to_excel(path, index=False, sheet_name="Sayaçlar")

    _format_workbook(path)
    warn_count = sum(1 for r in validated if r.get("KONTROL NOTU"))
    logger.info("Excel olusturuldu: %s (%d satir, %d uyarili).", path, len(validated), warn_count)
    return path


def merge_duplicate_daires(rows: list[dict]) -> list[dict]:
    """Ayni DAIRE'ye ait birden fazla kaydi teke indir + daireye gore sirala.

    Secim: seri numarali (dolu) kayit bosa tercih edilir; ikisi de doluysa
    daha uzun/net seri numarali kazanir. Kaybeden satirin NOTLAR'i doluysa
    ve kazananinki bossa, notlar birlestirilir (bilgi kaybi olmaz).
    """
    best: dict[str, dict] = {}
    order: list[str] = []
    for r in rows:
        key = _daire_key(r.get("DAİRE", ""))
        if not key:
            order.append(f"__noid_{len(order)}")
            best[order[-1]] = dict(r)
            continue
        if key not in best:
            best[key] = dict(r)
            order.append(key)
            continue
        cur = best[key]
        if _serial_score(r) > _serial_score(cur):
            winner, loser = dict(r), cur
        else:
            winner, loser = cur, dict(r)
        # Not kaybi olmasin: kazananin NOTLAR'i bossa, kaybedeninkini ekle
        w_note = str(winner.get("NOTLAR", "") or "").strip()
        l_note = str(loser.get("NOTLAR", "") or "").strip()
        if not w_note and l_note:
            winner["NOTLAR"] = l_note
        elif w_note and l_note and l_note.lower() not in w_note.lower():
            winner["NOTLAR"] = f"{w_note} / {l_note}"
        best[key] = winner
    merged = [best[k] for k in order]

    def _sort_key(r: dict) -> tuple[int, str]:
        m = re.search(r"\d+", str(r.get("DAİRE", "") or ""))
        return (int(m.group(0)) if m else 10**9, str(r.get("DAİRE", "")))

    merged.sort(key=_sort_key)
    return merged


def _format_workbook(path: str) -> None:
    import openpyxl

    wb = openpyxl.load_workbook(path)
    ws = wb["Sayaçlar"]

    # Baslik satiri
    for cell in ws[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 28
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    # DURUM kolonu kosullu renklendirme (deger bazli dogrudan boyama - basit ve guvenilir)
    try:
        durum_idx = COLUMNS.index("DURUM") + 1
    except ValueError:
        durum_idx = 5
    try:
        seri_idx = COLUMNS.index("YENİ SERİ NO (BARKOD)") + 1
    except ValueError:
        seri_idx = 8
    dup_groups = _duplicate_serial_groups(ws, seri_idx)
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        cell = row[durum_idx - 1]
        val = str(cell.value or "").strip().lower()
        if val == "tamamlandı":
            cell.fill = GREEN_FILL
            cell.font = GREEN_FONT
        elif val == "iptal":
            cell.fill = RED_FILL
            cell.font = RED_FONT
        elif val == "faruk":
            cell.fill = YELLOW_FILL
    _paint_duplicate_serials(ws, seri_idx, dup_groups)

    # Otomatik sutun genisligi (max 40)
    for i, col in enumerate(COLUMNS, start=1):
        letter = get_column_letter(i)
        max_len = len(col)
        for cell in ws[letter]:
            if cell.value is not None:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[letter].width = min(max_len + 4, 40)

    _add_dup_legend(ws)

    wb.save(path)


def _duplicate_serial_groups(ws, seri_idx: int) -> dict[str, list[int]]:
    """Ayni seri noyu paylasan satir numaralarini grupla (sadece 2+ gruplar)."""
    groups: dict[str, list[int]] = {}
    for r in range(2, ws.max_row + 1):
        serial = _digits(ws.cell(r, seri_idx).value)
        if serial:
            groups.setdefault(serial, []).append(r)
    return {s: rows for s, rows in groups.items() if len(rows) > 1}


def _paint_duplicate_serials(ws, seri_idx: int, dup_groups: dict[str, list[int]]) -> None:
    """Her mukerrer seri grubunun SERI + DAIRE hucrelerini ayni renge boya.

    Ayni seriler gozle eslesir; KONTROL NOTU'ndaki yazi da korunur.
    """
    try:
        daire_idx = COLUMNS.index("DAİRE") + 1
    except ValueError:
        daire_idx = 2
    for i, (serial, rows) in enumerate(sorted(dup_groups.items())):
        bg, fg = DUP_PALETTE[i % len(DUP_PALETTE)]
        fill = PatternFill("solid", fgColor=bg)
        font = Font(color=fg, bold=True)
        for r in rows:
            for c in (seri_idx, daire_idx):
                cell = ws.cell(r, c)
                cell.fill = fill
                cell.font = font


def _add_dup_legend(ws) -> None:
    """Tablonun altina mukerrer renk aciklamasi ekle (grup varsa)."""
    try:
        seri_idx = COLUMNS.index("YENİ SERİ NO (BARKOD)") + 1
        daire_idx = COLUMNS.index("DAİRE") + 1
    except ValueError:
        return
    dup_groups = _duplicate_serial_groups(ws, seri_idx)
    if not dup_groups:
        return
    start = ws.max_row + 2
    ws.cell(start, 1, value="MÜKERRER RENK LEJANTI (aynı renk = aynı seri no):").font = Font(bold=True)
    for i, (serial, rows) in enumerate(sorted(dup_groups.items())):
        bg, fg = DUP_PALETTE[i % len(DUP_PALETTE)]
        r = start + 1 + i
        daires = ", ".join(str(ws.cell(rr, daire_idx).value) for rr in rows)
        cell = ws.cell(r, 1, value=f"■ {serial} → Daire: {daires}")
        cell.fill = PatternFill("solid", fgColor=bg)
        cell.font = Font(color=fg, bold=True)
