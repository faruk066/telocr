"""Telegram bot ana dongu (async, python-telegram-bot v20+)."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import config
from ai_parser import parse_meter_photos
from excel_maker import make_excel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("telocr")

os.makedirs(config.TEMP_DIR, exist_ok=True)

# media_group_id -> {"photos": [file_id], "task": Task, "ack_chat": chat_id, "ask_msg_id": int}
_album_buffer: dict[str, dict] = {}

# chat_id -> {"photos": [file_id], "kind": "single"|"album", "group_id": str|None, "ask_msg_id": int}
_pending_choice: dict[int, dict] = {}

PROVIDER_LABEL = {"gemini": "Gemini", "nvidia": "NVIDIA"}


def _provider_keyboard(n_photos: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("1️⃣ Gemini", callback_data=f"ai:gemini:{n_photos}"),
                InlineKeyboardButton("2️⃣ NVIDIA", callback_data=f"ai:nvidia:{n_photos}"),
            ]
        ]
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Merhaba! Sayac degisim formu fotografi gonderin.\n"
        "Birden fazla fotografi album olarak tek seferde gonderebilirsiniz; "
        "hepsi tek bir Excel dosyasinda birlestirilir.\n"
        "Fotografi alinca size sorarim: 1-Gemini, 2-NVIDIA."
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Net bir form fotografi gonderin.\n"
        "- Tek foto: 1-Gemini / 2-NVIDIA butonlari (veya '1' / '2' yazin).\n"
        "- Album gonderirseniz ~2 sn beklenip toplu islenir, sonra yine sorulur.\n"
        "- Sonuc bicimlendirilmis .xlsx olarak geri gonderilir."
    )


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg or not msg.photo:
        return
    file_id = msg.photo[-1].file_id
    group_id = msg.media_group_id
    try:
        if group_id:
            buf = _album_buffer.get(group_id)
            if buf is None:
                ask = await msg.reply_text(
                    "Album aliniyor, toplaniyor... ⏳\n"
                    "Toplama bitince hangi AI ile islenecegini soracagim."
                )
                buf = {"photos": [], "task": None, "ack_chat": msg.chat_id,
                       "ask_msg_id": ask.message_id}
                _album_buffer[group_id] = buf
            else:
                try:
                    await context.bot.edit_message_text(
                        chat_id=msg.chat_id, message_id=buf["ask_msg_id"],
                        text=f"Album aliniyor ({len(buf['photos']) + 1} foto)... ⏳",
                    )
                except Exception:
                    pass
            buf["photos"].append(file_id)
            old_task = buf.get("task")
            if old_task and not old_task.done():
                old_task.cancel()
            buf["task"] = asyncio.create_task(_ask_album_choice_after_delay(group_id, context))
        else:
            ask = await msg.reply_text(
                "Fotograf alindi. Hangisiyle isleyeyim?\n1-Gemini, 2-NVIDIA",
                reply_markup=_provider_keyboard(1),
            )
            _pending_choice[msg.chat_id] = {
                "photos": [file_id], "kind": "single", "group_id": None,
                "ask_msg_id": ask.message_id,
            }
    except Exception:
        logger.error("on_photo hatasi.", exc_info=True)
        try:
            await msg.reply_text("Beklenmeyen bir hata olustu, lutfen tekrar deneyin.")
        except Exception:
            pass


async def on_provider_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data or not query.data.startswith("ai:"):
        return
    await query.answer()
    try:
        _, provider, _n = query.data.split(":")
    except ValueError:
        return
    if provider not in ("gemini", "nvidia"):
        return
    chat_id = query.message.chat_id
    pend = _pending_choice.pop(chat_id, None)
    if not pend:
        await query.edit_message_text("Bu secim zamanaşımına uğradı, fotoğrafı tekrar gönderin.")
        return
    try:
        await query.edit_message_text(
            f"{PROVIDER_LABEL[provider]} ile isleniyor ({len(pend['photos'])} foto), lutfen bekleyin... ⏳"
        )
    except Exception:
        pass
    if pend["kind"] == "album":
        await _run_album_job(chat_id, pend["photos"], provider, context,
                             status_msg=query.message.message_id)
    else:
        await _run_single_job(chat_id, pend["photos"][0], provider, context,
                              reply_to=pend.get("ask_msg_id"))


async def on_choice_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Butona basmak yerine '1' / '2' / 'gemini' / 'nvidia' yazanlar icin."""
    msg = update.message
    if not msg or not msg.text:
        return
    chat_id = msg.chat_id
    if chat_id not in _pending_choice:
        return
    t = msg.text.strip().lower()
    provider = None
    if t in ("1", "gemini", "1️⃣", "1-gemini", "1. gemini"):
        provider = "gemini"
    elif t in ("2", "nvidia", "2️⃣", "2-nvidia", "2. nvidia"):
        provider = "nvidia"
    if not provider:
        await msg.reply_text("Lutfen 1 (Gemini) veya 2 (NVIDIA) yazin ya da butona basin.")
        return
    pend = _pending_choice.pop(chat_id, None)
    if not pend:
        return
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=pend["ask_msg_id"],
            text=f"{PROVIDER_LABEL[provider]} ile isleniyor ({len(pend['photos'])} foto), lutfen bekleyin... ⏳",
        )
    except Exception:
        pass
    if pend["kind"] == "album":
        await _run_album_job(chat_id, pend["photos"], provider, context,
                             status_msg=pend.get("ask_msg_id"))
    else:
        await _run_single_job(chat_id, pend["photos"][0], provider, context,
                              reply_to=pend.get("ask_msg_id"))


async def _ask_album_choice_after_delay(group_id: str, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Album toplama bitince indirme YAPMADAN once AI secimini sorar."""
    try:
        await asyncio.sleep(config.ALBUM_COLLECT_DELAY_S)
        buf = _album_buffer.pop(group_id, None)
        if not buf or not buf["photos"]:
            return
        chat_id = buf["ack_chat"]
        file_ids: list[str] = buf["photos"]
        _pending_choice[chat_id] = {
            "photos": file_ids, "kind": "album", "group_id": group_id,
            "ask_msg_id": buf["ask_msg_id"],
        }
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=buf["ask_msg_id"],
                text=f"{len(file_ids)} foto toplandi. Hangisiyle isleyeyim?\n1-Gemini, 2-NVIDIA",
                reply_markup=_provider_keyboard(len(file_ids)),
            )
        except Exception:
            ask = await context.bot.send_message(
                chat_id,
                f"{len(file_ids)} foto toplandi. Hangisiyle isleyeyim?\n1-Gemini, 2-NVIDIA",
                reply_markup=_provider_keyboard(len(file_ids)),
            )
            _pending_choice[chat_id]["ask_msg_id"] = ask.message_id
    except asyncio.CancelledError:
        pass  # yeni foto geldi, zamanlayici sifirlandi


async def _progress_cb(chat_id: int, ask_msg_id: int | None,
                       context: ContextTypes.DEFAULT_TYPE, label: str) -> None:
    """NVIDIA ilk token uretmeye basladi -> kullaniciya canli bilgi."""
    try:
        if ask_msg_id:
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=ask_msg_id,
                text=f"{label} okumaya basladi, satirlar cikariliyor... ✍️",
            )
        else:
            await context.bot.send_message(
                chat_id, f"{label} okumaya basladi, satirlar cikariliyor... ✍️")
    except Exception:
        pass


async def _run_single_job(chat_id: int, file_id: str, provider: str,
                          context: ContextTypes.DEFAULT_TYPE,
                          reply_to: int | None = None) -> None:
    label = PROVIDER_LABEL.get(provider, provider)
    image_paths: list[str] = []
    try:
        image_paths.append(await _download_photo(context, file_id))
        if provider == "nvidia":
            async def _prog() -> None:
                await _progress_cb(chat_id, reply_to, context, label)
        else:
            _prog = None
        rows = await parse_meter_photos(image_paths, provider=provider,
                                        on_progress=_prog)
        xlsx = await asyncio.to_thread(make_excel, rows)
        with open(xlsx, "rb") as f:
            await context.bot.send_document(
                chat_id, document=f,
                filename=os.path.basename(xlsx),
                caption=f"✅ {len(rows)} satir aktarildi ({label}).",
            )
        _cleanup(image_paths + [xlsx])
    except ValueError as ve:
        logger.warning("Tekil isleme uyari (%s): %s", label, ve)
        await context.bot.send_message(chat_id, str(ve))
        _cleanup(image_paths)
    except Exception:
        logger.error("Tekil isleme hatasi (%s).", label, exc_info=True)
        await context.bot.send_message(chat_id, "Islem sirasinda bir hata olustu, lutfen tekrar deneyin.")
        _cleanup(image_paths)


async def _run_album_job(chat_id: int, file_ids: list[str], provider: str,
                         context: ContextTypes.DEFAULT_TYPE,
                         status_msg=None) -> None:
    label = PROVIDER_LABEL.get(provider, provider)
    ask_id = status_msg if isinstance(status_msg, int) else None
    image_paths: list[str] = []
    try:
        for fid in file_ids:
            image_paths.append(await _download_photo(context, fid))
        if provider == "nvidia":
            async def _prog() -> None:
                await _progress_cb(chat_id, ask_id, context, label)
        else:
            _prog = None
        rows: list[dict] = await parse_meter_photos(image_paths, provider=provider,
                                                    on_progress=_prog)
        xlsx = await asyncio.to_thread(make_excel, rows)
        with open(xlsx, "rb") as f:
            await context.bot.send_document(
                chat_id, document=f,
                filename=os.path.basename(xlsx),
                caption=f"✅ {len(rows)} satir aktarildi ({label}).",
            )
        _cleanup(image_paths + [xlsx])
    except ValueError as ve:
        logger.warning("Tekil isleme uyari (%s): %s", label, ve)
        await context.bot.send_message(chat_id, str(ve))
        _cleanup(image_paths)
    except Exception:
        logger.error("Tekil isleme hatasi (%s).", label, exc_info=True)
        await context.bot.send_message(chat_id, "Islem sirasinda bir hata olustu, lutfen tekrar deneyin.")
        _cleanup(image_paths)


async def _run_album_job(chat_id: int, file_ids: list[str], provider: str,
                         context: ContextTypes.DEFAULT_TYPE,
                         status_msg=None) -> None:
    label = PROVIDER_LABEL.get(provider, provider)
    image_paths: list[str] = []
    try:
        for fid in file_ids:
            image_paths.append(await _download_photo(context, fid))
        rows: list[dict] = await parse_meter_photos(image_paths, provider=provider)
        xlsx = await asyncio.to_thread(make_excel, rows)
        with open(xlsx, "rb") as f:
            await context.bot.send_document(
                chat_id, document=f,
                filename=os.path.basename(xlsx),
                caption=f"✅ {len(rows)} satir aktarildi ({len(file_ids)} fotograf, {label}).",
            )
        _cleanup(image_paths + [xlsx])
    except ValueError as ve:
        logger.warning("Album isleme uyari (%s): %s", label, ve)
        await context.bot.send_message(chat_id, str(ve))
        _cleanup(image_paths)
    except Exception:
        logger.error("Album isleme hatasi (%s).", label, exc_info=True)
        await context.bot.send_message(
            chat_id, "Islem sirasinda bir hata olustu, lutfen daha net fotograflarla tekrar deneyin.")
        _cleanup(image_paths)


async def _process_album_after_delay(group_id: str, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Geriye uyumluluk: artik _ask_album_choice_after_delay kullaniliyor.
    await _ask_album_choice_after_delay(group_id, context)


async def _process_single(file_id: str, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Geriye uyumluluk sarmalayici (artik secim sonrasi _run_single_job cagrilir).
    await _run_single_job(update.effective_chat.id, file_id, "gemini", context)


async def _download_photo(context: ContextTypes.DEFAULT_TYPE, file_id: str) -> str:
    tg_file = await context.bot.get_file(file_id)
    fname = f"{int(time.time() * 1000)}_{file_id[:12].replace(':', '')}.jpg"
    path = os.path.join(config.TEMP_DIR, fname)
    await tg_file.download_to_drive(path)
    logger.info("Fotograf indirildi: %s", path)
    return path


def _cleanup(paths: list[str]) -> None:
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            logger.warning("Gecici dosya silinemedi: %s", p)


async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Handler hatasi: %s", context.error, exc_info=True)
    try:
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text("Bir hata olustu, lutfen tekrar deneyin.")
    except Exception:
        pass


def main() -> None:
    if not config.TELEGRAM_BOT_TOKEN or not config.GEMINI_API_KEY:
        raise SystemExit("TELEGRAM_BOT_TOKEN / GEMINI_API_KEY eksik. .env dosyasini kontrol edin.")
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(CallbackQueryHandler(on_provider_choice, pattern=r"^ai:(gemini|nvidia):\d+$"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_choice_text))
    app.add_error_handler(_on_error)
    logger.info("Bot baslatiliyor (model=%s)...", config.GEMINI_MODEL)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
