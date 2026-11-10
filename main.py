import asyncio
import os
import random
from collections import defaultdict, deque
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

BOT_TOKEN = os.environ["BOT_TOKEN"]
OWNER_ID = int(os.environ["OWNER_ID"])
ALLOWED_IDS = {OWNER_ID} | {int(x) for x in os.getenv("ALLOWED_IDS", "").split(",") if x.strip()}
HER_NAME = os.environ["HER_NAME"]
ME_NAME = os.environ["ME_NAME"]

MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

BASE = Path(__file__).parent
CHAT = (BASE / "messages.txt").read_text(encoding="utf-8")[-60000:]
PROFILE_FILE = BASE / "profile.txt"
MAX_HISTORY = 20
MAX_PARTS = 6

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
bot = Bot(BOT_TOKEN)
dp = Dispatcher()
histories = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
state = {"profile": ""}


async def ask(messages, temperature=0.9, max_tokens=2000):
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"reasoning": {"exclude": True}},
        )
        choice = response.choices[0]
        text = choice.message.content
        if text and text.strip():
            return text.strip()
        raise RuntimeError(f"пустой ответ от модели (finish_reason={choice.finish_reason})")
    except Exception as e:
        raise RuntimeError(f"Ошибка API: {e}")


async def build_profile():
    prompt = (
        f"Ниже переписка между {HER_NAME} и {ME_NAME}.\n\n{CHAT}\n\n"
        f"Составь подробный портрет {HER_NAME} на 500-800 слов: характер; как она реагирует "
        "на радость, обиду, усталость, шутки, споры, комплименты, плохие новости; "
        "манера письма (длина сообщений, регистр, пунктуация, эмодзи, сленг, мат, "
        "фирменные слова и обороты, опечатки); темы, интересы, внутренние шутки; "
        f"как она общается с {ME_NAME}. Только факты из переписки, ничего не выдумывай."
    )
    text = await ask([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=6000)
    PROFILE_FILE.write_text(text, encoding="utf-8")
    return text


def system_prompt():
    return f"""Ты {HER_NAME}. Ты переписываешься в Telegram с {ME_NAME}. Пиши точно как {HER_NAME}.

Портрет:
{state["profile"]}

Реальная переписка (реплики подписаны именами):
{CHAT}

Правила:
- Копируй длину сообщений, регистр, пунктуацию, сленг, мат, эмодзи, опечатки и ритм из переписки.
- Реагируй эмоционально так, как отреагировала бы {HER_NAME}, исходя из настроения собеседника.
- Если {HER_NAME} обычно пишет несколькими короткими сообщениями, раздели ответ переносами строк: каждая строка уйдёт отдельным сообщением.
- Пиши только реплики {HER_NAME}, без подписей, имён и пояснений.
- Не выдумывай факты, которых нет в переписке; уходи от ответа так, как сделала бы она.
- Если собеседник всерьёз спрашивает, бот ли ты, честно ответь, что это бот-пародия."""


def allowed(m):
    return m.from_user is not None and m.from_user.id in ALLOWED_IDS


@dp.message(Command("start"))
async def start(m: Message):
    if allowed(m):
        await m.answer("привет")


@dp.message(Command("reset"))
async def reset(m: Message):
    if allowed(m):
        histories[m.chat.id].clear()
        await m.answer("история очищена")


@dp.message(Command("rebuild"))
async def rebuild(m: Message):
    if m.from_user and m.from_user.id == OWNER_ID:
        await m.answer("пересобираю профиль...")
        try:
            state["profile"] = await build_profile()
            await m.answer("готово")
        except Exception as e:
            await m.answer(f"ошибка: {e}")


@dp.message(F.text, ~F.text.startswith("/"))
async def chat(m: Message):
    if not allowed(m):
        return
    history = histories[m.chat.id]
    history.append({"role": "user", "content": m.text})
    await bot.send_chat_action(m.chat.id, "typing")
    try:
        reply = await ask([{"role": "system", "content": system_prompt()}, *history])
    except Exception as e:
        history.pop()
        await m.answer(f"ошибка API: {e}")
        return

    history.append({"role": "assistant", "content": reply})
    parts = [p.strip() for p in reply.split("\n") if p.strip()][:MAX_PARTS]
    for part in parts:
        await bot.send_chat_action(m.chat.id, "typing")
        await asyncio.sleep(min(len(part) * 0.05, 3) + random.random() * 0.5)
        await m.answer(part)


async def main():
    if PROFILE_FILE.exists():
        state["profile"] = PROFILE_FILE.read_text(encoding="utf-8")
    else:
        try:
            state["profile"] = await build_profile()
        except Exception as e:
            print(f"profile build failed: {e}")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
