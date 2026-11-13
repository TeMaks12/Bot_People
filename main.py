import asyncio
import json
import os
import random
from collections import defaultdict, deque
from contextlib import suppress
from pathlib import Path

from aiogram import BaseMiddleware, Bot, Dispatcher, F
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

PROVIDER = "klinch"  # "openrouter" или "klinch"
KLINCH_BASE_URL = "https://klinch.cloud/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

BOT_TOKEN = os.environ["BOT_TOKEN"]
OWNER_ID = int(os.environ["OWNER_ID"])
HER_NAME = os.environ["HER_NAME"]
ME_NAME = os.environ["ME_NAME"]

if PROVIDER not in ("openrouter", "klinch"):
    raise RuntimeError("PROVIDER должен быть openrouter или klinch")

MODELS_LIST = [
    "deepseek-flash",
    "deepseek-v4-pro",
    "claude-sonnet-5-5",
    "glm-5",
    "grok-4.5",
    "gpt-5.4-mini",
    "kimi-k2",
    "mistral-medium",
    "gemini-3.5-flash",
]
DEFAULT_MODEL = "deepseek-flash"

if PROVIDER == "klinch":
    BASE_URL = KLINCH_BASE_URL
    KEY_PREFIX = "KLINCH_API_KEY"
else:
    BASE_URL = OPENROUTER_BASE_URL
    KEY_PREFIX = "OPENROUTER_API_KEY"

DENIED_TEXT = "Ты не можешь пользоваться ботом"

BASE = Path(__file__).parent
CHAT = (BASE / "messages.txt").read_text(encoding="utf-8")[-15000:]
PROFILE_FILE = BASE / "profile.txt"
DATA_FILE = BASE / "data.json"
MAX_HISTORY = 20
MAX_PARTS = 6
DEBOUNCE = 2.5

REMINDER = (
    f"Напоминание: ты {HER_NAME}. Следи за ходом диалога. "
    "Если ты уже о чём-то спросила и собеседник ответил, не спрашивай это снова. "
    "Не повторяй свои прошлые реплики. Не заканчивай каждое сообщение вопросом «а ты?». "
    "Отвечай по смыслу на последнее сообщение собеседника, развивай тему или предложи свою."
    "Не рассказывай про своё состояние и дела из старой переписки, пока тебя не спросили о конкретном. "
)


def load_api_keys():
    keys = []
    i = 1
    while os.getenv(f"{KEY_PREFIX}_{i}"):
        keys.append(os.environ[f"{KEY_PREFIX}_{i}"].strip())
        i += 1
    return keys


API_KEYS = load_api_keys()
if not API_KEYS:
    raise RuntimeError(f"Не найдено ни одного {KEY_PREFIX}_1, {KEY_PREFIX}_2, ...")

clients = [AsyncOpenAI(base_url=BASE_URL, api_key=k) for k in API_KEYS]

data = json.loads(DATA_FILE.read_text(encoding="utf-8")) if DATA_FILE.exists() else {}
allowed_ids = {int(x) for x in data.get("users", [])}

if PROVIDER == "klinch":
    saved_model = data.get("model", DEFAULT_MODEL)
    start_model = saved_model if saved_model in MODELS_LIST else DEFAULT_MODEL
else:
    start_model = os.getenv("MODEL", "nvidia/nemotron-3.5-super-vl:free")

state = {
    "profile": "",
    "key": min(int(data.get("key", 0)), len(clients) - 1),
    "model": start_model,
    "username": "",
    "bot_id": 0,
}


def save_data():
    DATA_FILE.write_text(
        json.dumps({"users": sorted(allowed_ids), "key": state["key"], "model": state["model"]}),
        encoding="utf-8",
    )


bot = Bot(BOT_TOKEN)
dp = Dispatcher()
histories = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
pending = defaultdict(list)
locks = defaultdict(asyncio.Lock)


def is_group(m: Message):
    return m.chat.type != "private"


def extract_prompt(m: Message):
    text = (m.text or "").strip()
    if text.startswith("<"):
        return text[1:].strip()
    mention = f"@{state['username']}".lower()
    if text.lower().startswith(mention):
        return text[len(mention):].lstrip(" ,:").strip()
    reply = m.reply_to_message
    if reply and reply.from_user and reply.from_user.id == state["bot_id"]:
        return text
    return None


class AccessMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, event_data):
        user = event.from_user
        if user and (user.id == OWNER_ID or user.id in allowed_ids):
            return await handler(event, event_data)
        if isinstance(event, CallbackQuery):
            await event.answer(DENIED_TEXT, show_alert=True)
        elif not is_group(event) or extract_prompt(event) is not None:
            await event.answer(DENIED_TEXT)


dp.message.outer_middleware(AccessMiddleware())
dp.callback_query.outer_middleware(AccessMiddleware())

is_owner = F.from_user.id == OWNER_ID


async def ask(messages, temperature=0.7, max_tokens=3000):
    try:
        params = {
            "model": state["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if PROVIDER == "openrouter":
            params["extra_body"] = {"reasoning": {"exclude": True}}
        response = await clients[state["key"]].chat.completions.create(**params)
        choice = response.choices[0]
        text = choice.message.content
        if text and text.strip():
            return text.strip()
        raise RuntimeError(f"пустой ответ от модели (finish_reason={choice.finish_reason})")
    except Exception as e:
        raise RuntimeError(f"Ошибка API (ключ {state['key'] + 1}): {e}")


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
 - Переписка выше нужна только как образец стиля. События и факты из неё (учёба, билеты, мама, ссора и т.п.) считай прошлыми: не пересказывай их и не говори о них как о текущих, если собеседник сам не заговорил.
Отвечай только на то, что собеседник реально написал. Не отвечай на вопросы, которых он не задавал. На пример если пользователь написал "Привет", то можешь написать также приветствие, и спросить и начать какую-то тему для разговора. 

Портрет:
{state["profile"]}

Реальная переписка (реплики подписаны именами):
{CHAT}

Правила:
- Копируй длину сообщений, регистр, пунктуацию, сленг, мат, эмодзи, опечатки и ритм из переписки.
- Реагируй эмоционально так, как отреагировала бы {HER_NAME}, исходя из настроения собеседника.
- Следи за ходом диалога: если ты уже о чём-то спросила и собеседник ответил, не спрашивай это снова.
- Не повторяй свои прошлые реплики и не заканчивай каждое сообщение вопросом.
- Отвечай по смыслу на последнее сообщение собеседника, развивай тему или предлагай свою.
- Если {HER_NAME} обычно пишет несколькими короткими сообщениями, раздели ответ переносами строк: каждая строка уйдёт отдельным сообщением.
- Пиши только реплики {HER_NAME}, без подписей, имён и пояснений.
- Не выдумывай факты, которых нет в переписке; уходи от ответа так, как сделала бы она.
- Если собеседник всерьёз спрашивает, бот ли ты, честно ответь, что это бот-пародия."""


def key_text():
    return f"Текущий ключ: {state['key'] + 1} из {len(clients)}"


def key_markup():
    buttons = [
        InlineKeyboardButton(
            text=f"✅ {i + 1}" if i == state["key"] else str(i + 1),
            callback_data=f"key:{i}",
        )
        for i in range(len(clients))
    ]
    return InlineKeyboardMarkup(inline_keyboard=[buttons[i:i + 5] for i in range(0, len(buttons), 5)])


def model_text():
    return f"Текущая модель: {state['model']}"


def model_markup():
    buttons = [
        [
            InlineKeyboardButton(
                text=f"✅ {name}" if name == state["model"] else name,
                callback_data=f"model:{i}",
            )
        ]
        for i, name in enumerate(MODELS_LIST)
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


@dp.message(Command("start"))
async def start(m: Message):
    await m.answer("привет.")


@dp.message(Command("help"))
async def help_command(m: Message):
    if m.from_user.id == OWNER_ID:
        text = (
            "/start — начать\n"
            "/reset — очистить историю\n"
            "/help — список команд\n"
            "/rebuild — пересобрать профиль\n"
            "/key — выбрать API-ключ\n"
            "/model — выбрать модель\n"
            "/adduser <id> — добавить пользователя\n"
            "/deluser <id> — удалить пользователя\n"
            "/users — список пользователей"
        )
    else:
        text = (
            "/start — начать\n"
            "/reset — очистить историю\n"
            "/help — список команд"
        )
    text += "\n\nВ группах отвечает только на сообщения, начинающиеся с <, или на пинг"
    await m.answer(text)


@dp.message(Command("reset"))
async def reset(m: Message):
    histories[m.chat.id].clear()
    await m.answer("история очищена")


@dp.message(Command("rebuild"), is_owner)
async def rebuild(m: Message):
    await m.answer("пересобираю профиль...")
    try:
        state["profile"] = await build_profile()
        await m.answer("готово")
    except Exception as e:
        await m.answer(f"ошибка: {e}")


@dp.message(Command("key"), is_owner)
async def key_menu(m: Message):
    await m.answer(key_text(), reply_markup=key_markup())


@dp.callback_query(F.data.startswith("key:"), is_owner)
async def key_select(c: CallbackQuery):
    index = int(c.data.split(":")[1])
    if index >= len(clients):
        await c.answer("Нет такого ключа")
        return
    state["key"] = index
    save_data()
    with suppress(TelegramBadRequest):
        await c.message.edit_text(key_text(), reply_markup=key_markup())
    await c.answer(f"Ключ {index + 1}")


@dp.message(Command("model"), is_owner)
async def model_menu(m: Message):
    if PROVIDER != "klinch":
        await m.answer(f"Модель задаётся в конфиге: {state['model']}")
        return
    await m.answer(model_text(), reply_markup=model_markup())


@dp.callback_query(F.data.startswith("model:"), is_owner)
async def model_select(c: CallbackQuery):
    index = int(c.data.split(":")[1])
    if PROVIDER != "klinch" or index >= len(MODELS_LIST):
        await c.answer("Недоступно")
        return
    state["model"] = MODELS_LIST[index]
    save_data()
    with suppress(TelegramBadRequest):
        await c.message.edit_text(model_text(), reply_markup=model_markup())
    await c.answer(state["model"])


@dp.message(Command("adduser"), is_owner)
async def add_user(m: Message, command: CommandObject):
    arg = (command.args or "").strip()
    if not arg.isdigit():
        await m.answer("/adduser <id>")
        return
    allowed_ids.add(int(arg))
    save_data()
    await m.answer(f"добавлен: {arg}")


@dp.message(Command("deluser"), is_owner)
async def del_user(m: Message, command: CommandObject):
    arg = (command.args or "").strip()
    if not arg.isdigit():
        await m.answer("/deluser <id>")
        return
    allowed_ids.discard(int(arg))
    save_data()
    await m.answer(f"удалён: {arg}")


@dp.message(Command("users"), is_owner)
async def list_users(m: Message):
    lines = [f"{OWNER_ID} (owner)"] + [str(i) for i in sorted(allowed_ids)]
    await m.answer("\n".join(lines))


@dp.message(F.text, ~F.text.startswith("/"))
async def chat(m: Message):
    text = m.text
    group = is_group(m)
    if group:
        text = extract_prompt(m)
        if not text:
            return
    send = m.reply if group else m.answer
    pending[m.chat.id].append(text)
    await asyncio.sleep(DEBOUNCE)
    async with locks[m.chat.id]:
        texts = pending.pop(m.chat.id, [])
        if not texts:
            return
        history = histories[m.chat.id]
        history.append({"role": "user", "content": "\n".join(texts)})
        await bot.send_chat_action(m.chat.id, "typing")
        try:
            reply = await ask(
                [
                    {"role": "system", "content": system_prompt()},
                    *history,
                    {"role": "system", "content": REMINDER},
                ]
            )
        except Exception as e:
            history.pop()
            await send(f"ошибка API: {e}")
            return

        history.append({"role": "assistant", "content": reply})
        parts = [p.strip() for p in reply.split("\n") if p.strip()][:MAX_PARTS]
        for part in parts:
            await bot.send_chat_action(m.chat.id, "typing")
            await asyncio.sleep(min(len(part) * 0.05, 3) + random.random() * 0.5)
            await send(part)


async def main():
    me = await bot.get_me()
    state["username"] = me.username
    state["bot_id"] = me.id
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
