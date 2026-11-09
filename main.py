import asyncio
import os

import httpx
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message

bot = Bot(os.environ["BOT_TOKEN"])
dp = Dispatcher()
OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
OWNER_ID = int(os.environ["OWNER_ID"])
MODEL = os.getenv("MODEL", "google/gemma-4-31b-it:free")
history = []


@dp.message(Command("reset"))
async def reset(message: Message):
    if message.from_user.id == OWNER_ID:
        history.clear()
        await message.answer("Контекст очищен")


@dp.message()
async def chat(message: Message):
    if message.from_user.id != OWNER_ID or not message.text:
        return
    history.append({"role": "user", "content": message.text})
    del history[:-20]
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                json={"model": MODEL, "messages": history},
            )
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        history.pop()
        await message.answer(f"Ошибка: {e}")
        return
    history.append({"role": "assistant", "content": answer})
    await message.answer(answer[:4096])


async def main():
    await dp.start_polling(bot)


asyncio.run(main())
