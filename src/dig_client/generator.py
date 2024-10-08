import asyncio
import io
import random

import httpx
import torch
from PIL import Image

from .diff import load_model, generate, encode_prompts
from .meta import DEFAULT_NEGATIVE_PROMPT
from . import config


client = httpx.AsyncClient(timeout=3600)
pipe = load_model("KBlueLeaf/Kohaku-XL-Zeta", custom_vae=True)


async def get_task():
    while True:
        response = await client.get(f"{config.SERVER_URL}/task")
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            print("No tasks available")
            return None
        elif response.status_code == 409:
            print("Task was taken by another process, retrying...")
            await asyncio.sleep(0.1)  # Wait a bit before retrying
        else:
            print(f"Error getting task: {response.text}")
            return None


def generate_image(prompt: str | list[str] = "", seeds=-1):
    torch.cuda.empty_cache()
    (prompt_embeds, neg_prompt_embeds), (pooled_embeds2, neg_pooled_embeds2) = (
        encode_prompts(
            pipe,
            prompt,
            DEFAULT_NEGATIVE_PROMPT,
            take_all_eos=True,
        )
    )
    torch.cuda.empty_cache()
    result = generate(
        pipe,
        prompt_embeds,
        neg_prompt_embeds,
        pooled_embeds2,
        neg_pooled_embeds2,
        seeds=seeds,
    )
    torch.cuda.empty_cache()
    return result


async def complete_task(task_id: str, image: Image.Image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="WEBP", quality=100, lossless=False)
    img_byte_arr = img_byte_arr.getvalue()

    files = {"image": ("image.webp", img_byte_arr, "image/webp")}
    response = await client.post(f"{config.SERVER_URL}/complete/{task_id}", files=files)
    if response.status_code == 200:
        print(f"Task {task_id} completed successfully")
    else:
        print(f"Error completing task: {response.text}")


async def main():
    while True:
        tasks = []
        for _ in range(16):
            task = await get_task()
            if task is not None:
                tasks.append(task)
            else:
                break
        if tasks:
            print(f"Received task: {tasks}")
            try:
                images = generate_image(
                    [task["prompt"] for task in tasks],
                    [task["extra_args"].get("seeds", -1) for task in tasks],
                )
                await asyncio.gather(
                    *[
                        complete_task(task["task_id"], image)
                        for task, image in zip(tasks, images)
                    ]
                )
            except Exception as e:
                await asyncio.gather(
                    *[
                        client.get(f"{config.SERVER_URL}/reset/{task['task_id']}")
                        for task in tasks
                    ]
                )
                raise e
        else:
            print("No task available, waiting...")
            await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(main())
