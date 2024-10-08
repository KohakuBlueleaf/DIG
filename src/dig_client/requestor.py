import os
import asyncio
import httpx

import orjsonl
from tqdm import tqdm
from kgen.utils import remove_repeated_suffix
from kgen.formatter import seperate_tags, apply_format

DEFAULT_FORMAT = """<|special|>,
<|characters|>, <|copyrights|>,
<|artist|>,
<|general|>,
<|quality|>, <|meta|>, <|rating|>
"""

from . import config


client = httpx.AsyncClient(timeout=3600, limits=httpx.Limits(max_connections=1024))


async def request_image_generation(prompt: str, task_id: str = None, seed: int = None):
    extra_args = {}
    if task_id is not None:
        extra_args["task_id"] = task_id
    if seed is not None:
        extra_args["seed"] = seed
    response = await client.post(
        f"{config.SERVER_URL}/request",
        json={"prompt": prompt, "extra_args": extra_args},
    )
    if response.status_code == 200:
        task_id = response.json()["task_id"]
        # print(f"Task created with ID: {task_id}")
        return task_id
    else:
        print(f"Error creating task: {response.text}")
        return None


def load_prompts_coyo(file):
    datas = []
    for data in orjsonl.load(file):
        org_data = data["entry"]
        index = org_data["key"]
        result1 = data["result1"]
        result2 = data["result2"]
        org_prompt1 = remove_repeated_suffix(org_data["caption_llava_short"].strip())
        org_prompt2 = ".".join(
            remove_repeated_suffix(org_data["caption_llava"].strip()).split(".")[:2]
        )
        gen_prompt1 = result1["generated"]
        gen_prompt2 = result2["extended"]
        datas.append((index, org_prompt1, org_prompt2, gen_prompt1, gen_prompt2))
    return datas


async def main_coyo():
    global client
    datas = load_prompts_coyo("./data/coyo-output.jsonl")
    tasks = []
    for entry in datas:
        index, org_prompt1, org_prompt2, gen_prompt1, gen_prompt2 = entry
        tasks.append(
            request_image_generation(org_prompt1, f"coyo-{index}-short", int(index))
        )
        tasks.append(
            request_image_generation(org_prompt2, f"coyo-{index}-tlong", int(index))
        )
        tasks.append(
            request_image_generation(gen_prompt1, f"coyo-{index}-tipo-gen", int(index))
        )
        tasks.append(
            request_image_generation(gen_prompt2, f"coyo-{index}-tipo-ext", int(index))
        )
    for batch in tqdm(
        [tasks[i : i + 100] for i in range(0, len(tasks), 100)],
        total=len(tasks) // 100,
        desc="Requesting images",
    ):
        await asyncio.gather(*batch)


def load_prompts_gbc(file):
    datas = []
    for data in orjsonl.load(file):
        org_data = data["entry"]
        index = org_data["index"]
        result1 = data["result1"]
        result2 = data["result2"]
        org_prompt1 = remove_repeated_suffix(org_data["short_caption"].strip())
        org_prompt2 = ".".join(
            remove_repeated_suffix(org_data["detail_caption"].strip()).split(".")[:2]
        )
        gen_prompt1 = result1["generated"]
        gen_prompt2 = result2["extended"]
        datas.append((index, org_prompt1, org_prompt2, gen_prompt1, gen_prompt2))
    return datas


async def main_gbc():
    global client
    datas = load_prompts_gbc("./data/gbc-output.jsonl")
    tasks = []
    for entry in datas:
        index, org_prompt1, org_prompt2, gen_prompt1, gen_prompt2 = entry
        tasks.append(
            request_image_generation(org_prompt1, f"gbc-{index}-short", int(index))
        )
        tasks.append(
            request_image_generation(org_prompt2, f"gbc-{index}-tlong", int(index))
        )
        tasks.append(
            request_image_generation(gen_prompt1, f"gbc-{index}-tipo-gen", int(index))
        )
        tasks.append(
            request_image_generation(gen_prompt2, f"gbc-{index}-tipo-ext", int(index))
        )
    for batch in tqdm(
        [tasks[i : i + 100] for i in range(0, len(tasks), 100)],
        total=len(tasks) // 100,
        desc="Requesting images",
    ):
        await asyncio.gather(*batch)


def load_prompts_dan_scenery(file):
    datas = []
    for data in orjsonl.load(file):
        org_data = data["entry"]
        index = org_data["index"]
        result1 = data.get("result1", None) or data["result"]
        org_tags = seperate_tags(org_data["caption"])
        org_tags["general"] = ["scenery"]
        org_prompt1 = apply_format(org_tags, DEFAULT_FORMAT)
        if isinstance(result1, str):
            gen_prompt1 = result1
        else:
            gen_prompt1 = apply_format(result1, DEFAULT_FORMAT)
        datas.append((index, org_prompt1, gen_prompt1))
    return datas


async def main_dan_scenery():
    global client
    datas = load_prompts_dan_scenery("./data/scenery-output-promptist.jsonl")
    tasks = []
    for entry in datas:
        index, org_prompt1, gen_prompt1 = entry
        # tasks.append(
        #     request_image_generation(org_prompt1, f"dan-scenery-{index}", int(index))
        # )
        tasks.append(
            request_image_generation(gen_prompt1, f"dan-scenery-{index}-promptist", int(index))
        )
    for batch in tqdm(
        [tasks[i : i + 100] for i in range(0, len(tasks), 100)],
        total=len(tasks) // 100,
        desc="Requesting images",
    ):
        await asyncio.gather(*batch)


async def cleanup():
    await client.aclose()


async def main():
    try:
        # await main_coyo()
        # await main_gbc()
        await main_dan_scenery()
    finally:
        await cleanup()


if __name__ == "__main__":
    asyncio.run(main())
