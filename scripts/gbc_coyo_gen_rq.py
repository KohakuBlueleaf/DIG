import asyncio

import orjsonl
from tqdm import tqdm
from kgen.utils import remove_repeated_suffix
from kgen.formatter import seperate_tags, apply_format

import dig_client.config as config

config.SERVER_URL = "http://192.168.1.2:21224"
from dig_client.requestor import request_image_generation, cleanup


DEFAULT_FORMAT = """<|special|>,
<|characters|>, <|copyrights|>,
<|artist|>,
<|general|>,
<|quality|>, <|meta|>, <|rating|>
"""
BATCH_SIZE = 100
ALL_POSTFIX = ["", "oai", "promptist", "gpt2"]
ALL_CATE = ["coyo", "gbc"]
ALL_SHORT_KEY = ["caption_llava_short", "short_caption"]
ALL_LONG_KEY = ["caption_llava", "detail_caption"]
SHORT_KEY = "caption_llava_short"
LONG_KEY = "caption_llava"
CATE = "coyo"
POSTFIX = ""


def load_prompts(file):
    datas = []
    for data in orjsonl.load(file):
        org_data = data["entry"]
        index = org_data["index" if CATE == "gbc" else "key"]
        result1 = data["result1"]
        result2 = data["result2"]
        org_prompt1 = remove_repeated_suffix(org_data[SHORT_KEY].strip())
        org_prompt2 = ".".join(
            remove_repeated_suffix(org_data[LONG_KEY].strip()).split(".")[:2]
        )
        gen_prompt1 = result1["generated"] if isinstance(result1, dict) else result1
        gen_prompt2 = result2["extended"] if isinstance(result2, dict) else result2
        datas.append((index, org_prompt1, org_prompt2, gen_prompt1, gen_prompt2))
    return datas


async def main_gbc_coyo():
    global client
    datas = load_prompts(f"./data/{CATE}-output.jsonl")
    tasks = []
    for entry in datas:
        index, org_prompt1, org_prompt2, gen_prompt1, gen_prompt2 = entry
        tasks.append(
            request_image_generation(org_prompt1, f"{CATE}-{index}-short", int(index))
        )
        tasks.append(
            request_image_generation(org_prompt2, f"{CATE}-{index}-tlong", int(index))
        )
        tasks.append(
            request_image_generation(
                gen_prompt1, f"{CATE}-{index}-tipo-short", int(index)
            )
        )
        tasks.append(
            request_image_generation(
                gen_prompt2, f"{CATE}-{index}-tipo-tlong", int(index)
            )
        )
    for batch in tqdm(
        [tasks[i : i + BATCH_SIZE] for i in range(0, len(tasks), BATCH_SIZE)],
        total=len(tasks) // BATCH_SIZE + int(bool(len(tasks) % BATCH_SIZE)),
        desc="Requesting images",
    ):
        await asyncio.gather(*batch)


async def main_gbc_coyo_other():
    global client
    datas = load_prompts(f"./data/{CATE}-output-{POSTFIX}.jsonl")
    tasks = []
    for entry in datas:
        index, org_prompt1, org_prompt2, gen_prompt1, gen_prompt2 = entry
        tasks.append(
            request_image_generation(
                gen_prompt1, f"{CATE}-{index}-{POSTFIX}-short", int(index)
            )
        )
        tasks.append(
            request_image_generation(
                gen_prompt2, f"{CATE}-{index}-{POSTFIX}-tlong", int(index)
            )
        )
    for batch in tqdm(
        [tasks[i : i + BATCH_SIZE] for i in range(0, len(tasks), BATCH_SIZE)],
        total=len(tasks) // BATCH_SIZE + int(bool(len(tasks) % BATCH_SIZE)),
        desc="Requesting images",
        leave=False,
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
            request_image_generation(
                gen_prompt1, f"dan-scenery-{index}-promptist", int(index)
            )
        )
    for batch in tqdm(
        [tasks[i : i + 100] for i in range(0, len(tasks), 100)],
        total=len(tasks) // 100,
        desc="Requesting images",
        leave=False,
    ):
        await asyncio.gather(*batch)


async def main():
    global POSTFIX, CATE, SHORT_KEY, LONG_KEY
    is_first = True
    try:
        for POSTFIX in tqdm(ALL_POSTFIX):
            if is_first:
                is_first = False
                continue
            for CATE, SHORT_KEY, LONG_KEY in tqdm(
                zip(ALL_CATE, ALL_SHORT_KEY, ALL_LONG_KEY),
                total=len(ALL_CATE),
                leave=False,
            ):
                if POSTFIX:
                    await main_gbc_coyo_other()
                else:
                    await main_gbc_coyo()
        # await main_dan_scenery()
    finally:
        await cleanup()


if __name__ == "__main__":
    asyncio.run(main())
