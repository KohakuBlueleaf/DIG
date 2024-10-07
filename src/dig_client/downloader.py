import os
import asyncio
import httpx

import orjsonl
from tqdm import tqdm
from kgen.utils import remove_repeated_suffix


client = httpx.AsyncClient()


async def check_image_status(task_id: str):
    while True:
        try:
            response = await client.get(f"http://localhost:8000/download/{task_id}")
        except httpx.ReadError:
            continue
        if response.status_code == 200:
            # Image is ready
            os.makedirs("download", exist_ok=True)
            filename = f"download/{task_id}.webp"
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"Image downloaded and saved as {filename}")
            break
        elif response.status_code == 404:
            print("Image not ready yet, waiting...")
            await asyncio.sleep(5)  # Wait for 5 seconds before checking again
        else:
            print(f"Error checking image status: {response.text}")
            break


def load_prompts(file):
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


async def main():
    datas = load_prompts("./data/coyo-output.jsonl")
    tasks = []
    for entry in datas:
        index, org_prompt1, org_prompt2, gen_prompt1, gen_prompt2 = entry
        tasks.append(check_image_status(f"{index}-short"))
        tasks.append(check_image_status(f"{index}-tlong"))
        tasks.append(check_image_status(f"{index}-tipo-gen"))
        tasks.append(check_image_status(f"{index}-tipo-ext"))
    await asyncio.gather(*tasks[:16])


if __name__ == "__main__":
    asyncio.run(main())
