import os
import asyncio
import httpx

import orjsonl
from tqdm import tqdm
from kgen.utils import remove_repeated_suffix

client = httpx.AsyncClient()


async def request_image_generation(prompt: str, task_id: str = None, seed: int = None):
    extra_args = {}
    if task_id is not None:
        extra_args["task_id"] = task_id
    if seed is not None:
        extra_args["seed"] = seed
    response = await client.post(
        "http://localhost:8000/request",
        json={"prompt": prompt, "extra_args": extra_args},
    )
    if response.status_code == 200:
        task_id = response.json()["task_id"]
        print(f"Task created with ID: {task_id}")
        return task_id
    else:
        print(f"Error creating task: {response.text}")
        return None


async def check_image_status(task_id: str):
    while True:
        try:
            response = await client.get(f"http://localhost:8000/download/{task_id}")
        except httpx.ReadError:
            continue
        if response.status_code == 200:
            # Image is ready
            os.makedirs("download", exist_ok=True)
            filename = f"download/downloaded_{task_id}.webp"
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


async def test():
    prompt = "A beautiful sunset over the ocean"
    tasks = [request_image_generation(prompt) for _ in range(16)]
    tasks = await asyncio.gather(*tasks)
    finish_tasks = [check_image_status(task_id) for task_id in tasks if task_id]
    await asyncio.gather(*finish_tasks)


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
        tasks.append(
            request_image_generation(org_prompt1, f"{index}-short", int(index))
        )
        tasks.append(
            request_image_generation(org_prompt2, f"{index}-tlong", int(index))
        )
        tasks.append(
            request_image_generation(gen_prompt1, f"{index}-tipo-gen", int(index))
        )
        tasks.append(
            request_image_generation(gen_prompt2, f"{index}-tipo-ext", int(index))
        )
    await asyncio.gather(*tasks[:2000])


if __name__ == "__main__":
    asyncio.run(main())
