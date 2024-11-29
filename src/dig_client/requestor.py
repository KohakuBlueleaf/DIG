import httpx

from . import config


client = httpx.AsyncClient(timeout=3600, limits=httpx.Limits(max_connections=512))


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


async def cleanup():
    await client.aclose()
