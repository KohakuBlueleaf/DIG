import asyncio
import httpx
from PIL import Image
import io
import random

client = httpx.AsyncClient()

async def get_task():
    while True:
        response = await client.get("http://localhost:8000/task")
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            print("No tasks available")
            return None
        elif response.status_code == 409:
            print("Task was taken by another process, retrying...")
            await asyncio.sleep(1)  # Wait a bit before retrying
        else:
            print(f"Error getting task: {response.text}")
            return None


def generate_random_image(size=(256, 256)):
    image = Image.new("RGB", size)
    pixels = image.load()
    for i in range(size[0]):
        for j in range(size[1]):
            pixels[i, j] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
    return image


async def complete_task(task_id: str, image: Image.Image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    files = {"image": ("image.png", img_byte_arr, "image/png")}
    response = await client.post(
        f"http://localhost:8000/complete/{task_id}", files=files
    )
    if response.status_code == 200:
        print(f"Task {task_id} completed successfully")
    else:
        print(f"Error completing task: {response.text}")


async def main():
    while True:
        task = await get_task()
        if task:
            print(f"Received task: {task}")
            image = generate_random_image()
            await complete_task(task["task_id"], image)
        else:
            print("No task available, waiting...")
            await asyncio.sleep(5)  # Wait for 5 seconds before checking for new tasks


if __name__ == "__main__":
    asyncio.run(main())
