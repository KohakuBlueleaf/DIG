import os
import asyncio
import httpx


client = httpx.AsyncClient()


async def request_image_generation(prompt: str):
    response = await client.post(
        "http://localhost:8000/request", json={"prompt": prompt}
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


async def main():
    prompt = "A beautiful sunset over the ocean"
    tasks = [request_image_generation(prompt) for _ in range(16)]
    tasks = await asyncio.gather(*tasks)
    finish_tasks = [check_image_status(task_id) for task_id in tasks if task_id]
    await asyncio.gather(*finish_tasks)


if __name__ == "__main__":
    asyncio.run(main())
