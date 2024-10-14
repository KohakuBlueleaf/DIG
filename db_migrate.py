import asyncio
from dig_server.db import database_proxy, Task, initialize_db
from dig_server.server import get_db, lifespan
from tqdm import tqdm


def get_db():
    db = database_proxy.obj
    db.connect(reuse_if_open=True)
    try:
        yield db
    finally:
        if not db.is_closed():
            db.close()


async def main():
    async with lifespan(app=None):
        for db in get_db():
            with db.atomic():
                for task in tqdm(Task.select().where(Task.status == "completed")):
                    image_data = task.image_data
                    if image_data is not None:
                        with open(f"./images/{task.task_id}.webp", "wb") as f:
                            f.write(image_data)
                        task.image_path = f"images/{task.task_id}.webp"
                        task.image_data = None
                    if task.image_path is None:
                        task.image_path = f"images/{task.task_id}.webp"
                    task.save()
            with db.atomic():
                for task in tqdm(Task.select().where(Task.status == "processing")):
                    task.status = "pending"
                    task.image_path = task.image_path = None
                    task.save()


if __name__ == "__main__":
    asyncio.run(main())
