import asyncio
from dig_server.db import database_proxy, Task, initialize_db
from dig_server.server import get_db, lifespan


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
                finished_tasks = Task.select().where(Task.status == "completed")
                waiting_tasks = Task.select().where(Task.status == "pending")
                processing_tasks = Task.select().where(Task.status == "processing")
                print(len(waiting_tasks), len(finished_tasks), len(processing_tasks))
                progress = len(finished_tasks) / (
                    len(waiting_tasks) + len(processing_tasks) + len(finished_tasks)
                )
                print(f"Progress: {progress*100:.2f}%")


if __name__ == "__main__":
    asyncio.run(main())
