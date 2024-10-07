import os
import shutil
from uuid import uuid4
from contextlib import asynccontextmanager

from pydantic import BaseModel
from peewee import fn, SqliteDatabase, IntegrityError
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import FileResponse

from .db import database_proxy, Task, initialize_db


class PromptRequest(BaseModel):
    prompt: str


class TaskResponse(BaseModel):
    task_id: str


class TaskRequest(BaseModel):
    task_id: str
    prompt: str


def get_db():
    db = database_proxy.obj
    db.connect(reuse_if_open=True)
    try:
        yield db
    finally:
        if not db.is_closed():
            db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    db_path = os.environ.get("DB_PATH", "image_tasks.db")
    initialize_db(db_path)
    yield
    # Shutdown
    if not database_proxy.obj.is_closed():
        database_proxy.obj.close()


app = FastAPI(lifespan=lifespan)


@app.post("/request", response_model=TaskResponse)
async def create_task(
    prompt_request: PromptRequest, db: SqliteDatabase = Depends(get_db)
):
    task_id = str(uuid4())
    with db.atomic():
        Task.create(task_id=task_id, prompt=prompt_request.prompt)
    return TaskResponse(task_id=task_id)


@app.get("/task", response_model=TaskRequest)
async def get_task(db: SqliteDatabase = Depends(get_db)):
    with db.atomic() as transaction:
        try:
            task = (
                Task.select()
                .where(Task.status == "pending")
                .order_by(Task.created_at)
                .get()
            )
            task.status = "processing"
            task.save()
            return TaskRequest(task_id=task.task_id, prompt=task.prompt)
        except Task.DoesNotExist:
            raise HTTPException(status_code=404, detail="No pending tasks available")
        except IntegrityError:
            # Another process might have taken the task, rollback and try again
            transaction.rollback()
            raise HTTPException(
                status_code=409,
                detail="Task was taken by another process, please try again",
            )


@app.post("/complete/{task_id}")
async def complete_task(
    task_id: str, image: UploadFile = File(...), db: SqliteDatabase = Depends(get_db)
):
    with db.atomic():
        task = Task.select().where(Task.task_id == task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        if task.status != "processing":
            raise HTTPException(
                status_code=400, detail="Task is not in processing state"
            )

        # Save the uploaded image
        image_path = f"images/{task_id}.png"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        task.status = "completed"
        task.image_path = image_path
        task.save()

    return {"message": "Task completed successfully"}


@app.get("/download/{task_id}")
async def download_image(task_id: str, db: SqliteDatabase = Depends(get_db)):
    with db.atomic():
        task = Task.get_or_none(Task.task_id == task_id)
    if not task or task.status != "completed":
        raise HTTPException(
            status_code=404, detail="Image not found or task not completed"
        )

    return FileResponse(
        task.image_path, media_type="image/png", filename=f"{task_id}.png"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
