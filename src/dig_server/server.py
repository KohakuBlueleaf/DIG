import os
import json
import io
from uuid import uuid4
from contextlib import asynccontextmanager
from typing import Optional

from PIL import Image
from pydantic import BaseModel, Field
from peewee import fn, SqliteDatabase, IntegrityError
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import Response

from .db import database_proxy, Task, initialize_db


class PromptRequest(BaseModel):
    prompt: str
    extra_args: Optional[dict[str, int | float | str | bool]] = Field(
        default_factory=dict
    )


class TaskResponse(BaseModel):
    task_id: str


class TaskRequest(BaseModel):
    task_id: str
    prompt: str
    extra_args: dict[str, int | float | str | bool] = Field(default_factory=dict)


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
    db_path = os.environ.get("DB_PATH", "db/image_tasks.db")
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
    if "task_id" in prompt_request.extra_args:
        task_id = prompt_request.extra_args.pop("task_id")
    else:
        task_id = str(uuid4())
    with db.atomic():
        if Task.select().where(Task.task_id == task_id).exists():
            prev_task = Task.get(Task.task_id == task_id)
            prev_task.status = "pending"
            prev_task.prompt = prompt_request.prompt
            prev_task.extra_args = json.dumps(
                prompt_request.extra_args, ensure_ascii=False
            )
            prev_task.save()
        else:
            Task.create(
                task_id=task_id,
                prompt=prompt_request.prompt,
                extra_args=json.dumps(prompt_request.extra_args, ensure_ascii=False),
            )
    return TaskResponse(task_id=task_id)


@app.get("/reset/{task_id}")
async def reset_task(task_id: str, db: SqliteDatabase = Depends(get_db)):
    with db.atomic():
        task = Task.get(Task.task_id == task_id)
        task.status = "pending"
        task.save()
    return {"message": "Task reset successfully"}


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
            return TaskRequest(
                task_id=task.task_id,
                prompt=task.prompt,
                extra_args=json.loads(task.extra_args or "{}"),
            )
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
                status_code=400,
                detail="Task is not in processing state: " + task.status,
            )

        image.file.seek(0)
        img = Image.open(image.file)
        result = io.BytesIO()
        img.save(result, format="WEBP")

        task.status = "completed"
        task.image_data = result.getvalue()
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

    if not task.image_data:
        raise HTTPException(status_code=404, detail="Image data not found")

    return Response(content=task.image_data, media_type="image/webp")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
