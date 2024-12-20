from peewee import *
import datetime
import os

database_proxy = DatabaseProxy()


class BaseModel(Model):
    class Meta:
        database = database_proxy


class Task(BaseModel):
    task_id = CharField(unique=True, index=True)
    prompt = TextField()
    extra_args = TextField(null=True)
    image_path = CharField(null=True)
    status = CharField(default="pending")  # pending, processing, completed
    created_at = DateTimeField(default=datetime.datetime.now)


def initialize_db(db_path="db/image_tasks.db"):
    database = SqliteDatabase(
        db_path,
        pragmas={
            "journal_mode": "wal",
            "cache_size": -1024 * 256,  # 256MB cache
            "mmap_size": 1024 * 1024 * 1024,  # 1GB mmap
            "synchronous": "normal",
            "temp_store": "memory",
            "foreign_keys": 1,
            "ignore_check_constraints": 0,
        },
    )
    database_proxy.initialize(database)


def create_tables():
    with database_proxy.obj:
        database_proxy.obj.create_tables([Task], safe=True)


if __name__ == "__main__":
    db_path = os.environ.get("DB_PATH", "db/image_tasks.db")
    directory = os.path.dirname(db_path)
    os.makedirs(directory, exist_ok=True)
    initialize_db(db_path)
    create_tables()
    print("Database initialized and tables created.")
