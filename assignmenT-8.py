import os
import sqlite3
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ---------------------------------------------------------
# 1. Database Setup & Initialization
# ---------------------------------------------------------
DB_FILE = "app.db"


def init_db():
    """Create database tables if they do not exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT
        );
    """
    )
    conn.commit()
    conn.close()


# Initialize table on launch
init_db()


def get_db():
    """Helper to get a database connection with dict-like row output."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------
# 2. FastAPI Application & Routes
# ---------------------------------------------------------
app = FastAPI(
    title="Single-File Render App",
    description="Full FastAPI + SQLite CRUD application ready for Render deployment.",
)


class ItemCreate(BaseModel):
    name: str
    description: Optional[str] = None


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "API is live! Visit /docs for interactive testing.",
    }


@app.get("/items")
def list_items():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM items;")
    rows = cursor.fetchall()
    conn.close()
    return {"items": [dict(row) for row in rows]}


@app.post("/items", status_code=201)
def create_item(item: ItemCreate):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO items (name, description) VALUES (?, ?);",
        (item.name, item.description),
    )
    conn.commit()
    item_id = cursor.lastrowid
    conn.close()
    return {"id": item_id, "name": item.name, "description": item.description}


# ---------------------------------------------------------
# 3. Local Execution Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    # Get port assigned by Render ($PORT env variable) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)