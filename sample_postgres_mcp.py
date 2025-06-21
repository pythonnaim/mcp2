import os
import psycopg2
from fastapi import FastAPI
from mcp import MCPToolOutput
from mcp.server.fastmcp import FastMCP
from typing import Optional, Dict, Any, List

# Read DB config from environment
PGHOST = os.getenv("PGHOST")
PGPORT = os.getenv("PGPORT", "25060")
PGDATABASE = os.getenv("PGDATABASE")
PGUSER = os.getenv("PGUSER")
PGPASSWORD = os.getenv("PGPASSWORD")
PGSSLMODE = os.getenv("PGSSLMODE", "require")

# MCP instance
mcp = FastMCP("Postgres MCP Connector")

# Database connection helper
def get_connection():
    return psycopg2.connect(
        host=PGHOST,
        port=PGPORT,
        dbname=PGDATABASE,
        user=PGUSER,
        password=PGPASSWORD,
        sslmode=PGSSLMODE
    )

# Tool: execute_query
@mcp.tool
def execute_query(sql: str) -> MCPToolOutput:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql)
    try:
        rows = cur.fetchall()
        headers = [desc[0] for desc in cur.description]
        results = [dict(zip(headers, row)) for row in rows]
    except psycopg2.ProgrammingError:
        results = []
    cur.close()
    conn.close()
    return {
        "results": results,
        "cursor": None,
        "cursorFinished": True
    }

# Tool: fetch_data
@mcp.tool
def fetch_data(table: str, limit: int = 10) -> Dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor()
    sql = f"SELECT * FROM {table} LIMIT {limit};"
    cur.execute(sql)
    headers = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {
        "contentType": "application/json",
        "content": [dict(zip(headers, row)) for row in rows]
    }

# FastAPI + SSE mount
app = FastAPI()
app.mount("/sse", mcp.sse_app())
