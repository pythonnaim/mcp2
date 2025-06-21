import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Sequence

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    TextContent,
    Tool,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

class DatabaseConfig:
    def __init__(self, host: str, port: int, database: str, user: str, password: str, ssl_mode: str = "require"):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.ssl_mode = ssl_mode
        
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}?sslmode={self.ssl_mode}"

class PostgreSQLMCPServer:
    def __init__(self):
        self.server = Server("digitalocean-postgres-mcp")
        self.db_configs = self._load_database_configs()
        self.bearer_token = os.getenv("BEARER_TOKEN")
        
        # Register handlers
        self.server.list_tools = self.list_tools
        self.server.call_tool = self.call_tool

    def _load_database_configs(self) -> Dict[str, DatabaseConfig]:
        """Load database configurations from environment variables."""
        configs = {}
        
        # Primary database
        primary_config = DatabaseConfig(
            host=os.getenv("DATABASE_HOST", ""),
            port=int(os.getenv("DATABASE_PORT", "25060")),
            database=os.getenv("DATABASE_NAME", ""),
            user=os.getenv("DATABASE_USER", "svc_readonly"),
            password=os.getenv("DATABASE_PASSWORD", ""),
            ssl_mode=os.getenv("DATABASE_SSL_MODE", "require")
        )
        configs["primary"] = primary_config
        
        # Additional databases (optional)
        additional_dbs = os.getenv("ADDITIONAL_DATABASES", "")
        if additional_dbs:
            for i, db_config in enumerate(additional_dbs.split(",")):
                try:
                    host, name, user, password = db_config.strip().split(":")
                    configs[f"db_{i+2}"] = DatabaseConfig(
                        host=host,
                        port=25060,  # Standard DigitalOcean port
                        database=name,
                        user=user,
                        password=password
                    )
                except ValueError:
                    logger.warning(f"Invalid database config format: {db_config}")
        
        return configs

    def _get_connection(self, db_name: str = "primary"):
        """Get database connection for specified database."""
        if db_name not in self.db_configs:
            raise ValueError(f"Database '{db_name}' not configured")
        
        config = self.db_configs[db_name]
        return psycopg2.connect(
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.user,
            password=config.password,
            sslmode=config.ssl_mode,
            cursor_factory=psycopg2.extras.RealDictCursor
        )

    async def list_tools(self, request: ListToolsRequest) -> ListToolsResult:
        """List available database tools."""
        tools = [
            Tool(
                name="execute_query",
                description="Execute a SQL query on the specified database. Use for SELECT statements to fetch data safely with a read-only user.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute (SELECT statements recommended)"
                        },
                        "database": {
                            "type": "string",
                            "description": "Database name to query (default: primary)",
                            "default": "primary"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of rows to return (default: 100)",
                            "default": 100,
                            "maximum": 1000
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="fetch_data",
                description="Fetch data using natural language query. Provide a business question and the system will generate and execute appropriate SQL.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Natural language question about the data (e.g., 'Show top 10 customers by MRR')"
                        },
                        "database": {
                            "type": "string",
                            "description": "Database name to query (default: primary)",
                            "default": "primary"
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional context about the data structure or business domain"
                        }
                    },
                    "required": ["question"]
                }
            ),
            Tool(
                name="get_schema",
                description="Get database schema information including tables, columns, and relationships.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "database": {
                            "type": "string",
                            "description": "Database name to inspect (default: primary)",
                            "default": "primary"
                        },
                        "table_name": {
                            "type": "string",
                            "description": "Specific table name to inspect (optional)"
                        }
                    }
                }
            )
        ]
        
        return ListToolsResult(tools=tools)

    async def call_tool(self, request: CallToolRequest) -> CallToolResult:
        """Handle tool calls."""
        try:
            if request.name == "execute_query":
                return await self._execute_query(request.arguments)
            elif request.name == "fetch_data":
                return await self._fetch_data(request.arguments)
            elif request.name == "get_schema":
                return await self._get_schema(request.arguments)
            else:
                raise ValueError(f"Unknown tool: {request.name}")
                
        except Exception as e:
            logger.error(f"Error executing tool {request.name}: {str(e)}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")],
                isError=True
            )

    async def _execute_query(self, args: Dict[str, Any]) -> CallToolResult:
        """Execute a raw SQL query."""
        query = args.get("query", "").strip()
        database = args.get("database", "primary")
        limit = min(args.get("limit", 100), 1000)  # Cap at 1000 rows
        
        if not query:
            raise ValueError("Query cannot be empty")
        
        # Basic safety check - only allow SELECT statements
        if not query.upper().strip().startswith("SELECT"):
            raise ValueError("Only SELECT statements are allowed for security reasons")
        
        # Add limit if not present
        if "LIMIT" not in query.upper():
            query += f" LIMIT {limit}"
        
        with self._get_connection(database) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                
                # Convert to list of dictionaries
                data = [dict(row) for row in results]
                
                response_text = f"Query executed successfully. Found {len(data)} rows.\n\n"
                if data:
                    # Show column headers
                    headers = list(data[0].keys())
                    response_text += "Columns: " + ", ".join(headers) + "\n\n"
                    
                    # Show first few rows as example
                    response_text += "Sample data:\n"
                    for i, row in enumerate(data[:5]):  # Show first 5 rows
                        response_text += f"Row {i+1}: {dict(row)}\n"
                    
                    if len(data) > 5:
                        response_text += f"... and {len(data) - 5} more rows\n"
                    
                    # Include full data as JSON for the AI to process
                    response_text += f"\nComplete results (JSON):\n{json.dumps(data, indent=2, default=str)}"
                else:
                    response_text += "No data found."
        
        return CallToolResult(
            content=[TextContent(type="text", text=response_text)]
        )

    async def _fetch_data(self, args: Dict[str, Any]) -> CallToolResult:
        """Convert natural language to SQL and execute."""
        question = args.get("question", "").strip()
        database = args.get("database", "primary")
        context = args.get("context", "")
        
        if not question:
            raise ValueError("Question cannot be empty")
        
        # First, get schema information to help generate SQL
        schema_info = await self._get_schema({"database": database})
        schema_text = schema_info.content[0].text if schema_info.content else ""
        
        # Generate SQL query based on the natural language question
        # This is a simplified approach - in production, you'd use an LLM here
        sql_query = self._generate_sql_from_question(question, schema_text, context)
        
        # Execute the generated query
        return await self._execute_query({
            "query": sql_query,
            "database": database,
            "limit": 100
        })

    def _generate_sql_from_question(self, question: str, schema: str, context: str) -> str:
        """
        Generate SQL from natural language question.
        This is a basic implementation - in production, integrate with an LLM.
        """
        question_lower = question.lower()
        
        # Basic patterns for common queries
        if "top" in question_lower and "customer" in question_lower:
            if "mrr" in question_lower or "revenue" in question_lower:
                return "SELECT customer_id, customer_name, SUM(amount) as total_revenue FROM orders GROUP BY customer_id, customer_name ORDER BY total_revenue DESC"
        
        elif "count" in question_lower and "order" in question_lower:
            return "SELECT COUNT(*) as total_orders FROM orders"
        
        elif "recent" in question_lower or "latest" in question_lower:
            return "SELECT * FROM orders ORDER BY created_at DESC"
        
        # Default to showing tables if we can't parse the question
        return "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"

    async def _get_schema(self, args: Dict[str, Any]) -> CallToolResult:
        """Get database schema information."""
        database = args.get("database", "primary")
        table_name = args.get("table_name")
        
        with self._get_connection(database) as conn:
            with conn.cursor() as cursor:
                if table_name:
                    # Get specific table schema
                    cursor.execute("""
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_name = %s AND table_schema = 'public'
                        ORDER BY ordinal_position
                    """, (table_name,))
                    columns = cursor.fetchall()
                    
                    response_text = f"Schema for table '{table_name}':\n\n"
                    for col in columns:
                        response_text += f"- {col['column_name']}: {col['data_type']} {'(nullable)' if col['is_nullable'] == 'YES' else '(not null)'}\n"
                        if col['column_default']:
                            response_text += f"  Default: {col['column_default']}\n"
                else:
                    # Get all tables and their column counts
                    cursor.execute("""
                        SELECT t.table_name,
                               COUNT(c.column_name) as column_count,
                               string_agg(c.column_name, ', ' ORDER BY c.ordinal_position) as columns
                        FROM information_schema.tables t
                        LEFT JOIN information_schema.columns c ON t.table_name = c.table_name
                        WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
                        GROUP BY t.table_name
                        ORDER BY t.table_name
                    """)
                    tables = cursor.fetchall()
                    
                    response_text = f"Database schema for '{database}':\n\n"
                    for table in tables:
                        response_text += f"Table: {table['table_name']} ({table['column_count']} columns)\n"
                        response_text += f"Columns: {table['columns']}\n\n"
        
        return CallToolResult(
            content=[TextContent(type="text", text=response_text)]
        )

# FastAPI app for HTTPS + Auth
app = FastAPI(title="DigitalOcean PostgreSQL MCP Server")

# Global MCP server instance
mcp_server = PostgreSQLMCPServer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token."""
    expected_token = os.getenv("BEARER_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: BEARER_TOKEN not set"
        )
    
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "databases": list(mcp_server.db_configs.keys())}

@app.post("/mcp")
async def mcp_endpoint(request: Request, credentials: HTTPAuthorizationCredentials = Depends(verify_token)):
    """Main MCP endpoint with SSE transport."""
    
    async def generate_sse():
        """Generate Server-Sent Events for MCP communication."""
        try:
            # Read the request body
            body = await request.body()
            request_data = json.loads(body.decode())
            
            # Handle MCP requests
            if request_data.get("method") == "tools/list":
                result = await mcp_server.list_tools(ListToolsRequest())
                response = {
                    "jsonrpc": "2.0",
                    "id": request_data.get("id"),
                    "result": {
                        "tools": [tool.model_dump() for tool in result.tools]
                    }
                }
            elif request_data.get("method") == "tools/call":
                params = request_data.get("params", {})
                tool_request = CallToolRequest(
                    name=params.get("name"),
                    arguments=params.get("arguments", {})
                )
                result = await mcp_server.call_tool(tool_request)
                response = {
                    "jsonrpc": "2.0",
                    "id": request_data.get("id"),
                    "result": {
                        "content": [content.model_dump() for content in result.content],
                        "isError": getattr(result, 'isError', False)
                    }
                }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_data.get("id"),
                    "error": {
                        "code": -32601,
                        "message": "Method not found"
                    }
                }
            
            # Send SSE response
            yield f"data: {json.dumps(response)}\n\n"
            
        except Exception as e:
            logger.error(f"Error in SSE handler: {e}")
            error_response = {
                "jsonrpc": "2.0",
                "id": request_data.get("id") if 'request_data' in locals() else None,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
            yield f"data: {json.dumps(error_response)}\n\n"
    
    return StreamingResponse(
        generate_sse(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_SERVER_PORT", "8000"))
    
    logger.info(f"Starting DigitalOcean PostgreSQL MCP Server on {host}:{port}")
    logger.info(f"Configured databases: {list(mcp_server.db_configs.keys())}")
    
    uvicorn.run(app, host=host, port=port)