set -e

echo "Setting up DigitalOcean PostgreSQL MCP Server..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file with your database credentials and configuration."
    exit 1
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv env
source env/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Test database connection
echo "Testing database connection..."
python3 -c "
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

try:
    conn = psycopg2.connect(
        host=os.getenv('DATABASE_HOST'),
        port=int(os.getenv('DATABASE_PORT', 25060)),
        database=os.getenv('DATABASE_NAME'),
        user=os.getenv('DATABASE_USER'),
        password=os.getenv('DATABASE_PASSWORD'),
        sslmode=os.getenv('DATABASE_SSL_MODE', 'require')
    )
    print('✓ Database connection successful!')
    conn.close()
except Exception as e:
    print(f'✗ Database connection failed: {e}')
    exit(1)
"

echo "Setup complete! You can now run the server with:"
echo "  python sample_postgres_mcp.py"
echo ""
echo "Or use Docker:"
echo "  docker-compose up -d"
