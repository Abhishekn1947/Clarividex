#!/bin/bash

# =============================================================================
# Future Prediction AI - Setup Script
# =============================================================================

set -e

echo "🚀 Setting up Future Prediction AI..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python3 --version

# Check Node version
echo -e "${YELLOW}Checking Node.js version...${NC}"
node --version

# Create virtual environment
echo -e "${YELLOW}Creating Python virtual environment...${NC}"
cd backend
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Go back to root
cd ..

# Install frontend dependencies
echo -e "${YELLOW}Installing frontend dependencies...${NC}"
cd frontend
npm install
cd ..

# Create .env if not exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cp .env.example .env
    echo -e "${RED}⚠️  Please edit .env and add your API keys${NC}"
fi

# Create data directories
echo -e "${YELLOW}Creating data directories...${NC}"
mkdir -p data/cache data/predictions data/accuracy

echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your ANTHROPIC_API_KEY"
echo "2. Run: ./scripts/start.sh"
echo ""
echo "Optional: Install Ollama for offline AI fallback:"
echo "  - Download from https://ollama.ai"
echo "  - Run: ollama pull llama3.2:3b"
