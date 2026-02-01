#!/bin/bash

# =============================================================================
# Install Ollama for Offline AI Model Support
# =============================================================================

set -e

echo "🤖 Installing Ollama for offline AI support..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if Ollama is already installed
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}Ollama is already installed!${NC}"
else
    echo -e "${YELLOW}Installing Ollama...${NC}"

    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "Detected macOS"
        brew install ollama || {
            echo "Homebrew not found. Please install Ollama manually from https://ollama.ai"
            exit 1
        }
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "Detected Linux"
        curl -fsSL https://ollama.ai/install.sh | sh
    else
        echo "Unsupported OS. Please install Ollama manually from https://ollama.ai"
        exit 1
    fi
fi

# Start Ollama service (if not running)
echo -e "${YELLOW}Starting Ollama service...${NC}"
ollama serve &
sleep 2

# Pull recommended model
echo -e "${YELLOW}Pulling recommended model (llama3.2:3b)...${NC}"
echo "This may take a few minutes on first download..."
ollama pull llama3.2:3b

echo ""
echo -e "${GREEN}✅ Ollama setup complete!${NC}"
echo ""
echo "Installed model: llama3.2:3b (2GB, fast inference)"
echo ""
echo "The app will automatically use Ollama when Claude API is unavailable."
echo ""
echo "To pull additional models:"
echo "  ollama pull mistral        # 7B model, better quality"
echo "  ollama pull llama3.1:8b    # 8B model, even better"
