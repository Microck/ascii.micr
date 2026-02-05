#!/bin/bash

# ASCII Art Generator Launcher (Linux/Mac)
# =========================================
# 
# This script activates the virtual environment and launches the Streamlit app
# 
# Usage:
#   ./run.sh
#   or
#   bash run.sh

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}          ${GREEN}ASCII.MICR - Local ASCII Art Generator${NC}         ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    python3 -m venv venv
    
    echo -e "${YELLOW}Installing dependencies...${NC}"
    source venv/bin/activate
    pip install --upgrade pip
    pip install torch pillow numpy tqdm streamlit watchdog
else
    echo -e "${GREEN}✓ Virtual environment found${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Check if gradscii-art exists
if [ ! -d "gradscii-art" ]; then
    echo -e "${YELLOW}Cloning gradscii-art repository...${NC}"
    git clone https://github.com/stong/gradscii-art.git
fi

echo ""
echo -e "${GREEN}Starting ASCII Art Generator...${NC}"
echo -e "${BLUE}The application will open in your browser automatically${NC}"
echo ""

# Launch Streamlit
streamlit run app.py --server.port=8501 --server.address=localhost

# Deactivate when done (this won't execute until streamlit stops)
deactivate
