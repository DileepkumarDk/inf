#!/bin/bash

# Quick fix for missing Python.h in conda environment

echo "Installing Python development headers..."
conda install -y -c conda-forge python-dev

echo ""
echo "Verifying installation..."
PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
if [ -f "$PYTHON_INCLUDE/Python.h" ]; then
    echo "✓ Python.h found at: $PYTHON_INCLUDE/Python.h"
    echo ""
    echo "Ready to build! Run: bash build.sh"
else
    echo "✗ Python.h still not found"
    echo "Try: sudo apt-get install python3-dev"
fi
