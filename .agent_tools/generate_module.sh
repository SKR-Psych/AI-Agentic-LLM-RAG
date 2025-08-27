#!/bin/bash

# Define possible directories to generate modules in
DIRS=(
  "ai_agentic_core/agents"
  "ai_agentic_core/tools"
  "benchmarks"
  "notebooks"
  "rust_core/src"
  "tests"
)

# Select a random target directory
target_dir=${DIRS[$RANDOM % ${#DIRS[@]}]}
mkdir -p "$target_dir"

# Generate natural-looking filenames and class names
adjectives=("smart" "quick" "robust" "flexible" "lazy" "keen" "silent" "modular")
nouns=("runner" "manager" "resolver" "engine" "router" "handler" "interface" "coordinator")
suffix=$(date +%s)

# Determine file extension and content based on directory
if [[ "$target_dir" == *"rust_core"* ]]; then
  # Rust module
  filename_snake="${adjectives[$RANDOM % ${#adjectives[@]}]}_${nouns[$RANDOM % ${#nouns[@]}]}_$suffix.rs"
  filepath="$target_dir/$filename_snake"
  
  # Create Rust module
  cat << EOF > "$filepath"
//! Auto-generated Rust module

/// Example function
pub fn example() -> String {
    "Hello from Rust module!".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example() {
        assert_eq!(example(), "Hello from Rust module!");
    }
}
EOF

  # Add module to lib.rs if it's a Rust module
  if [[ -f "rust_core/src/lib.rs" ]]; then
    mod_name=$(basename "$filename_snake" .rs);
    echo "pub mod $mod_name;" >> "rust_core/src/lib.rs"
  fi
  
elif [[ "$target_dir" == *"notebooks"* ]]; then
  # Jupyter Notebook
  filename_snake="analysis_$suffix.ipynb"
  filepath="$target_dir/$filename_snake"
  
  # Create a simple Jupyter notebook
  cat << EOF > "$filepath"
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Analysis $(date +'%Y-%m-%d')\n",
    "## Overview\n",
    "This is an auto-generated analysis notebook."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Example code cell\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "print('Hello from Jupyter!')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
EOF

elif [[ "$target_dir" == *"benchmarks"* ]]; then
  # Benchmark file
  filename_snake="benchmark_$suffix.py"
  filepath="$target_dir/$filename_snake"
  
  # Create a benchmark file
  cat << EOF > "$filepath"
"""Performance benchmarks for the system."""
import time
import random

def benchmark_example():
    """Example benchmark function."""
    start_time = time.time()
    
    # Example benchmark logic
    total = 0
    for _ in range(1000000):
        total += random.random()
    
    elapsed = time.time() - start_time
    print(f"Benchmark completed in {elapsed:.4f} seconds")
    return elapsed

if __name__ == "__main__":
    benchmark_example()
EOF

elif [[ "$target_dir" == *"tests"* ]]; then
  # Test file
  filename_snake="test_${adjectives[$RANDOM % ${#adjectives[@]}]}_${nouns[$RANDOM % ${#nouns[@]}]}.py"
  filepath="$target_dir/$filename_snake"
  
  # Create a test file
  cat << EOF > "$filepath"
"""Test module for verifying functionality."""
import pytest

def test_example():
    """Example test case."""
    assert 1 + 1 == 2

class TestFeature:
    """Test class for feature verification."""
    
    def test_another_example(self):
        """Another test case."""
        assert "hello".upper() == "HELLO"
EOF

else
  # Default Python module
  filename_snake="${adjectives[$RANDOM % ${#adjectives[@]}]}_${nouns[$RANDOM % ${#nouns[@]}]}_$suffix.py"
  filepath="$target_dir/$filename_snake"
  
  # Create Python module file with class stub
  cat << EOF > "$filepath"
"""Module for core functionality."""

class ${filename_snake%.py^}:
    """Module for core agent functionality."""

    def __init__(self):
        self.active = False

    def start(self):
        """Initialize or trigger the module."""
        self.active = True

    def process(self, input_data):
        """Main logic to handle incoming data."""
        print(f"Processing: {input_data}")
        return {"status": "success", "data": input_data}


if __name__ == "__main__":
    # Example usage
    module = ${filename_snake%.py^}()
    module.start()
    result = module.process("test data")
    print(f"Result: {result}")
EOF
fi

echo "Created module: $filepath"

# Git add, commit and push if file is new
git add --intent-to-add "$filepath"

if git diff --cached --quiet; then
    echo "No new file changes to commit."
else
    # Use consistent Git identity
    git config --global user.name "Agentic Bot"
    git config --global user.email "agent@llmrag.com"

    # Generate appropriate commit message based on file type
    filename=$(basename "$filepath")
    if [[ "$filename" == test_* ]]; then
        commit_msg="🧪 Add test: ${filename%.py}"
    elif [[ "$filename" == benchmark_* ]]; then
        commit_msg="📊 Add benchmark: ${filename%.py}"
    elif [[ "$filename" == *.ipynb ]]; then
        commit_msg="docs: add analysis notebook ${filename%.ipynb}"
    elif [[ "$filename" == *.rs ]]; then
        commit_msg="feat(rust): add module ${filename%.rs}"
    else
        commit_msg="feat: add module ${filename%.py}"
    fi

    git commit -m "$commit_msg"
    git push origin main
    echo "Pushed: $filename_snake"
fi
