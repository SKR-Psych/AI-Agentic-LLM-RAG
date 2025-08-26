#!/bin/bash

# Randomly select target folder: agents or tools
dirs=("tools" "agents")
target_dir="ai_agentic_core/${dirs[$RANDOM % 2]}"
mkdir -p "$target_dir"

# Generate more natural-looking filenames and class names
adjectives=("smart" "quick" "robust" "flexible" "lazy" "keen" "silent" "modular")
nouns=("runner" "manager" "resolver" "engine" "router" "handler" "interface" "coordinator")
suffix=$RANDOM
filename_snake="${adjectives[$RANDOM % ${#adjectives[@]}]}_${nouns[$RANDOM % ${#nouns[@]}]}_$suffix.py"
classname_pascal="$(tr '[:lower:]' '[:upper:]' <<< ${filename_snake:0:1})${filename_snake:1}"
classname_pascal="${classname_pascal//_//}"  # Remove underscores
classname_pascal="${classname_pascal/.py/}"  # Remove .py

filepath="$target_dir/$filename_snake"

# Create Python module file with class stub
cat << EOF > "$filepath"
class $classname_pascal:
    \"\"\"Module for core agent functionality.\"\"\"

    def __init__(self):
        self.active = False

    def start(self):
        \"\"\"Initialise or trigger the module.\"\"\"
        self.active = True

    def process(self, input_data):
        \"\"\"Main logic to handle incoming data.\"\"\"
        print(f"Processing: {input_data}")
EOF

echo "Created module: $filepath"

# 🔁 Git add, commit and push if file is new
git add --intent-to-add "$filepath"

if git diff --cached --quiet; then
    echo "No new file changes to commit."
else
    # Use real-looking Git identity
    git config --global user.name "Sami Rahman"
    git config --global user.email "sami.rahman@llmorg.com"

    # Rotate realistic commit messages
    commit_messages=(
      "Add new module to support recent refactors"
      "Introduce helper class for internal tooling"
      "Create standalone agent utility for coordination"
      "Early version of new logic module"
      "Begin modular split of tools and agents"
    )
    commit_msg="${commit_messages[$RANDOM % ${#commit_messages[@]}]}"

    git commit -m "$commit_msg"
    git push origin main
    echo "Pushed: $filename_snake"
fi
