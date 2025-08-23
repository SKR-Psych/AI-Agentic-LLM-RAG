#!/bin/bash

# 📁 Randomly select target folder: agents or tools
dirs=("tools" "agents")
target_dir="ai_agentic_core/${dirs[$RANDOM % 2]}"
mkdir -p "$target_dir"

# 🐍 Generate snake_case filename and PascalCase classname
timestamp=$(date +%s)
filename_snake="module_${timestamp}.py"
classname_pascal="Module${timestamp:5:5}"

filepath="$target_dir/$filename_snake"

# 🧠 Write Python module with class stub and 2 methods
cat << EOF > "$filepath"
class $classname_pascal:
    \"\"\"Auto-generated module for agentic behaviour simulation.\"\"\"

    def __init__(self):
        self.status = 'idle'

    def run(self):
        \"\"\"Run the module's default routine.\"\"\"
        pass

    def execute_task(self, task: str):
        \"\"\"Execute a named task passed in as a string.\"\"\"
        print(f"Executing: {task}")
EOF

echo "✅ Created module: $filepath"
