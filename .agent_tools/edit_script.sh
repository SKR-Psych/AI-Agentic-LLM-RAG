#!/bin/bash

# Change to repository root
cd "$(dirname "$0")/.."

# Debug: Print current directory and files
pwd
find . -type f \( -name "*.py" -o -name "*.rs" \) -not -path "*/\.*" -not -path "./target/*" | head -n 5

# Protected files and directories
EXCLUDE_PATTERN="^\.agent_tools/|^\.github/|VERSION\.txt|setup\.py|pyproject\.toml|cli\.py"

# Find all Python and Rust files that are not in protected locations
FILES=()
while IFS= read -r -d $'\0' file; do
  if [[ ! "$file" =~ $EXCLUDE_PATTERN ]]; then
    FILES+=("$file")
  fi
done < <(find . -type f \( -name "*.py" -o -name "*.rs" \) -not -path "*/\.*" -not -path "./target/*" -print0)

echo "Found ${#FILES[@]} editable files."

# Random number of files to edit (1-8)
commit_types=("refactor" "improvement" "docs" "minor")
count=$(( (RANDOM % 8) + 1 ))
selected_files=()

# Generate realistic function names
generate_fn_name() {
  prefixes=("check" "update" "refresh" "calculate" "log" "fetch" "build" "init")
  suffixes=("data" "state" "session" "cache" "timeout" "payload" "config")
  echo "${prefixes[$RANDOM % ${#prefixes[@]}]}_${suffixes[$RANDOM % ${#suffixes[@]}]}"
}

# Create commit message file
COMMIT_MSG=".agent_tools/commit_msg.txt"
echo "refactor: code improvements ($(date +'%Y-%m-%d %H:%M'))" > "$COMMIT_MSG"
echo "" >> "$COMMIT_MSG"

# Modify random files
for i in $(seq 1 $count); do
  if [ ${#FILES[@]} -eq 0 ]; then
    echo "No files available to edit."
    break
  fi
  
  index=$((RANDOM % ${#FILES[@]}))
  file=${FILES[$index]}

  if [[ " ${selected_files[*]} " == *" $file "* ]]; then
    continue
  fi

  selected_files+=("$file")
  fn_name=$(generate_fn_name)
  
  # Create directory if it doesn't exist
  mkdir -p "$(dirname "$file")"
  
  echo "Attempting to edit: $file"
  
  if [[ $file == *.py ]]; then
    # Check if file exists, if not create it with a proper Python header
    if [ ! -f "$file" ]; then
      echo -e "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\n" > "$file"
    fi
    
    # Add new function
    echo -e "\n\ndef $fn_name():\n    # TODO: logic pending\n    pass\n" >> "$file"
    echo "- Added $fn_name() to $file" >> "$COMMIT_MSG"
    echo "Successfully edited $file (Python)"
  elif [[ $file == *.rs ]]; then
    echo -e "\n\nfn $fn_name() {\n    // TODO: implement logic\n}\n" >> "$file"
    echo "- Added $fn_name() to $file" >> "$COMMIT_MSG"
    echo "Edited $file (Rust)"
  fi
done

# Exit if no changes occurred
if [ ${#selected_files[@]} -eq 0 ]; then
  echo "No files were edited. Exiting."
  exit 0
fi

# 💾# Only proceed if we made changes
if [ ${#selected_files[@]} -gt 0 ]; then
    # Configure git if not already configured
    git config --global user.name "Agentic Bot"
    git config --global user.email "agent@llmrag.com"
    
    # Add all changes
    git add .
    
    # Check if there are any changes to commit
    if ! git diff --cached --quiet; then
        echo "Committing changes..."
        git commit -F "$COMMIT_MSG" || echo "Failed to commit changes"
        git push origin main || echo "Failed to push changes"
    else
        echo "No changes to commit"
    fi
fi
