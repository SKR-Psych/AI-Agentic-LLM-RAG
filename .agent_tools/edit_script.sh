#!/bin/bash

# 10% chance to skip this run
if [ $((RANDOM % 10)) -lt 1 ]; then
  echo " Skipping this run (random chance)"
  exit 0
fi

# Protected files and directories
EXCLUDE_PATTERN="^\.agent_tools/|^\.github/|VERSION\.txt|setup\.py|pyproject\.toml|cli\.py"

# Find all Python and Rust files that are not in protected locations
FILES=()
while IFS= read -r -d $'\0' file; do
  # Skip files matching exclude pattern
  if [[ ! "$file" =~ $EXCLUDE_PATTERN ]]; then
    FILES+=("$file")
  fi
done < <(find . -type f \( -name "*.py" -o -name "*.rs" \) -not -path "*/\.*" -print0)

commit_types=("refactor" "feature" "docs" "fix")
count=$(( (RANDOM % 3) + 1 ))  # edit 1–3 files
selected_files=()

for i in $(seq 1 $count); do
  index=$((RANDOM % ${#FILES[@]}))
  file=${FILES[$index]}
  selected_files+=("$file")

  # Generate random function name (e.g. auto_fn_98523_49383)
  fn_name="auto_fn_$(date +%s | cut -c6-)_$RANDOM"

  # Append a fake function based on file type
  if [[ $file == *.py ]]; then
    echo -e "\n# Auto-generated Python function\n\ndef $fn_name():\n    pass\n" >> "$file"
  elif [[ $file == *.rs ]]; then
    echo -e "\n// Auto-generated Rust function\n\nfn $fn_name() {\n    // todo\n}\n" >> "$file"
  fi
done

# Write commit message for downstream steps
commit_type=${commit_types[$RANDOM % ${#commit_types[@]}]}
echo "$commit_type: 🤖 Edited ${#selected_files[@]} files (auto)" > .agent_tools/commit_msg.txt