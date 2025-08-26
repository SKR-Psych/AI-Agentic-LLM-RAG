#!/bin/bash

# 10% chance to skip this run
if [ $((RANDOM % 10)) -lt 1 ]; then
  echo "⏭️ Skipping this run (random chance)"
  exit 0
fi

# 🛡️ Protected files and directories
EXCLUDE_PATTERN="^\.agent_tools/|^\.github/|VERSION\.txt|setup\.py|pyproject\.toml|cli\.py"

# 🔍 Find all Python and Rust files that are not in protected locations
FILES=()
while IFS= read -r -d $'\0' file; do
  if [[ ! "$file" =~ $EXCLUDE_PATTERN ]]; then
    FILES+=("$file")
  fi
done < <(find . -type f \( -name "*.py" -o -name "*.rs" \) -not -path "*/\.*" -print0)

# 🧠 Randomly select and modify 1–3 files
commit_types=("refactor" "feature" "docs" "fix")
count=$(( (RANDOM % 3) + 1 ))
selected_files=()

for i in $(seq 1 $count); do
  index=$((RANDOM % ${#FILES[@]}))
  file=${FILES[$index]}
  selected_files+=("$file")

  fn_name="auto_fn_$(date +%s | cut -c6-)_$RANDOM"

  if [[ $file == *.py ]]; then
    echo -e "\n# Auto-generated Python function\n\ndef $fn_name():\n    pass\n" >> "$file"
  elif [[ $file == *.rs ]]; then
    echo -e "\n// Auto-generated Rust function\n\nfn $fn_name() {\n    // todo\n}\n" >> "$file"
  fi
done

# 📝 Git add, commit, and push if there are changes
sync
git add --intent-to-add .

if git diff --cached --quiet; then
  echo "⚠No changes to commit."
else
  git config --global user.name "AgenticBot"
  git config --global user.email "agent@llmrag.com"

  commit_type=${commit_types[$RANDOM % ${#commit_types[@]}]}
  git commit -m "$commit_type: 🤖 Edited ${#selected_files[@]} files (auto)"
  git push origin main
  echo "✅ Pushed auto-edit commit"
fi
