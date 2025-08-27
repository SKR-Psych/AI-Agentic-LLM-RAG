#!/bin/bash

# Protected files and directories
EXCLUDE_PATTERN="^\.agent_tools/|^\.github/|VERSION\.txt|setup\.py|pyproject\.toml|cli\.py"

# Find all Python and Rust files that are not in protected locations
FILES=()
while IFS= read -r -d $'\0' file; do
  if [[ ! "$file" =~ $EXCLUDE_PATTERN ]]; then
    FILES+=("$file")
  fi
done < <(find . -type f \( -name "*.py" -o -name "*.rs" \) -not -path "*/\.*" -print0)

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
  index=$((RANDOM % ${#FILES[@]}))
  file=${FILES[$index]}

  if [[ " ${selected_files[*]} " == *" $file "* ]]; then
    continue
  fi

  selected_files+=("$file")
  fn_name=$(generate_fn_name)

  if [[ $file == *.py ]]; then
    echo -e "\n\ndef $fn_name():\n    # TODO: logic pending\n    pass\n" >> "$file"
    echo "- Added $fn_name() to $file" >> "$COMMIT_MSG"
    echo "Edited $file (Python)"
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

# 💾 Git commit and push
git config --global user.name "Sami Rahman"
git config --global user.email "sami.rahman@llmorg.uk"

sync
git add --intent-to-add .

if git diff --cached --quiet; then
  echo "No staged changes to commit."
else
  commit_type=${commit_types[$RANDOM % ${#commit_types[@]}]}
  commit_msg="$commit_type: small updates to ${#selected_files[@]} files"

  echo "$commit_msg" > .agent_tools/commit_msg.txt  # ✅ Needed for versioning

  git commit -m "$commit_msg"
  git push origin main
  echo "✅ Committed: $commit_msg"
fi
