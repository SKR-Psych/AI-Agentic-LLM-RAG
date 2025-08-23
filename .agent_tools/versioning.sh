#!/bin/bash

# 📄 Ensure VERSION.txt exists
touch VERSION.txt

# Initialise version if empty
if [ ! -s VERSION.txt ]; then
  echo "0.1.0" > VERSION.txt
fi

# 🔢 Read and parse version number
current=$(cat VERSION.txt)
IFS='.' read -ra parts <<< "$current"
major=${parts[0]}
minor=${parts[1]}
patch=${parts[2]}

# 🧾 Extract commit type from latest commit message
commit_type=$(head -n 1 .agent_tools/commit_msg.txt | cut -d':' -f1)

# 🎯 Apply semantic versioning rules
case "$commit_type" in
  feature)
    minor=$((minor + 1))
    patch=0
    ;;
  fix|refactor)
    patch=$((patch + 1))
    ;;
  docs|chore)
    patch=$((patch + 1))
    ;;
  *)
    echo "⚠️ Unrecognised commit type '$commit_type'. Defaulting to patch bump."
    patch=$((patch + 1))
    ;;
esac

# ✏️ Write new version to file
new_version="$major.$minor.$patch"
echo "$new_version" > VERSION.txt
echo "🔖 Bumped version to $new_version"
