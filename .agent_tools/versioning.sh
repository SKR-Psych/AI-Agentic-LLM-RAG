#!/bin/bash

# 📄 Ensure VERSION.txt exists and initialise if empty
touch VERSION.txt
if [ ! -s VERSION.txt ]; then
  echo "0.1.0" > VERSION.txt
fi

# 🔢 Read and parse version number with fallback defaults
current=$(cat VERSION.txt)
IFS='.' read -ra parts <<< "$current"
major=${parts[0]:-0}
minor=${parts[1]:-1}
patch=${parts[2]:-0}

# 🧾 Extract commit type (defensive fallback to 'patch')
commit_type="minor"
if [[ -f .agent_tools/commit_msg.txt && -s .agent_tools/commit_msg.txt ]]; then
  raw_commit_type=$(head -n 1 .agent_tools/commit_msg.txt | cut -d':' -f1 | xargs)
  if [[ "$raw_commit_type" =~ ^(feature|fix|refactor|docs|chore|minor|improvement)$ ]]; then
    commit_type="$raw_commit_type"
  else
    echo "⚠️ Unknown commit type '$raw_commit_type'. Defaulting to patch bump."
    commit_type="minor"
  fi
else
  echo "⚠️ Missing or empty commit_msg.txt. Defaulting to patch bump."
fi

# 🎯 Apply semantic versioning rules
case "$commit_type" in
  feature)
    minor=$((minor + 1))
    patch=0
    ;;
  fix|refactor)
    patch=$((patch + 1))
    ;;
  docs|chore|minor|improvement)
    patch=$((patch + 1))
    ;;
  *)
    patch=$((patch + 1))
    ;;
esac

# ✏️ Write new version
new_version="$major.$minor.$patch"
echo "$new_version" > VERSION.txt
echo "🔖 Bumped version to $new_version"
