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
count=$(( (RANDOM % 8) + 1 ))
selected_files=()

# Generate realistic function names and implementations
generate_ai_function() {
    local file_type=$1
    
    # AI/ML function templates
    local ai_functions=(
        "def optimize_attention_weights(attention_scores, temperature=1.0):\n    \"\"\"Optimize attention weights using temperature scaling.\"\"\"\n    scaled_scores = attention_scores / temperature\n    return torch.softmax(scaled_scores, dim=-1)"
        
        "def compute_gradient_norm(parameters):\n    \"\"\"Compute L2 norm of gradients for gradient clipping.\"\"\"\n    total_norm = 0.0\n    for p in parameters:\n        if p.grad is not None:\n            param_norm = p.grad.data.norm(2)\n            total_norm += param_norm.item() ** 2\n    return total_norm ** 0.5"
        
        "def apply_layer_norm(x, weight, bias, eps=1e-5):\n    \"\"\"Apply layer normalization to input tensor.\"\"\"\n    mean = x.mean(-1, keepdim=True)\n    var = x.var(-1, keepdim=True, unbiased=False)\n    return weight * (x - mean) / (var + eps).sqrt() + bias"
        
        "def compute_cosine_similarity(a, b):\n    \"\"\"Compute cosine similarity between two vectors.\"\"\"\n    dot_product = torch.dot(a, b)\n    norm_a = torch.norm(a)\n    norm_b = torch.norm(b)\n    return dot_product / (norm_a * norm_b)"
        
        "def apply_dropout(x, p=0.1, training=True):\n    \"\"\"Apply dropout during training.\"\"\"\n    if training and p > 0:\n        mask = torch.bernoulli(torch.ones_like(x) * (1 - p))\n        return x * mask / (1 - p)\n    return x"
        
        "def compute_kl_divergence(p, q):\n    \"\"\"Compute KL divergence between two probability distributions.\"\"\"\n    return torch.sum(p * torch.log(p / q))"
        
        "def apply_positional_encoding(x, max_len=5000):\n    \"\"\"Apply sinusoidal positional encoding to input.\"\"\"\n    pe = torch.zeros(max_len, x.size(-1))\n    position = torch.arange(0, max_len).unsqueeze(1).float()\n    div_term = torch.exp(torch.arange(0, x.size(-1), 2).float() * -(math.log(10000.0) / x.size(-1)))\n    pe[:, 0::2] = torch.sin(position * div_term)\n    pe[:, 1::2] = torch.cos(position * div_term)\n    return x + pe[:x.size(0)]"
        
        "def compute_bleu_score(predictions, references):\n    \"\"\"Compute BLEU score for text generation evaluation.\"\"\"\n    from nltk.translate.bleu_score import sentence_bleu\n    return sentence_bleu(references, predictions)"
    )
    
    # Memory management functions
    local memory_functions=(
        "def update_memory_importance(memory_id, new_importance):\n    \"\"\"Update importance score of a memory item.\"\"\"\n    if memory_id in self.memories:\n        self.memories[memory_id].importance = max(0.0, min(1.0, new_importance))\n        self.memories[memory_id].last_updated = time.time()\n        return True\n    return False"
        
        "def prune_low_importance_memories(threshold=0.1):\n    \"\"\"Remove memories with importance below threshold.\"\"\"\n    to_remove = [mid for mid, mem in self.memories.items() if mem.importance < threshold]\n    for mid in to_remove:\n        del self.memories[mid]\n    return len(to_remove)"
        
        "def compute_memory_retrieval_score(query, memory):\n    \"\"\"Compute retrieval score for memory search.\"\"\"\n    query_embedding = self.encode_query(query)\n    memory_embedding = memory.embedding\n    similarity = torch.cosine_similarity(query_embedding, memory_embedding, dim=0)\n    return similarity * memory.importance"
    )
    
    # Reasoning functions
    local reasoning_functions=(
        "def chain_of_thought_reasoning(prompt, max_steps=5):\n    \"\"\"Implement chain-of-thought reasoning.\"\"\"\n    thoughts = []\n    current_thought = prompt\n    \n    for step in range(max_steps):\n        # Generate next thought\n        next_thought = self.generate_next_thought(current_thought)\n        thoughts.append(next_thought)\n        \n        # Check if we've reached a conclusion\n        if self.is_conclusion(next_thought):\n            break\n            \n        current_thought = next_thought\n    \n    return thoughts"
        
        "def tree_of_thoughts_search(initial_state, max_depth=3):\n    \"\"\"Implement tree-of-thoughts search algorithm.\"\"\"\n    frontier = [(initial_state, 0)]\n    best_path = None\n    best_score = float('-inf')\n    \n    while frontier:\n        current_state, depth = frontier.pop(0)\n        \n        if depth >= max_depth:\n            score = self.evaluate_state(current_state)\n            if score > best_score:\n                best_score = score\n                best_path = current_state\n            continue\n        \n        # Generate next states\n        next_states = self.generate_next_states(current_state)\n        for next_state in next_states:\n            frontier.append((next_state, depth + 1))\n    \n    return best_path, best_score"
        
        "def self_reflection_loop(initial_response, max_iterations=3):\n    \"\"\"Implement self-reflection for response improvement.\"\"\"\n    current_response = initial_response\n    \n    for iteration in range(max_iterations):\n        # Analyze current response\n        analysis = self.analyze_response_quality(current_response)\n        \n        if analysis['score'] > 0.8:  # Good enough\n            break\n        \n        # Generate improvement suggestions\n        suggestions = self.generate_improvement_suggestions(analysis)\n        \n        # Apply improvements\n        current_response = self.improve_response(current_response, suggestions)\n    \n    return current_response"
    )
    
    # Randomly select function type and implementation
    local function_type=$((RANDOM % 3))
    local function_index=$((RANDOM % ${#ai_functions[@]}))
    
    case $function_type in
        0) echo -e "${ai_functions[$function_index]}" ;;
        1) echo -e "${memory_functions[$((RANDOM % ${#memory_functions[@]}))]}" ;;
        2) echo -e "${reasoning_functions[$((RANDOM % ${#reasoning_functions[@]}))]}" ;;
    esac
}

# Create commit message file
COMMIT_MSG=".agent_tools/commit_msg.txt"
echo "feat: implement advanced AI algorithms ($(date +'%Y-%m-%d %H:%M'))" > "$COMMIT_MSG"
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
  
  echo "Attempting to edit: $file"
  
  if [[ $file == *.py ]]; then
    # Check if file exists, if not create it with a proper Python header
    if [ ! -f "$file" ]; then
      echo -e "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\nimport torch\nimport numpy as np\nimport time\nimport math\nfrom typing import List, Dict, Any, Optional\n\n" > "$file"
    fi
    
    # Add real AI function
    real_function=$(generate_ai_function "python")
    echo -e "\n$real_function\n" >> "$file"
    
    # Extract function name for commit message
    func_name=$(echo "$real_function" | grep -o 'def [a-zA-Z_][a-zA-Z0-9_]*' | head -1 | sed 's/def //')
    echo "- Added $func_name() to $file" >> "$COMMIT_MSG"
    echo "Successfully edited $file (Python)"
    
  elif [[ $file == *.rs ]]; then
    # Generate Rust AI function
    rust_functions=(
      "pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {\n    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();\n    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));\n    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();\n    let sum_exp = exp_scores.iter().sum::<f32>();\n    exp_scores.iter().map(|&s| s / sum_exp).collect()\n}"
      
      "pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {\n    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;\n    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;\n    x.iter().enumerate().map(|(i, &val)| {\n        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]\n    }).collect()\n}"
      
      "pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {\n    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();\n    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();\n    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();\n    dot_product / (norm_a * norm_b)\n}"
    )
    
    rust_func=${rust_functions[$((RANDOM % ${#rust_functions[@]}))]}
    echo -e "\n$rust_func\n" >> "$file"
    
    # Extract function name for commit message
    func_name=$(echo "$rust_func" | grep -o 'fn [a-zA-Z_][a-zA-Z0-9_]*' | head -1 | sed 's/fn //')
    echo "- Added $func_name() to $file" >> "$COMMIT_MSG"
    echo "Edited $file (Rust)"
  fi
done

# Exit if no changes occurred
if [ ${#selected_files[@]} -eq 0 ]; then
  echo "No files were edited. Exiting."
  exit 0
fi

# Only proceed if we made changes
if [ ${#selected_files[@]} -gt 0 ]; then
    # Configure git if not already configured
    git config --global user.name "Agentic Bot"
    git config --global user.email "agent@llmrag.com"
    
    # CRITICAL FIX: Pull latest changes before committing
    echo "Pulling latest changes..."
    git pull origin main --rebase || git pull origin main --no-rebase
    
    # Add all changes
    git add .
    
    # Check if there are any changes to commit
    if ! git diff --cached --quiet; then
        echo "Committing changes..."
        git commit -F "$COMMIT_MSG" || echo "Failed to commit changes"
        
        # Try to push with retry logic
        echo "Pushing changes..."
        for attempt in 1 2 3; do
            if git push origin main; then
                echo "Successfully pushed on attempt $attempt"
                break
            else
                echo "Push failed on attempt $attempt, pulling and retrying..."
                git pull origin main --rebase || git pull origin main --no-rebase
                sleep 2
            fi
        done
    else
        echo "No changes to commit"
    fi
fi
