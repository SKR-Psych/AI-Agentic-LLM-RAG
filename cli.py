#!/usr/bin/env python3

"""
Advanced Command Line Interface for the Agentic AI System

This CLI provides access to advanced reasoning capabilities including multiple reasoning modes,
memory integration, and tool usage.

Examples:
    # Basic usage with default chain-of-thought reasoning
    $ python cli.py --prompt "Analyse the effect of prompt length on reasoning depth"
    
    # Use tree-of-thought reasoning with custom parameters
    $ python cli.py --prompt "Plan a research project on AI safety" \
                   --mode tree_of_thought \
                   --max-steps 5 \
                   --temperature 0.8
    
    # Enable memory and get verbose output
    $ python cli.py --prompt "What have we discussed about AI alignment?" \
                   --enable-memory \
                   --verbose
"""

import argparse
import json
import sys
from typing import Dict, Any, Optional
from enum import Enum

from ai_agentic_core.reasoning_engine import ReasoningEngine, ReasoningMode
from ai_agentic_core.memory_manager import MemoryManager
from ai_agentic_core.memory_analyzer import MemoryAnalyzer
from ai_agentic_core.profiler import PerformanceProfiler
from ai_agentic_core.code_analyzer import CodeAnalyzer

class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"

def print_result(result: Dict[str, Any], format: OutputFormat = OutputFormat.TEXT) -> None:
    """Print the result in the specified format."""
    if format == OutputFormat.JSON:
        print(json.dumps(result, indent=2))
    elif format == OutputFormat.MARKDOWN:
        print("# Reasoning Result\n")
        if 'answer' in result:
            print(f"**Answer:** {result['answer']}\n")
        if 'steps' in result:
            print("## Reasoning Steps\n")
            for i, step in enumerate(result['steps'], 1):
                print(f"### Step {i}")
                print(f"**Thought:** {step['thought']}")
                if step.get('action'):
                    print(f"**Action:** {step['action']}")
                if step.get('observation'):
                    print(f"**Observation:** {step['observation']}")
                if step.get('confidence'):
                    print(f"**Confidence:** {step['confidence']:.2f}")
                print()
    else:  # Default text format
        if 'answer' in result:
            print("\n🔍 Final Answer:")
            print(f"{result['answer']}\n")
        
        if 'steps' in result:
            print("🚀 Reasoning Steps:")
            for i, step in enumerate(result['steps'], 1):
                print(f"\nStep {i}:")
                print(f"  💭 {step['thought']}")
                if step.get('action'):
                    print(f"  🛠️  Action: {step['action']}")
                if step.get('observation'):
                    print(f"  👀 Observation: {step['observation']}")
                if step.get('confidence'):
                    print(f"  📊 Confidence: {step['confidence']:.2f}")

def get_reasoning_mode(mode_str: str) -> ReasoningMode:
    """Convert string to ReasoningMode enum."""
    try:
        return ReasoningMode(mode_str.lower())
    except ValueError:
        valid_modes = [m.value for m in ReasoningMode]
        raise ValueError(f"Invalid reasoning mode: {mode_str}. Valid modes: {', '.join(valid_modes)}")

def parse_key_value_args(args: Optional[list]) -> Dict[str, Any]:
    """Parse key=value arguments into a dictionary."""
    if not args:
        return {}
    
    result = {}
    for arg in args:
        try:
            key, value = arg.split('=', 1)
            # Try to evaluate the value (for bool, int, float, etc.)
            try:
                value = eval(value, {})
            except (NameError, SyntaxError):
                pass  # Keep as string if evaluation fails
            result[key] = value
        except ValueError:
            print(f"Warning: Ignoring malformed argument: {arg}")
    return result

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Agentic AI Command Line Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True, 
        help="The prompt or question to reason about"
    )
    
    # Reasoning configuration
    parser.add_argument(
        "--mode", 
        type=str, 
        default="chain_of_thought", 
        help=f"Reasoning mode to use: {', '.join([m.value for m in ReasoningMode])}"
    )
    parser.add_argument(
        "--max-steps", 
        type=int, 
        default=10, 
        help="Maximum number of reasoning steps to take"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7, 
        help="Sampling temperature (0.0 to 1.0, higher is more random)"
    )
    
    # Memory configuration
    parser.add_argument(
        "--enable-memory", 
        action="store_true", 
        help="Enable memory for context across interactions"
    )
    parser.add_argument(
        "--memory-file", 
        type=str, 
        default="memory.db", 
        help="File to store memory data"
    )
    
    # Output configuration
    parser.add_argument(
        "--format", 
        type=str, 
        choices=[f.value for f in OutputFormat], 
        default=OutputFormat.TEXT.value,
        help="Output format"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Show detailed reasoning process"
    )
    
    # Additional parameters as key=value pairs
    parser.add_argument(
        "--param", 
        action="append", 
        help="Additional parameters as key=value pairs"
    )

    # Memory analysis, profiling, and code analysis arguments
    parser.add_argument("--memory-analysis", action="store_true", help="Run memory analysis")
    parser.add_argument("--profile", help="Profile a specific function or file")
    parser.add_argument("--code-quality", help="Analyze code quality of a file or directory")
    
    args = parser.parse_args()
    
    try:
        # Initialize components
        engine = ReasoningEngine()
        memory = MemoryManager() if args.enable_memory else None
        
        # Load memory if enabled
        if memory and args.memory_file:
            try:
                memory.load(args.memory_file)
                if args.verbose:
                    print(f"Loaded memory from {args.memory_file}")
            except FileNotFoundError:
                if args.verbose:
                    print("No existing memory file found, starting fresh")
        
        # Prepare parameters
        params = {
            'max_steps': args.max_steps,
            'temperature': args.temperature,
            'verbose': args.verbose,
            **parse_key_value_args(args.param)
        }
        
        # Run reasoning
        result = engine.think(
            prompt=args.prompt,
            mode=get_reasoning_mode(args.mode),
            **params
        )
        
        # Save memory if enabled
        if memory and args.memory_file:
            memory.save(args.memory_file)
            if args.verbose:
                print(f"\n💾 Memory saved to {args.memory_file}")
        
        # Display results
        print_result(result, OutputFormat(args.format))
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
