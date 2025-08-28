"""Analyze code quality and generate improvement suggestions."""

import ast
import os
from typing import Dict, List, Any, Tuple
import re

class CodeAnalyzer:
    """Analyze Python code quality and complexity."""
    
    def __init__(self):
        self.metrics = {}
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            metrics = {
                'file_path': file_path,
                'total_lines': len(content.split('\n')),
                'code_lines': len([l for l in content.split('\n') if l.strip() and not l.strip().startswith('#')]),
                'comment_lines': len([l for l in content.split('\n') if l.strip().startswith('#')]),
                'blank_lines': len([l for l in content.split('\n') if not l.strip()]),
                'functions': [],
                'classes': [],
                'complexity': 0,
                'issues': []
            }
            
            # Analyze AST
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = self._analyze_function(node)
                    metrics['functions'].append(func_info)
                    metrics['complexity'] += func_info['complexity']
                
                elif isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node)
                    metrics['classes'].append(class_info)
                    metrics['complexity'] += class_info['complexity']
            
            # Check for common issues
            metrics['issues'] = self._find_common_issues(content, tree)
            
            return metrics
            
        except Exception as e:
            return {'error': str(e), 'file_path': file_path}
    
    def _analyze_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze a function definition."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.Call):
                complexity += 0.5
        
        return {
            'name': node.name,
            'line_number': node.lineno,
            'complexity': complexity,
            'arguments': len(node.args.args),
            'has_docstring': ast.get_docstring(node) is not None
        }
    
    def _analyze_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze a class definition."""
        complexity = 1
        methods = []
        
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                methods.append(child.name)
                complexity += self._analyze_function(child)['complexity']
        
        return {
            'name': node.name,
            'line_number': node.lineno,
            'complexity': complexity,
            'methods': methods,
            'has_docstring': ast.get_docstring(node) is not None
        }
    
    def _find_common_issues(self, content: str, tree: ast.AST) -> List[str]:
        """Find common code quality issues."""
        issues = []
        
        # Check for long functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.body) > 20:
                    issues.append(f"Function '{node.name}' is too long ({len(node.body)} lines)")
        
        # Check for magic numbers
        numbers = re.findall(r'\b\d{3,}\b', content)
        if len(numbers) > 5:
            issues.append("Too many magic numbers detected")
        
        # Check for TODO comments
        todos = re.findall(r'TODO|FIXME|XXX|HACK', content, re.IGNORECASE)
        if todos:
            issues.append(f"Found {len(todos)} TODO/FIXME comments")
        
        return issues
    
    def analyze_project(self, project_path: str = '.') -> Dict[str, Any]:
        """Analyze entire project."""
        project_metrics = {
            'total_files': 0,
            'total_lines': 0,
            'total_complexity': 0,
            'files': [],
            'summary': {}
        }
        
        for root, dirs, files in os.walk(project_path):
            if 'venv' in root or '__pycache__' in root or '.git' in root:
                continue
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    file_metrics = self.analyze_file(file_path)
                    
                    if 'error' not in file_metrics:
                        project_metrics['files'].append(file_metrics)
                        project_metrics['total_files'] += 1
                        project_metrics['total_lines'] += file_metrics['total_lines']
                        project_metrics['total_complexity'] += file_metrics['complexity']
        
        # Generate summary
        if project_metrics['files']:
            project_metrics['summary'] = {
                'average_complexity': project_metrics['total_complexity'] / project_metrics['total_files'],
                'average_lines_per_file': project_metrics['total_lines'] / project_metrics['total_files'],
                'most_complex_file': max(project_metrics['files'], key=lambda x: x['complexity']),
                'total_issues': sum(len(f['issues']) for f in project_metrics['files'])
            }
        
        return project_metrics

# Usage
analyzer = CodeAnalyzer()
project_analysis = analyzer.analyze_project()
print(json.dumps(project_analysis, indent=2))


def check_config():
    # TODO: logic pending
    pass

