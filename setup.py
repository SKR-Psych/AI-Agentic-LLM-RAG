from setuptools import setup, find_packages

setup(
    name="agentic_core",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[],
    author="Sami Rahman",
    author_email="sami@hypebeast.com",
    description="Agentic reasoning engine for autonomous LLM applications.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SKR-Psych/AI-Agentic-LLM-RAG",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
