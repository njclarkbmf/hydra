from setuptools import setup, find_packages

setup(
    name="hydra-mlops",
    version="0.1.0",
    packages=find_packages(),
    description="A comprehensive MLOps framework built around LanceDB and n8n.io",
    author="Your Name",
    author_email="your.email@example.com",
    install_requires=[
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "lancedb>=0.2.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "pydantic>=1.10.0",
        "python-dotenv>=1.0.0",
        "requests>=2.28.0",
        "scikit-learn>=1.2.0",
        "sentence-transformers>=2.2.0",
        "torch>=2.0.0",
        "scipy>=1.10.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
