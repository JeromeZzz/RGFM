"""
KumoRFM安装配置
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# 读取requirements
with open('requirements.txt') as f:
    required = f.read().splitlines()
    # 过滤注释和空行
    required = [line for line in required if line and not line.startswith('#')]

setup(
    name="kumorfm",
    version="0.1.0",
    author="KumoRFM Team",
    author_email="kumorfm@example.com",
    description="KumoRFM: 关系数据基础模型",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kumorfm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=required,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "networkx>=2.6.0",
            "graphviz>=0.19.0",
        ],
        "advanced": [
            "wandb>=0.13.0",
            "optuna>=3.0.0",
            "ray[tune]>=2.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "kumorfm-demo=examples.demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)