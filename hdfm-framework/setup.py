"""
Setup script for HDFM Framework
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "Hierarchical Dendritic Forest Management Framework"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]
else:
    requirements = [
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'networkx>=2.6.0',
        'matplotlib>=3.3.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.0',
    ]

setup(
    name='hdfm-framework',
    version='0.2.0',
    author='Justin Hart',
    author_email='viridisnorthllc@gmail.com',
    description='Computational framework for entropy-minimizing corridor network optimization',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jdhart81/hdfm-framework',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: GIS',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords=[
        'forest management',
        'corridor optimization',
        'landscape ecology',
        'conservation biology',
        'network analysis',
        'entropy minimization',
        'climate adaptation',
        'dendritic networks',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=2.12.0',
            'black>=21.0',
            'flake8>=3.9.0',
            'mypy>=0.900',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0',
        ],
        'gis': [
            'geopandas>=0.9.0',
            'rasterio>=1.2.0',
            'shapely>=1.7.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'hdfm-validate=examples.synthetic_landscape_validation:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
