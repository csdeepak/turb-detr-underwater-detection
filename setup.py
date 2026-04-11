from setuptools import find_packages, setup

setup(
    name="turb-detr",
    version="0.1.0",
    description="Turbidity-Aware RT-DETR for Underwater Plastic Debris Detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
