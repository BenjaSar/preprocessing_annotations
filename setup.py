"""Setup configuration for preprocessing_annotations package."""

from setuptools import setup, find_packages

setup(
    name="preprocessing-annotations",
    version="0.2.0",
    description="MEP Floor Plan Annotation Pipeline",
    author="Computer Vision Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "fitz>=0.0.1",  # PyMuPDF
        "Pillow>=9.0.0",
        "opencv-python>=4.6.0",
        "numpy>=1.21.0",
        # OCR backends (PaddleOCR 2.x with GPU support)
        "paddlepaddle-gpu>=2.6.0,<3.0",  # PaddleOCR GPU engine (CUDA 11.8 bundled)
        "paddleocr>=2.7.0,<3.0",         # PaddleOCR wrapper (2.x API)
        # VLM and other dependencies
        "anthropic>=0.7.0",
        "segment-anything>=1.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "mlflow>=2.0.0",
    ],
    extras_require={
        "easyocr": ["easyocr>=1.6.0"],  # Optional: legacy OCR backend
    },
    entry_points={
        "console_scripts": [
            "annotate-pipeline=preprocessing_annotations.orchestration.pipeline:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
