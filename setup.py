from setuptools import setup, find_packages

setup(
    name="cag_emotion_tracker",
    version="1.0.0",
    description="Cache-Augmented Generation for Real-Time Emotion Tracking",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "cag-emotion=main:main",
        ],
    },
)
