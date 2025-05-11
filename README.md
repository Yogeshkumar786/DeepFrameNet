# 🎭 Deepfake Video Detection
A comprehensive deep learning pipeline to detect deepfake videos using both video and text-based models. This project supports preprocessing, metadata generation, model training, and inference. It also includes support for large datasets using Git LFS and GitHub Codespaces.

## 📌 Features
- ✅ Deepfake detection using video and textual metadata
- 🧠 Pretrained model support (`.pth`, `.safetensors`)
- 🎥 Video preprocessing and frame extraction
- 📄 Automatic metadata generation
- 💾 Git LFS support for large files
- 💻 GitHub Codespaces ready
- 🚀 Modular, extensible Python codebase

## 🧾 Table of Contents
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Using Git LFS](#using-git-lfs)
- [Usage Guide](#usage-guide)
- [Running in GitHub Codespaces](#running-in-github-codespaces)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

## 🗂️ Project Structure
Deepfake-Detector-Updated/
├── model/ # Trained models and configs
│ ├── model.pth # Model weights (LFS)
│ └── text_model/
│ ├── config.json
│ └── model.safetensors # HuggingFace format (LFS)
├── videos/ # Raw input videos (LFS)
├── preprocessed_videos/ # Frame-by-frame video data
├── tokenizer/ # Tokenizer files (optional)
├── .gitignore
├── .gitattributes
├── metadata.json
├── gen_metadata.py # Simple metadata script
├── generate_metadata.py # Advanced metadata generation
├── model.py # Model architecture
├── predict.py # Inference script
├── train.py # Training loop
├── video_processing.py # Preprocesses videos into frames
└── README.md

## ⚙️ Setup Instructions

### 1. Clone the Repository (with LFS)
Make sure [Git LFS is installed](https://git-lfs.github.com/):

git lfs install
git clone https://github.com/Yogeshkumar786/Deepfake-Video-Detection.git
cd Deepfake-Video-Detection
git lfs pull
### 2. Set Up a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
### 3. Install Dependencies
If there's a requirements.txt, run:
pip install -r requirements.txt
Otherwise, install manually:

pip install torch torchvision transformers opencv-python tqdm

## 💾 Using Git LFS
This project uses [Git Large File Storage (LFS)] to manage big files:

Tracked files:
.mp4 — input videos

.pth — model checkpoints

.safetensors — HuggingFace model format

Commands to manage:
git lfs install
git lfs track "*.mp4"
git lfs track "*.pth"
git lfs track "*.safetensors"
git add .gitattributes
To push LFS files:

git add path/to/large_file
git commit -m "Add large file"
git push origin main
## 🧪 Usage Guide
### 1. Preprocess Videos
Extracts frames from videos and stores them:
python video_processing.py

### 2. Generate Metadata
python generate_metadata.py
or the simpler version:
python gen_metadata.py

### 3. Train the Model
python train.py
Optional flags can be added to choose model type or dataset.

### 4. Run Inference
python predict.py --input_path videos/MY.mp4
💻 Running in GitHub Codespaces
This project is fully compatible with GitHub Codespaces.

Inside Codespaces:
git lfs pull        # Pull large files
python train.py     # Run training
python predict.py   # Run inference
No special changes required. Just make sure .gitattributes and Git LFS are present.

## 🧑‍💻 Contributing
We welcome contributions! Here's how you can help:

Fork the repository.
Create a new branch:
git checkout -b feature-name
Commit your changes and push:

git commit -m "Add new feature"
git push origin feature-name
Submit a Pull Request.

Please follow clean code practices and write meaningful commit messages.

## 🛠️ Troubleshooting
Problem -> Solution
error: src refspec main does not match any -> Run git branch — you're likely on master not main
error: file >100MB -> Track the file using Git LFS
RPC failed: curl 55 -> File too large or unstable connection-split into smaller commits
fatal: Could not read from remote repository -> Check your remote URL with git remote -v and fix with git remote set-url origin ...
LFS files not pulling in Codespaces -> Run git lfs pull inside the Codespace terminal

## 🙏 Acknowledgements
OpenAI, HuggingFace, and PyTorch communities for foundational tools.
Researchers working on deepfake detection for inspiring this project.
Everyone contributing to fighting misinformation and AI misuse.


## 🖥️ Running on Local PC
If you prefer to run the project on your local machine, follow these steps:
✅ Prerequisites
Make sure the following are installed:
Python 3.8+

Git
Git LFS
pip (comes with Python)
## 🚀 Steps
Clone the repository and pull LFS files
git lfs install
git clone https://github.com/Yogeshkumar786/Deepfake-Video-Detection.git
cd Deepfake-Video-Detection
git lfs pull
(Optional but Recommended) Create a virtual environment

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
Install required packages

pip install -r requirements.txt
If requirements.txt is missing, install manually:

pip install torch torchvision transformers opencv-python tqdm
Run preprocessing or training scripts

Example usage:
python video_processing.py              # Preprocess videos into frames
python generate_metadata.py            # Generate metadata
python train.py                        # Train model
python predict.py --input_path MY.mp4  # Run prediction

## 💡 Tips
Use a GPU-enabled machine for faster model training and inference.
Make sure large files (.mp4, .pth, .safetensors) are pulled via LFS before running the code.
Avoid placing raw datasets inside the repo folder unless tracked with .gitignore or Git LFS.