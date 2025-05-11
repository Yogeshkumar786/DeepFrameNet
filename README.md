# 🎭 Deepfake Video Detection
A comprehensive deep learning pipeline to detect deepfake videos using both video and text-based models. This project supports preprocessing, metadata generation, model training, and inference. The dataset is hosted on Google Drive.

## 📌 Features
- ✅ Deepfake detection using video and textual metadata
- 🧠 Pretrained model support (`.pth`, `.safetensors`)
- 🎥 Video preprocessing and frame extraction
- 📄 Automatic metadata generation
- 💻 GitHub Codespaces ready
- 🚀 Modular, extensible Python codebase

## 🧾 Table of Contents
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Dataset Access](#dataset-access)
- [Usage Guide](#usage-guide)
- [Running in GitHub Codespaces](#running-in-github-codespaces)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)
- [Running on Local PC](#running-on-local-pc)

## 🗂️ Project Structure

Deepfake-Detector-Updated/
├── model/

│   ├── model.pth

│   └── text_model/

│       ├── config.json

│       └── model.safetensors

├── videos/

├── preprocessed_videos/

├── tokenizer/

├── .gitignore

├── metadata.json

├── gen_metadata.py

├── generate_metadata.py

├── model.py

├── predict.py

├── train.py

├── video_processing.py

└── README.md

## ⚙️ Setup Instructions

### 1. Clone the Repository
bash->
git clone https://github.com/Yogeshkumar786/Deepfake-Video-Detection.git
cd Deepfake-Video-Detection

### 2. Install Dependencies
pip install torch torchvision transformers opencv-python tqdm

## 💾 Dataset Access
The dataset (videos, model weights, and other large files) is hosted on Google Drive. Download the dataset from the following link and place the files in the appropriate directories:

https://drive.google.com/drive/folders/18yZ9Qw7lYfoZmXNZirAjxZZYjALS4pq9?usp=sharing

Instructions:

1. Download the dataset folder.
2. Extract the contents.
3. Move .mp4 files to the videos/ directory.
4. Move .pth and .safetensors files to the model/ directory.

## 🧪 Usage Guide
1. Preprocess Videos
   Extracts frames from videos and stores them: python video_processing.py
2. Generate Metadata
   python gen_metadata.py
3. Train the Model
   python train.py
4. Run Inference
   python predict.py --input_path videos/MY.mp4

## 💻 Running in GitHub Codespaces
This project is fully compatible with GitHub Codespaces.

Inside Codespaces:

1. Download the dataset from the Google Drive link.
2. Place the files in the appropriate directories (videos/, model/).
3. Run scripts as needed:
   python train.py     
   python predict.py

## 🙏 Acknowledgements
1. OpenAI, HuggingFace, and PyTorch communities for foundational tools.
2. Researchers working on deepfake detection for inspiring this project.
3. Everyone contributing to fighting misinformation and AI misuse.

## 🖥️ Running on Local PC
### ✅ Prerequisites
Make sure the following are installed:
1. Python 3.8+
2. Git
3. pip (comes with Python)

### 🚀 Steps
1. Clone the repository:
   git clone https://github.com/Yogeshkumar786/Deepfake-Video-Detection.git
   cd Deepfake-Video-Detection
2. Download the dataset from the Google Drive link and place files in videos/ and model/.
3. Install required packages:
pip install torch torchvision transformers opencv-python tqdm
4. Run preprocessing or training scripts:
python video_processing.py
python generate_metadata.py
python train.py
python predict.py --input_path MY.mp4

## 💡 Tips
1. Use a GPU-enabled machine for faster model training and inference.
2. Ensure dataset files are correctly placed in videos/ and model/ before running the code.
3. Avoid placing raw datasets inside the repo folder unless tracked with .gitignore.
