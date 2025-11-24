# Label Bill (Typhoon OCR & Fine-tuning)

This project is a comprehensive solution for Thai receipt OCR (Optical Character Recognition) and information extraction, leveraging the **Typhoon OCR** model (based on Qwen-VL). It includes a FastAPI backend for serving the model, a React frontend for user interaction, and scripts for fine-tuning the model on Thai handwriting datasets.

## 🚀 Features

-   **OCR & Extraction**: Extracts key information from Thai receipts (Merchant Name, Date, Items, Total, Tax, etc.) using a Vision-Language Model.
-   **Fine-tuning Pipeline**: Complete script (`model.py`) for fine-tuning Qwen 2.5 VL on Thai handwriting datasets with LoRA and Flash Attention.
-   **Interactive UI**: Modern React-based frontend to upload images and view extracted data.
-   **FastAPI Backend**: Efficient API server to handle image processing requests.

## 📂 Project Structure

```
.
├── backend/                # FastAPI Backend
│   ├── server.py           # Main API server
│   └── ...
├── frontend/               # React Frontend (Vite)
│   ├── src/                # Source code
│   └── ...
├── model.py                # Qwen 2.5 VL Fine-tuning script
├── pyphoon.py              # Standalone script for testing Typhoon OCR
├── DownloadModel.py        # Utility to download models
└── receipts/               # Sample receipt images
```

## 🛠️ Tech Stack

-   **Backend**: Python, FastAPI, Uvicorn, PyTorch, Transformers, Pillow
-   **Frontend**: TypeScript, React, Vite, Tailwind CSS, Lucide React
-   **Model**: [scb10x/typhoon-ocr1.5-2b](https://huggingface.co/scb10x/typhoon-ocr1.5-2b) (Qwen 2.5 VL based)

## ⚡ Getting Started

### Prerequisites

-   Python 3.10+
-   Node.js 18+
-   CUDA-compatible GPU (recommended for model inference and training)

### 1. Backend Setup

```bash
# Create and activate virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn torch transformers pillow accelerate peft datasets qwen-vl-utils
# Note: You might need to install flash-attn separately if supported

# Run the server
python backend/server.py
```
The API will be available at `http://localhost:5001`.

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```
The UI will be available at `http://localhost:5173`.

### 3. Fine-tuning

To fine-tune the model on Thai handwriting data:

```bash
python model.py
```
*Note: This script is configured for a GPU environment (e.g., Kaggle, Colab) and may require adjustments for local execution depending on your hardware.*

## 📝 Usage

1.  Start the Backend server.
2.  Start the Frontend dev server.
3.  Open the frontend URL in your browser.
4.  Upload a receipt image.
5.  View the extracted JSON data.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
