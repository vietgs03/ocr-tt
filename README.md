# 🚀 OCR-TT - Advanced Local OCR System

Production-ready OCR system optimized for Vietnamese administrative documents with AI-powered accuracy.

## ✨ Features

- **🎯 Dual-Mode Processing**
  - Fast Mode: PaddleOCR (6s, 85-90% accuracy)
  - Accurate Mode: DeepSeek-OCR VLM (60-80s chunked, 95-98% accuracy)

- **⚡ Performance Optimizations**
  - Parallel chunked processing
  - Multi-core utilization
  - Model preloading & keep-alive

- **🧠 AI-Powered**
  - DeepSeek VLM for language understanding
  - Context-aware text correction
  - Automatic formatting to Markdown

- **🌐 API & GUI**
  - RESTful API with FastAPI
  - Streaming support for real-time updates
  - Modern web interface

## 📋 Requirements

- Python 3.10+
- CPU (GPU optional for 10x speedup)
- Ollama (for DeepSeek-OCR model)

## 🛠️ Installation

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 3. Pull DeepSeek-OCR model
ollama pull deepseek-ocr

# 4. Install Tesseract (optional)
sudo apt-get install tesseract-ocr tesseract-ocr-vie
```

## 🚀 Quick Start

### CLI Usage

```python
from selflearning_ocr import SelfLearningOCR

ocr = SelfLearningOCR()
result = ocr.process_image("document.jpg")
print(result)
```

### API Server

```bash
# Start API server
python3 ocr_api.py

# Server runs at http://localhost:8000
# Open gui_demo.html in browser for web interface
```

## 📊 Performance Comparison

| Engine | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| Tesseract | 11s | 70-80% | Quick drafts |
| **PaddleOCR** | **6s** | **85-90%** | **Daily work** ⭐ |
| DeepSeek (full) | 170s | 95-98% | Critical docs |
| **DeepSeek (chunked)** | **60-80s** | **95-98%** | **VIP docs** ⭐ |

## 📁 Project Structure

```
ocr-tt/
├── ocr_api.py              # FastAPI server
├── selflearning_ocr.py     # Main OCR engine with caching
├── chunked_ocr.py          # Parallel chunked processing
├── vlm_local_ocr.py        # DeepSeek VLM integration
├── fast_local_ocr.py       # Hybrid PaddleOCR + LLM
├── gui_demo.html           # Web interface
├── pdf_extractor.py        # PDF to images utility
├── crawler.py              # Image acquisition
└── smart_trainer.py        # Dictionary training
```

## 🎯 Recommended Usage

**For Office Environment (100 docs/day):**
- 95 docs → Fast Mode (PaddleOCR): 9.5 minutes
- 5 docs → Accurate Mode (DeepSeek chunked): 5 minutes
- **Total: ~15 minutes** (vs 4.7 hours with full DeepSeek!)

## 🔧 API Endpoints

```bash
# Fast Mode (PaddleOCR)
curl -X POST http://localhost:8000/ocr/fast \
  -F "file=@document.jpg"

# Accurate Mode (DeepSeek)
curl -X POST http://localhost:8000/ocr/accurate \
  -F "file=@document.jpg"

# Streaming Mode
curl -X POST http://localhost:8000/ocr/stream \
  -F "file=@document.jpg"

# Health Check
curl http://localhost:8000/health
```

## 🧪 Testing

```bash
# Benchmark all engines
python3 benchmark_ocr_tiers.py

# Test parallel processing
python3 parallel_ocr_test.py

# Clean quality comparison
python3 clean_ocr_comparison.py
```

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

## 📝 License

MIT License

## 🙏 Acknowledgments

- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) - Vision Language Model
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Fast OCR engine
- [Ollama](https://ollama.com/) - Local LLM runtime

## 📧 Contact

Issues and questions: [GitHub Issues](https://github.com/vietgs03/ocr-tt/issues)
