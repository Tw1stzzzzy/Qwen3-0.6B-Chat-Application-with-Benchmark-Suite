# Qwen3-0.6B Chat Application & Benchmark Suite

A complete chat application based on Qwen3-0.6B model with memory optimization and comprehensive performance testing tools.

## ✨ Core Features

- 🤖 **Intelligent Chat** - Interactive chat interface powered by Qwen3-0.6B
- 🔧 **Memory Optimization** - Automatic memory management to prevent GPU OOM errors
- 📊 **Performance Testing** - 5 benchmark tests for comprehensive model evaluation
- 🎨 **Beautiful Interface** - Gradio web interface with adjustable parameters

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch Chat Application

```bash
python3 gradio_chat_app.py
```


### 3. Run Performance Tests (Optional)

```bash
# Full test suite (takes 10-15 minutes)
python3 qwen3_benchmark.py

# Individual tests
python3 qwen3_benchmark.py truthful     # Truthfulness test
python3 qwen3_benchmark.py law          # Legal knowledge test
python3 qwen3_benchmark.py reasoning    # Reasoning ability test
```

## 📋 Test Categories

| Test Type | Questions | Capability Tested |
|-----------|-----------|-------------------|
| TruthfulQA | 50 | Hallucination detection |
| MMLU-Law | 50 | Professional knowledge |
| ARC-Easy | 30 | Basic reasoning |
| Multi-turn | 5×3 rounds | Memory & context understanding |
| Adversarial | 10 prompts | Safety & robustness |

## 🔧 System Requirements

- **Python**: 3.8+
- **Memory**: 8GB+ RAM
- **Storage**: 5GB available space
- **GPU**: Recommended NVIDIA GPU (8GB+ VRAM)

## 💡 Usage Tips

### Chat Application
- **Memory Issues**: Reduce Max Tokens to 64-128
- **Long Conversations**: Regularly click Clear button to free memory
- **Parameter Tuning**: Temperature controls creativity, Top-p controls vocabulary diversity

### Performance Testing
- **First Run**: Model download takes time, please be patient
- **Test Results**: Saved as JSON files with detailed Q&A records
- **Custom Testing**: Edit qwen3_benchmark.py to adjust test quantities

## 📁 Project Structure

```
qwen3_0.6b/
├── gradio_chat_app.py          # Main chat application
├── qwen3_benchmark.py          # Benchmark testing tool
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── qwen3_benchmark_results_*.json  # Test results (auto-generated)
```

## 🛠️ Troubleshooting

### Common Issues

**Q: "Model not found" error on startup**
```bash
# Check network connection, model will auto-download from HuggingFace
# First run requires downloading ~1.5GB model files
```

**Q: GPU out of memory error**
```bash
# 1. Reduce Max Tokens parameter
# 2. Close other GPU programs
# 3. Restart application
```

**Q: Benchmark tests fail**
```bash
# 1. Ensure chat application runs normally
# 2. Check sufficient storage space
# 3. Run individual tests to isolate issues
```

**Q: Web interface inaccessible**
```bash
# Check terminal output for URL
# Default address: http://localhost:7860
# Firewall may need to allow port 7860
```

## 📞 Technical Support

If you encounter issues:
1. Check Python version (requires 3.8+)
2. Ensure all dependencies are properly installed
3. Review terminal error messages
4. Check system resource usage

## 📄 License

This project is open-sourced under the Apache 2.0 License.

---

🌟 **Start experiencing the power of Qwen3-0.6B now!** 
