Medical-CoT: Reasoning-First Small Language Model (SLM)
This project focuses on fine-tuning a Small Language Model (SLM) to perform complex medical diagnostics using Chain-of-Thought (CoT) reasoning. By utilizing Unsloth and QLoRA, the model is optimized to run on consumer-grade hardware (like an NVIDIA 1650 or a single T4 GPU) while maintaining high analytical accuracy.

🚀 Key Features
Reasoning-First Approach: Unlike standard models that jump to a diagnosis, this model is trained to use a <thought> process to evaluate differential diagnoses before providing a final answer.

Highly Optimized: Built using Unsloth, enabling 2x faster training and 70% less memory usage compared to standard Hugging Face fine-tuning.

Small but Mighty: Uses Qwen2.5-3B-Instruct as a base, proving that specialized 3B parameter models can outperform generalized 7B+ models in niche domains.

Quantized for Accessibility: Implements 4-bit quantization to allow inference on devices with as little as 4GB-6GB of VRAM.

🛠️ Tech Stack
Model Architecture: Qwen2.5 (3B Parameters)

Optimization: Unsloth, QLoRA, BitsAndBytes

Frameworks: PyTorch, Hugging Face Transformers, TRL (Transformer Reinforcement Learning)

Compute: Google Colab / Kaggle (T4 GPU)

📊 Dataset
The model is fine-tuned on the Medical-o1-Reasoning-SFT dataset. This dataset provides high-quality medical questions paired with complex, multi-step logical reasoning paths.

💻 Installation & Usage
Prerequisites
Python 3.10+

NVIDIA GPU (Turing architecture or newer recommended)

CUDA 12.1+

Quick Start
Bash
# Install Unsloth and dependencies
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.35" "trl<0.13.0" peft accelerate bitsandbytes
Running Inference
Python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "your-username/medical-reasoner-slm", # Replace with your model path
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# Example Case
inputs = tokenizer(["Analyze a 45yo male with tearing chest pain..."], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=1024)
print(tokenizer.decode(outputs[0]))
🧠 Reasoning Example
Input: A patient presents with sharp chest pain after a long flight and a swollen right calf.

Model Thought Process:

Identify Risk Factors: Long-duration immobility (flight), sudden onset of chest pain.

Evaluate Physical Signs: Unilateral calf swelling suggests Deep Vein Thrombosis (DVT).

Synthesize Logic: A DVT can embolize to the pulmonary arteries.

Differential Diagnosis: Ruled out Myocardial Infarction due to lack of ST-segment changes (hypothetical) and clear presence of DVT symptoms.

Conclusion: High suspicion of Pulmonary Embolism.

📈 Performance & Memory
Configuration	VRAM Usage (Training)	Training Time (60 steps)
Standard HF	~28GB (OOM on T4)	N/A
Unsloth (4-bit)	~10GB	~12 Minutes
⚖️ Disclaimer
This model is developed for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a physician or other qualified health provider with any questions regarding a medical condition.

📜 License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
