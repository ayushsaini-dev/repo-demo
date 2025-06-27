# MEDIGUIDE Project: Medical Question-Answering with GPT-2

## Overview
The MEDIGUIDE project fine-tunes GPT-2 models for medical question-answering, focusing on generating symptom lists for conditions like diabetes. It uses the MedQuAD dataset and implements three fine-tuning approaches: Basic Fine-Tuning, Prompt-Tuning (PEFT), and LoRA Fine-Tuning (PEFT). The project evaluates models using ROUGE scores, latency, model size, and perplexity.

## Directory Structure
- `data/`: Contains `train_data.json`, `val_data.json`, and `test_data.json` (you need to provide these).
- `scripts/`: Contains Python scripts for each step of the pipeline.
- `utils/`: Utility functions for evaluation.
- `requirements.txt`: List of dependencies.
- `LICENSE`: MIT License.

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/mediguide_project.git
   cd mediguide_project
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Data**
   - Place `train_data.json`, `val_data.json`, and `test_data.json` in the `data/` directory.
   - The data should be in JSON format with question-answer pairs, as used in the MedQuAD dataset.

## Usage
Run the scripts in the following order to replicate the project:

1. **Install Requirements**
   ```bash
   python scripts/01_install_requirements.py
   ```

2. **Preprocess Data**
   ```bash
   python scripts/02_data_preprocessing.py
   ```

3. **Basic Fine-Tuning**
   ```bash
   python scripts/03_basic_finetune.py
   ```

4. **Prompt-Tuning**
   ```bash
   python scripts/04_prompt_tuning.py
   ```

5. **LoRA Fine-Tuning**
   ```bash
   python scripts/05_lora_finetune.py
   ```

6. **Evaluate Models**
   ```bash
   python scripts/06_evaluate_models.py
   ```

7. **Compute Perplexity**
   ```bash
   python scripts/07_compute_perplexity.py
   ```

8. **Test the LoRA Model**
   ```bash
   python scripts/08_test_model.py
   ```

## Results
- **Basic Fine-Tuned Model**: ROUGE-1: 0.2479, ROUGE-2: 0.1681, ROUGE-L: 0.1983, Latency: 0.791s, Perplexity: 2.3795
- **Prompt-Tuned Model**: ROUGE-1: 0.1722, ROUGE-2: 0.0268, ROUGE-L: 0.1060, Latency: 1.570s, Perplexity: 3.8763
- **LoRA Fine-Tuned Model**: ROUGE-1: 0.2143, ROUGE-2: 0.1084, ROUGE-L: 0.1429, Latency: 1.955s, Perplexity: 11.4105

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.