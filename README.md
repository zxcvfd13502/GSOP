# Beyond Token Pruning: Operation Pruning in Vision-Language Models

A **tuning-free** VLM/MLLM inference acceleration framework that **searches to prune operations rather than tokens**.

---

## 🔧 Installation

```bash
conda create -n gsop python=3.10 -y
conda activate gsop

cd lmms-eval
pip install -e .

cd ../LLaVA
pip install -e .

pip install easydict
```

For additional setup instructions, please refer to:

- [LLaVA Repository](https://github.com/haotian-liu/LLaVA)
- [lmms-eval Repository](https://github.com/EvolvingLMMs-Lab/lmms-eval)

---

## 🚀 Usage

### Inference

```bash
bash scripts/gsop_inference.sh
```

### Search

```bash
bash scripts/gsop_search.sh
```

---

Some benchmarks (e.g., **TextVQA**) may produce results that differ from commonly reported metrics when run on `lmms-eval`. Please follow the evaluation setup detailed in [Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) for those benchmarks.
