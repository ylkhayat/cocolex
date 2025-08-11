# CoCoLex: Confidence-guided Copy-based Decoding for Grounded Legal Text Generation

**CoCoLex** is a decoding strategy for improving **faithfulness and factual accuracy** in **legal text generation**. It enhances standard large language models (LLMs) by dynamically combining model-generated outputs with context-based copying, guided by model confidence scores.

📌 Official implementation of our ACL 2025 paper:  
[**CoCoLex: Confidence-guided Copy-based Decoding for Grounded Legal Text Generation**](https://arxiv.org/pdf/2508.05534)

---

## Key Features

- ✅ Improves **faithfulness** in legal text generation
- 📚 Leverages **retrieved legal context** via copy-based decoding
- 🎯 Uses **confidence-based interpolation** of generation and copy distributions
- 🧪 Benchmarked on **five legal NLP datasets**
  - Check out all datasets hosted at https://huggingface.co/collections/ylkhayat/cocolex-generation-workshop-675b01abab4daf21781a67e4

---

## Abstract

LLMs have strong potential for legal NLP but often generate unfaithful or hallucinated text. Retrieval-Augmented Generation (RAG) adds external knowledge but doesn’t ensure it's used effectively. **CoCoLex** solves this by interpolating model predictions with a copy-based distribution over the retrieved context, guided by model confidence. Experiments show that CoCoLex improves groundedness and faithfulness in long-form legal generation.

---

## Paper and Citation

📕 **Published at**: ACL 2025  
🔗 [PDF](https://aclanthology.org/2025.acl-long.931.pdf) | [Abstract](https://aclanthology.org/2025.acl-long.931/)

```bibtex
@inproceedings{t-y-s-s-etal-2025-cocolex,
  title     = "{C}o{C}o{L}ex: Confidence-guided Copy-based Decoding for Grounded Legal Text Generation",
  author    = "T.y.s.s, Santosh and Elkhayat, Youssef Tarek and Ichim, Oana and Shetty, Pranav and Wang, Dongsheng and Ma, Zhiqiang and Nourbakhsh, Armineh and Liu, Xiaomo",
  booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year      = "2025",
  address   = "Vienna, Austria",
  publisher = "Association for Computational Linguistics",
  pages     = "19002--19018",
  url       = "https://aclanthology.org/2025.acl-long.931/",
}
```

## Usage Example

```python
from cocolex import CoCoLex

# Initialize CoCoLex model
model = CoCoLex(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    device=0
)

# Define prompts
prompts = ["Given the contract details above, summarize the obligations of each party."]
# Contexts are what will be passed to the model as context - List of strings
contexts = ["This agreement is entered into by the Parties on January 1, 2025..."]

# Datastore construction parameter (will change for CoCoLex+)
references = copy.deepcopy(contexts)

# Generate tokens
outputs = model.generate(
    prompts=prompts,
    contexts=contexts,
    references=references,
    max_length=100,
)

# Decode and print the generated text
decoded_output = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(decoded_output[0])
```

## 🧪 Experiment environment (conda) and notebook

Follow these steps to set up a minimal environment for the experiment notebook under `experiment/` and run it.

1. Create a conda env

```bash
conda create -n cocolex-exp python=3.10 -y
conda activate cocolex-exp
```

2. Install requirements for the experiment

```bash
python -m pip install --upgrade pip
pip install -r experiment/requirements.txt
```

4. Run the notebook

- Open [experiment/cocolex.ipynb](experiment/cocolex.ipynb) in VS Code or Jupyter, and select the `Python (cocolex-exp)` kernel.
- Execute the cells to reproduce the data loading and prompt construction pipeline.

Notes:

- The experiment uses public Hugging Face datasets (ylkhayat/\*-generation-workshop). Access requires internet.

### CoLex (copy-only)

```python
outputs = model.generate(
    prompts=prompts,
    contexts=contexts,
    references=references,
    max_length=100,
    lamba=0.5, # CoLex uses a fixed lambda value for copy-based distribution
)
```

### CoCoLex (confidence-guided; default)

```python
outputs = model.generate(
    prompts=prompts,
    contexts=contexts,
    references=copy.deepcopy(references),
    max_length=100,
)
```

### CoCoLex-Plus (uses chunked datastore references)

```python
# References are the datastore entries, which can be longer documents to support the full input - List of List of strings
full_contexts = [
  [
    "This agreement is entered into by the Parties on January 1, 2025. The obligations of each party are as follows: ...",
    "The contract stipulates that Party A must deliver goods by March 1, 2025, while Party B must make payment within 30 days."
  ]
]
outputs = model.generate(
    prompts=prompts,
    contexts=contexts,
    references=full_contexts,
    max_length=100,
    use_plus=True,  # Enable CoCoLex+ mode
)
```

### Ada + CoCoLex

```python
outputs = model.generate(
    prompts=prompts,
    contexts=contexts,
    references=copy.deepcopy(references),
    max_length=100,
    use_jsd=True
)
```

## Manual

### Generate Function Parameters

| Parameter                    | Type                                | Default      | Description                                                                                                           |
| ---------------------------- | ----------------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------- |
| `prompts`                    | `List[str]`                         | Required     | Input prompts to generate text from.                                                                                  |
| `contexts`                   | `List[str]`                         | Required     | Context strings that are prepended to prompts.                                                                        |
| `references`                 | `Union[List[str], List[List[str]]]` | Required     | Reference texts for datastore construction and copy-based probability calculation.                                    |
| `lamba`                      | `float`                             | `None`       | Fixed interpolation weight between model and copy distributions. If `None`, uses confidence-guided dynamic weighting. |
| `max_length`                 | `int`                               | `256`        | Maximum number of tokens to generate.                                                                                 |
| `entropy_strategy`           | `str`                               | `'exp_norm'` | Strategy for computing entropy-based confidence ('exp_norm', 'sigmoid').                                              |
| `entropy_sigmoid_threshold`  | `float`                             | `0.5`        | Threshold for sigmoid-based entropy confidence calculation.                                                           |
| `lambda_smoothing_factor`    | `float`                             | `0.3`        | Smoothing factor for temporal lambda updates.                                                                         |
| `decoding_strategy`          | `str`                               | `'greedy'`   | Token sampling strategy ('greedy' or 'top_p' or 'top_k').                                                             |
| `top_p_value`                | `float`                             | `0.9`        | Nucleus sampling probability threshold for top-p decoding.                                                            |
| `top_k_value`                | `int`                               | `20`         | Number of top tokens to consider for top-k sampling.                                                                  |
| `k`                          | `int`                               | `10`         | Number of nearest neighbors to retrieve from datastore.                                                               |
| `datastore_from_layer_index` | `int`                               | `-1`         | Model layer index to use for datastore queries (-1 = last layer).                                                     |
| `use_repetition_penalty`     | `bool`                              | `True`       | Whether to apply repetition penalty during sampling.                                                                  |
| `repetition_penalty_value`   | `float`                             | `1.5`        | Penalty factor for repeated tokens (>1.0 discourages repetition).                                                     |
| `temperature`                | `float`                             | `1.0`        | Sampling temperature for controlling randomness.                                                                      |
| `min_length_ratio`           | `float`                             | `0.1`        | Minimum generation length as ratio of `max_length`.                                                                   |
| `use_faiss`                  | `bool`                              | `False`      | Whether to use FAISS for efficient similarity search.                                                                 |
| `distance_method`            | `str`                               | `'euc'`      | Distance metric for datastore retrieval ('euc' or 'cos').                                                             |
| `use_jsd`                    | `bool`                              | `False`      | Whether to use Jensen-Shannon Divergence for distribution mixing (Ada mode).                                          |
| `use_plus`                   | `bool`                              | `False`      | Whether to use CoCoLex+ mode with chunked datastore references.                                                       |

**Returns:** `List[List[int]]` - List of generated token sequences for each input prompt.
