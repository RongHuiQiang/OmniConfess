# OmniConfess

A training-free decoding framework for mitigating cross-modal hallucinations 
in omni-modal foundation models, audited at token, layer, and trajectory 
granularities through self-confess decoding.

## Repository Contents

| Directory          | Description                                              |
|--------------------|----------------------------------------------------------|
| `OmniConfess/`     | Core implementation (Qwen-Omni and MiniCPM-Omni variants)|
| `OmniHalluBench/`  | Unified 3,540-sample omni-modal hallucination benchmark  |

## OmniConfess Framework

OmniConfess is a training-free decoding framework that audits and repairs 
hallucinations across three granularities:

- **Token-level audit** — detects token-wise inconsistency during decoding
- **Layer-level audit** — monitors cross-layer drift in hidden states
- **Trajectory-level audit** — controls error accumulation along the generation trajectory

Two model variants are provided under `OmniConfess/`:
- `omniconfess_qwen.py` — Qwen-Omni adaptation
- `omniconfess_minicpm.py` — MiniCPM-Omni adaptation

## OmniHalluBench

A unified benchmark spanning omni-modal inputs:

| Dataset    | Modality       | Description                           |
|------------|----------------|---------------------------------------|
| AVHBench   | Video + Audio  | Audio-visual hallucination benchmark  |
| CMM        | Video + Audio  | Cross-modal multimodal hallucination  |
| HaloQuest  | Image          | Visual hallucination evaluation       |
| PHD        | Image          | Image hallucination benchmark         |
| PubMedQA   | Text           | Biomedical question answering         |
| RAGTruth   | Text           | RAG hallucination benchmark           |

We release the unified annotation JSONs under `OmniHalluBench/`. 
Multimodal media files (images, videos, audio) should be obtained from the 
corresponding original public sources; relative paths in each JSON's media 
fields indicate the expected layout.

## Baselines

The `OmniConfess/methods/` directory provides implementations of comparison 
methods (Self-Consistency, Guided Decoding, Prompt-based Decoding, 
Tree-of-Thought, DoLa, VCD) for both Qwen-Omni and MiniCPM-Omni.

## Evaluation

Evaluation scripts are in `OmniConfess/eval/`:
- `single_eval.py` — per-sample evaluation
- `final_eval.py` — aggregated metrics

The LLM-as-a-judge evaluation reads the API key from the environment variable 
`ZHIPU_API_KEY` (or substitute with any compatible LLM API).

## Installation

```bash
pip install -r requirements.txt
```

## License

MIT License. See `LICENSE` for details.
