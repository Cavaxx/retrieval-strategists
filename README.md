# Celebrity Image Retrieval Across Domains

**Comparing general-purpose vision models, metric learning, and specialized face embeddings for celebrity re-identification.**

Machine Learning course team project by **Retrieval Strategists**. Given a celebrity photograph, the system returns the ten highest-ranked gallery images using embedding similarity, while handling unrelated identities and synthetic-face distractors.

**Best reported result: 96.2% Top-1 accuracy and 963.8 / 1,000 on the official competition test split**, using a frozen InsightFace model.

This repository is Davide Cavallaro's portfolio fork of the [original team repository](https://github.com/Wardy48/retrieval-strategists). The results and implementation are collaborative work; individual contributions are listed below.

## Project overview

Image classification assigns a label to an image. Image retrieval instead asks: **which gallery images show the same person as this query?** This requires representations that preserve identity despite changes in pose, lighting, background, and image appearance.

We compared four approaches, from an ImageNet-pretrained baseline to a specialized face recognition model. Our work combined dataset preparation, identity-disjoint evaluation, model fine-tuning, quantitative benchmarking, and qualitative error analysis.

**Project stack:** Python, PyTorch, Torchvision, InsightFace, OpenCLIP, NumPy, OpenCV, PySpark, Matplotlib, and Jupyter.

## Results

The following results are reported in Table 1 of our final course report. They are team results on the **official competition test split**, rather than newly reproduced measurements from this fork.

| Model | Approach | Top-1 | Top-5 | Top-10 | Score / 1,000 |
| --- | --- | ---: | ---: | ---: | ---: |
| ResNet-50 | Frozen ImageNet features | 7.6% | 20.2% | 26.6% | 132.8 |
| ResNet-50 + ArcFace | Fine-tuned identity embeddings | 49.0% | 69.0% | 79.0% | 580.0 |
| CLIP ViT-B/32 | Zero-shot image embeddings | 57.6% | 76.0% | 83.0% | 656.6 |
| **InsightFace `buffalo_l`** | **Frozen, specialized face embeddings** | **96.2%** | **96.6%** | **96.8%** | **963.8** |

Top-k accuracy measures whether at least one image of the correct identity appears among the first k results. The competition score uses accuracies expressed as fractions:

```text
Score = 600 × Top-1 + 300 × Top-5 + 100 × Top-10
```

The strongest model improved Top-1 accuracy by **88.6 percentage points** over the baseline. In this benchmark, specialized face embeddings outperformed both our fine-tuned CNN and the general-purpose CLIP encoder. Because the systems differ in pretraining, architecture, and preprocessing, this comparison does not isolate the effect of alignment or loss function alone.

## Method

1. **Prepare query and gallery sets.** Keep identities separate across the custom training, validation, and test partitions.
2. **Extract embeddings.** Use ResNet-50, the custom ArcFace model, CLIP, or InsightFace. The InsightFace pipeline includes face detection and landmark alignment, selecting the largest detected face when several are present.
3. **Normalize and rank.** L2-normalize embeddings, compute cosine similarity through a matrix product, and retrieve the top ten gallery matches with `torch.topk`.
4. **Evaluate.** Compare Top-1, Top-5, Top-10, the composite score, and visual retrieval examples.

The custom ResNet-50 + ArcFace model uses a 512-dimensional projection head, AdamW, and a backbone learning rate ten times lower than the head learning rate. The report describes 25 training epochs, batch size 64, a three-epoch warmup, and cosine learning-rate decay. Experiments were conducted on an NVIDIA Tesla V100 with 16 GB of allocated VRAM.

## Data and evaluation scope

| Dataset | Composition and purpose |
| --- | --- |
| Custom v1 | 6,000 images combining CelebA identities, cartoon distractors, and entirely synthetic faces |
| Custom v2 | 30,000 images with the same category proportions |
| Custom v3 | v1 plus ten real queries and ten corresponding face-reenactment gallery images for an exploratory domain-shift check |
| Competition data pool | 6,000 images: 250 celebrity identities with 20 real images each, plus 1,000 entirely synthetic distractors |

The custom v1/v2 datasets use an 80/10/10 split with celebrity identities isolated across partitions. Their images comprise two-thirds celebrity photographs, one-sixth cartoons, and one-sixth entirely synthetic faces.

**Evaluation boundary:** the official benchmark tested real-query to real-gallery identity retrieval in the presence of synthetic distractors. It did **not** quantitatively test retrieval of face-reenacted versions of the same person. The small v3 experiment is exploratory and should not be interpreted as broad validation of real-to-synthetic identity matching.

## My contribution — Davide Cavallaro

I served as the **primary academic author and compiler of the final report**, turning the team's model-development summaries and dataset documentation into a cohesive account of the research.

- Drafted the Introduction, Method, Results and Ablations, Qualitative Analysis, and Conclusion sections.
- Synthesized the experimental findings and documented the comparison between representation-learning approaches.
- Managed the final report structure, formatting, and LaTeX integration.

This contribution focused on **technical writing, research synthesis, and communicating ML experiments and their limitations**. Model implementation, training, and dataset engineering were led by the teammates credited below.

## Repository guide

| Path | Contents |
| --- | --- |
| [`arcface/train_arcface.py`](arcface/train_arcface.py) | Custom ResNet-50 + ArcFace training pipeline |
| [`resnet_insightface/insightface.py`](resnet_insightface/insightface.py) | InsightFace embedding extraction and retrieval |
| [`clip/clip_evaluate_zeroshot.py`](clip/clip_evaluate_zeroshot.py) | Zero-shot CLIP evaluation |
| [`clip/clip_train.py`](clip/clip_train.py) | Additional CLIP experimentation script |
| [`compare_models_v3.py`](compare_models_v3.py) | InsightFace, CLIP, and custom ArcFace comparison, JSON outputs, and retrieval visualizations |
| [`analysis_compare.ipynb`](analysis_compare.ipynb) | Comparative analysis notebook |
| [`arcface/analysis.ipynb`](arcface/analysis.ipynb) | ArcFace experiment analysis |
| [`dataset_creation_work/`](dataset_creation_work/) | Dataset download, partitioning, and competition split scripts |
| [`download_RDD_split.py`](download_RDD_split.py) | Dataset download and identity-aware partitioning with PySpark |

## Getting started

This is a research repository. Datasets and the trained custom checkpoint must be prepared separately; the checked-in `requirements.txt` does not include every experiment dependency.

### 1. Set up an environment

```bash
git clone https://github.com/Cavaxx/retrieval-strategists.git
cd retrieval-strategists
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install numpy tqdm matplotlib open_clip_torch insightface==0.7.3 opencv-python onnxruntime
```

The example installs ONNX Runtime for CPU inference. For GPU inference, configure a compatible CUDA/PyTorch environment and use the appropriate `onnxruntime-gpu` package instead. Pretrained models may download weights on first use. PySpark dataset preparation also requires Java.

### 2. Prepare the data

See [`dataset_creation_work/README.md`](dataset_creation_work/README.md) for dataset reconstruction guidance; the linked instructions require a University of Trento account. The downloader in `download_RDD_split.py` targets a dataset asset in the upstream repository's releases and depends on that asset remaining available.

For `compare_models_v3.py`, place the v3 dataset at the repository root with these paths:

| Path | Expected contents |
| --- | --- |
| `retrieval_strategists_full_dataset_in_partitions_v3/validation/query/` | Validation query images |
| `retrieval_strategists_full_dataset_in_partitions_v3/validation/gallery/` | Validation gallery images |
| `retrieval_strategists_full_dataset_in_partitions_v3/test/query/` | Test query images, when using `--split test` |
| `retrieval_strategists_full_dataset_in_partitions_v3/test/gallery/` | Test gallery images, when using `--split test` |

Celebrity filenames must preserve the identity convention, such as `5472_3.jpg`. Each gallery must contain at least ten images.

### 3. Compare pretrained models

Skip the custom model if `best_model.pth` is unavailable:

```bash
python compare_models_v3.py --skip custom --save-json --visualize
```

This evaluates InsightFace and CLIP on the local v3 validation split and writes ranked retrieval JSON files and a comparison image under `results/`. It does not recreate the official competition test results shown above.

To include the custom ArcFace model, supply a compatible `best_model.pth` at the repository root and omit `--skip custom`. Training is available through `python arcface/train_arcface.py`; review its `DATASET_VERSION` and data paths first. Its default configuration expects the separate `dataset_final/` layout.

The standalone `resnet_insightface/insightface.py` additionally imports `model_evaluator`, which is absent from this fork, and expects a `newdata/` layout. Use the comparison script as the entry point for pretrained-model comparison.

## Limitations and next steps

- **Detection failures:** the InsightFace implementation substitutes a zero vector when no face can be extracted, producing uninformative similarity scores for that image.
- **Limited validation coverage:** fixed query/gallery assignments and the small face-reenactment sample limit conclusions about robustness.
- **Pretraining and bias:** demographic performance and possible overlap between celebrity identities and pretrained-model data were not independently audited.
- **Reproducibility:** external datasets, model downloads, checkpoints, and environment configuration are required. The setup commands document the code's requirements; they are not an end-to-end reproduction guarantee.

Future work could improve failed-detection handling, rotate query selections across repeated evaluations, and expand testing of synthetic identity matching.

## Team and report

| Team member | Contribution documented in the report |
| --- | --- |
| Nary Ly | Co-led model selection, training, evaluation, ablation studies, and model documentation |
| Nardos Tunkala | Co-led model selection, training, evaluation, ablation studies, and model documentation |
| Edoardo Luigi Altieri | Led dataset curation, custom partitions, query/gallery organization, and dataset documentation |
| Davide Cavallaro | Primary report author; research synthesis, report structure, and LaTeX integration |

**Report:** *Beyond the Real: Open-Set Celebrity Re-Identification in Synthetic and Abstract Image Spaces* — Nary Ly, Edoardo Luigi Altieri, Nardos Tunkala, and Davide Cavallaro. Final course project report, Retrieval Strategists.

**Acknowledgments:** the project builds on ResNet, ArcFace, CLIP/OpenCLIP, and InsightFace, with data sourced from CelebA, Face4FairShifts, SyntheticFacesHQ, and FaceForensics. Dataset access and pretrained-model use remain subject to their respective terms.
