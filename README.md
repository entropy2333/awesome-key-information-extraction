# Awesome Key Infomation Extraction

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of papers about key information extraction.

Paperswithcode links will be preferred.

Welcome contributions!

## Tabel of Contents

- [Awesome Key Infomation Extraction](#awesome-key-infomation-extraction)
  - [Tabel of Contents](#tabel-of-contents)
  - [Datasets](#datasets)
  - [Survey](#survey)
  - [Toolkits](#toolkits)
  - [Models](#models)
    - [:star:LLM-Based](#starllm-based)
    - [Graph-Based](#graph-based)
    - [Transformer-Based](#transformer-based)
    - [Grid-Based](#grid-based)
    - [End-to-end](#end-to-end)
    - [Others](#others)
  - [Related Repositories](#related-repositories)


## Datasets

|     Name     | Title                                                                                                           |                                                                    Links                                                                     |
| :----------: | --------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------: |
|     DUE      | DUE: End-to-End Document Understanding Benchmark                                                                |      [[link]](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/069059b7ef840f0c74a814ec9237b6ec-Abstract-round2.html)      |
|   RVL-CDIP   | Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval                           |                 [[link]](https://paperswithcode.com/dataset/rvl-cdip)[[download]](https://www.cs.cmu.edu/~aharley/rvl-cdip/)                 |
|    SROIE     | ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction                                         |                 [[link]](https://paperswithcode.com/dataset/sroie)[[download]](https://rrc.cvc.uab.es/?ch=13&com=downloads)                  |
|    FUNSD     | FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents                                              |                   [[link]](https://paperswithcode.com/dataset/funsd)[[download]](https://guillaumejaume.github.io/FUNSD/)                    |
|    XFUND     | XFUND: A Multilingual Form Understanding Benchmark                                                              |                                               [[link]](https://github.com/doc-analysis/XFUND)                                                |
|     CORD     | CORD: A Consolidated Receipt Dataset for Post-OCR Parsing                                                       |                                                  [[link]](https://github.com/clovaai/cord)                                                   |
|    EPHOIE    | Towards Robust Visual Information Extraction in Real World: New Dataset and Novel Solution                      |                                                 [[link]](https://github.com/HCIILAB/EPHOIE)                                                  |
|    EATEN     | EATEN: Entity-aware Attention for Single Shot Visual Text Extraction                                            |                                                [[link]](https://github.com/beacandler/EATEN)                                                 |
| Train Ticket | PICK: Processing Key Information Extraction from Documents using Improved Graph Learning-Convolutional Networks |    [[link]](https://github.com/wenwenyu/PICK-pytorch)[[download]](https://drive.google.com/file/d/1o8JktPD7bS74tfjz-8dVcZq_uFS6YEGh/view)    |
|     POIE     | Visual Information Extraction in the Wild: Practical Dataset and End-to-end Solution                            | [[link]](https://github.com/jfkuang/CFAM)[[download]](https://drive.google.com/file/d/1eEMNiVeLlD-b08XW_GfAGfPmmII-GDYs/view?usp=share_link) |

## Survey

| Year | Title                                                   |                                      Links                                       |
| ---- | ------------------------------------------------------- | :------------------------------------------------------------------------------: |
| 2023 | On the Hidden Mystery of OCR in Large Multimodal Models | [[link]](https://paperswithcode.com/paper/on-the-hidden-mystery-of-ocr-in-large) |
| 2021 | Document AI: Benchmarks, Models and Applications        |   [[link]](https://paperswithcode.com/paper/document-ai-benchmarks-models-and)   |

## Toolkits


| Year | Title                                                                                |                                                   Links                                                   |
| ---- | ------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------: |
| 2022 | DavarOCR: A Toolbox for OCR and Multi-Modal Document Understanding                   | [[paper]](https://arxiv.org/pdf/2207.06695v1.pdf)[[code]](https://github.com/hikopensource/davar-lab-ocr) |
| 2021 | MMOCR: A Comprehensive Toolbox for Text Detection, Recognition and Understanding     |      [[paper]](https://arxiv.org/pdf/2108.06543v1.pdf)[[code]](https://github.com/open-mmlab/mmocr)       |
| 2020 | PP-OCR: A Practical Ultra Lightweight OCR System                                     |   [[paper]](https://arxiv.org/pdf/2009.09941v3.pdf)[[code]](https://github.com/PaddlePaddle/PaddleOCR)    |
| 2024 | ANLS* -- A Universal Document Processing Metric for Generative Large Language Models | [[paper]](https://arxiv.org/pdf/2402.03848)[[code]](https://github.com/deepopinion/anls_star_metric)      |

## Models

### :star:LLM-Based

| Pub.  | Year  | Title                                                                                                     |                                           Links                                            |
| :---: | :---: | --------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------: |
| ICML  | 2023  | BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models    |     [[link]](https://paperswithcode.com/paper/blip-2-bootstrapping-language-image-pre)     |
| Arxiv | 2023  | InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning                      |   [[link]](https://paperswithcode.com/paper/instructblip-towards-general-purpose-vision)   |
| Arxiv | 2023  | MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models                    |       [[link]](https://paperswithcode.com/paper/minigpt-4-enhancing-vision-language)       |
| Arxiv | 2023  | Visual Instruction Tuning                                                                                 |            [[link]](https://paperswithcode.com/paper/visual-instruction-tuning)            |
| Arxiv | 2023  | Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond      |    [[link]](https://paperswithcode.com/paper/qwen-vl-a-frontier-large-vision-language)     |
| Arxiv | 2023  | mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality                               |     [[link]](https://paperswithcode.com/paper/mplug-owl-modularization-empowers-large)     |
| Arxiv | 2023  | mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding                      |    [[link]](https://paperswithcode.com/paper/mplug-docowl-modularized-multimodal-large)    |
| Arxiv | 2023  | mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration                  |  [[link]](https://paperswithcode.com/paper/mplug-owl2-revolutionizing-multi-modal-large)   |
| Arxiv | 2023  | Otter: A Multi-Modal Model with In-Context Instruction Tuning                                             |    [[link]](https://paperswithcode.com/paper/otter-a-multi-modal-model-with-in-context)    |
| Arxiv | 2023  | UReader: Universal OCR-free Visually-situated Language Understanding with Multimodal Large Language Model |  [[link]](https://paperswithcode.com/paper/ureader-universal-ocr-free-visually-situated)   |
| Blog  | 2023  | Fuyu-8B: A Multimodal Architecture for AI Agents                                                          | [[blog]](https://www.adept.ai/blog/fuyu-8b)[[model]](https://huggingface.co/adept/fuyu-8b) |


### Graph-Based
|     Pub.     | Year  | Title                                                                                                           |                                          Links                                           |
| :----------: | :---: | --------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------: |
|    ICDAR     | 2023  | LayoutGCN: A Lightweight Architecture for Visually Rich Document Understanding                                  |        [[paper]](https://link.springer.com/chapter/10.1007/978-3-031-41682-8_10)         |
| ACL-Findings | 2021  | Spatial Dependency Parsing for Semi-Structured Document Information Extraction                                  |  [[link]](https://paperswithcode.com/paper/spatial-dependency-parsing-for-2d-document)   |
|    Arxiv     | 2021  | Spatial Dual-Modality Graph Reasoning for Key Information Extraction                                            | [[link]](https://paperswithcode.com/paper/spatial-dual-modality-graph-reasoning-for-key) |
|     ICPR     | 2020  | PICK: Processing Key Information Extraction from Documents using Improved Graph Learning-Convolutional Networks |  [[link]](https://paperswithcode.com/paper/pick-processing-key-information-extraction)   |

### Transformer-Based

|  Pub.  | Year  | Title                                                                                                               |                                                                      Links                                                                      |
| :----: | :---: | ------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------: |
|  ACL   | 2022  | LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding          |                                 [[link]](https://paperswithcode.com/paper/lilt-a-simple-yet-effective-language)                                 |
|  ACL   | 2022  | FormNet: Structural Encoding beyond Sequential Modeling in Form Document Information Extraction                     |                            [[link]](https://paperswithcode.com/paper/formnet-structural-encoding-beyond-sequential)                             |
|  CVPR  | 2022  | XYLayoutLM: Towards Layout-Aware Multimodal Networks For Visually-Rich Document Understanding                       |                              [[link]](https://paperswithcode.com/paper/xylayoutlm-towards-layout-aware-multimodal)                              |
| Arxiv  | 2022  | LoPE: Learnable Sinusoidal Positional Encoding for Improving Document Transformer Model                             |                            [[link]](https://paperswithcode.com/paper/lope-learnable-sinusoidal-positional-encoding)                             |
| Arxiv  | 2022  | LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking                                        |                             [[link]](https://paperswithcode.com/paper/layoutlmv3-pre-training-for-document-ai-with)                             |
| Arxiv  | 2022  | ERNIE-Layout: Layout-Knowledge Enhanced Multi-modal Pre-training for Document Understanding                         |                             [[link]](https://paperswithcode.com/paper/ernie-layout-layout-knowledge-enhanced-multi)                             |
|  AAAI  | 2022  | BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents |                               [[link]](https://paperswithcode.com/paper/bros-a-layout-aware-pre-trained-language)                               |
| ICDAR  | 2021  | ViBERTgrid: A Jointly Trained Multi-Modal 2D Document Representation for Key Information Extraction from Documents  | [[link]](https://paperswithcode.com/paper/vibertgrid-a-jointly-trained-multi-modal-2d)[[code]](https://github.com/ZeningLin/ViBERTgrid-PyTorch) |
| Arxiv  | 2021  | TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models                                      |                              [[link]](https://paperswithcode.com/paper/trocr-transformer-based-optical-character)                               |
| ACM-MM | 2021  | StrucTexT: Structured Text Understanding with Multi-Modal Transformers                                              |                             [[link]](https://paperswithcode.com/paper/structext-structured-text-understanding-with)                             |
|  ACL   | 2021  | LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding                                       |                               [[link]](https://paperswithcode.com/paper/layoutlmv2-multi-modal-pre-training-for)                                |
|  KDD   | 2020  | LayoutLM: Pre-training of Text and Layout for Document Image Understanding                                          |                             [[link]](https://paperswithcode.com/paper/layoutlm-pre-training-of-text-and-layout-for)                             |


### Grid-Based

| Pub.  | Year  | Title                                                                                                              |                                          Links                                          |
| :---: | :---: | ------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------: |
| ICDAR | 2021  | ViBERTgrid: A Jointly Trained Multi-Modal 2D Document Representation for Key Information Extraction from Documents | [[link]](https://paperswithcode.com/paper/vibertgrid-a-jointly-trained-multi-modal-2d)  |
| ICDAR | 2021  | VisualWordGrid: Information Extraction From Scanned Documents Using A Multimodal Approach                          | [[link]](https://paperswithcode.com/paper/visualwordgrid-information-extraction-from-1) |
| NIPS  | 2019  | BERTgrid: Contextualized Embedding for 2D Document Representation and Understanding                                |   [[link]](https://paperswithcode.com/paper/bertgrid-contextualized-embedding-for-2d)   |
| EMNLP | 2018  | Chargrid: Towards Understanding 2D Documents                                                                       | [[link]](https://paperswithcode.com/paper/chargrid-towards-understanding-2d-documents)  |

### End-to-end

|  Pub.  | Year  | Title                                                                                |                                          Links                                           |
| :----: | :---: | ------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------: |
| ICDAR  | 2023  | Visual Information Extraction in the Wild: Practical Dataset and End-to-end Solution |   [[link]](https://paperswithcode.com/paper/visual-information-extraction-in-the-wild)   |
|  ICML  | 2023  | Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding      | [[link]](https://paperswithcode.com/paper/pix2struct-screenshot-parsing-as-pretraining)  |
|  ECCV  | 2022  | OCR-free Document Understanding Transformer                                          |   [[link]](https://paperswithcode.com/paper/donut-document-understanding-transformer)    |
| Arxiv  | 2022  | TRIE++: Towards End-to-End Information Extraction from Visually Rich Documents       |      [[link]](https://paperswithcode.com/paper/trie-towards-end-to-end-information)      |
|  ICCV  | 2021  | DocFormer: End-to-End Transformer for Document Understanding                         | [[link]](https://paperswithcode.com/paper/docformer-end-to-end-transformer-for-document) |
| ACM-MM | 2020  | TRIE: End-to-End Text Reading and Information Extraction for Document Understanding  | [[link]](https://paperswithcode.com/paper/trie-end-to-end-text-reading-and-information)  |
| ICDAR  | 2019  | EATEN: Entity-aware Attention for Single Shot Visual Text Extraction                 | [[link]](https://paperswithcode.com/paper/eaten-entity-aware-attention-for-single-shot)  |

### Others

| Pub.  | Year  | Title                                                                                                  |                                      Links                                       |
| :---: | :---: | ------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------: |
| ICDAR | 2023  | Information Extraction from Documents: Question Answering vs Token Classification in real-world setups | [[link]](https://paperswithcode.com/paper/information-extraction-from-documents) |



## Related Repositories

- https://paperswithcode.com/task/key-information-extraction
- https://github.com/tstanislawek/awesome-document-understanding/blob/main/topics/kie/README.md
- :star:https://github.com/SCUT-DLVCLab/Document-AI-Recommendations#vie

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=entropy2333/awesome-key-information-extraction&type=Date)](https://star-history.com/#entropy2333/awesome-key-information-extraction&Date)
