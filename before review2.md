# 🛡️ AUTHENTIX AI: Architectural Reconnaissance & Pre-Review 2 Specification

> [!IMPORTANT]
> **Document Type:** Principal Architecture & Methodology Audit  
> **Target Audience:** Review-2 Academic Panel & Project Mentors  
> **Mission:** To provide an authoritative, visually structured, and exhaustive defense of the Authentix AI system architecture, data processing, and ML methodology.

---

## 🛠️ SECTION 1: Post-Download Data Preparation, Cleaning & Outlier Filtering

Raw datasets (Kaggle zips, Google Drive folders) are inherently noisy and dangerous for training. Feeding raw data directly into PyTorch DataLoaders leads to corrupted tensors, gradient explosion, shortcut learning, and data leakage. We use a **strict deterministic pipeline** to sanitize data *before* it touches any training script. This forms Milestone **M0.5: Dataset Curation, Cleaning & Preprocessing Gate**.

### 🖼️ 1.1 Image Forensics (GenImage, Synthbuster, CIFAKE)

| Step | Action | Parameter / Threshold | Rationale (Why we do it) |
| :--- | :--- | :--- | :--- |
| **1. Integrity** | `PIL.Image.verify()` | Checks for corrupted headers | Drops incomplete/broken downloads that crash training loops. |
| **2. Deduplication** | `pHash` calculation | Hamming distance $\le 4$ | Prevents **Data Leakage** by guaranteeing no near-identical images exist between train and test splits. |
| **3. Dimensions** | Aspect Ratio & Size | $0.5 \le W/H \le 2.0$, Min $256\times 256$ | Prevents severe visual distortion and loss of artifact fidelity during tensor resizing. |
| **4. Channels** | RGB Standardization | Convert to 3-Channel | Strips RGBA (Alpha) and Grayscale to ensure matrix dimension compatibility for CNNs. |
| **5. Normalization**| ImageNet Stats | Mean: `[0.48, 0.45, 0.40]` | Standardizes pixel distributions for faster optimizer convergence. |

### 🎬 1.2 Video Forensics (FaceForensics++, DFDC Sample)

| Step | Action | Parameter / Threshold | Rationale (Why we do it) |
| :--- | :--- | :--- | :--- |
| **1. Container** | `ffprobe` execution | Valid `.mp4` headers | Drops corrupted video streams and unreadable codecs. |
| **2. Sampling** | Temporal Extraction | 1 frame per 15 (~2 fps), Max 30 | Processing 30fps is computationally impossible locally. Normalizes sequence lengths. |
| **3. Extraction** | MTCNN Face Crop | Spatial Margin = $1.3\times$ bbox | Captures surrounding edge artifacts (where deepfake blending often fails). |
| **4. Outliers** | Laplacian Variance | $\sigma^2 \ge 100.0$ | Discards excessively blurry frames that confuse the optimizer and destroy gradients. |
| **5. Resize** | Crop Normalization | $224 \times 224$ | Prepares standard tensor sizes for the vision backbone. |

### 🌐 1.3 URL Phishing (Malicious URLs, Tranco)

| Step | Action | Parameter / Threshold | Rationale (Why we do it) |
| :--- | :--- | :--- | :--- |
| **1. Parsing** | `tldextract` | Accurate domain splitting | Separates TLDs correctly (e.g., `co.uk`), preventing naive string-split errors. |
| **2. Length** | Bound filtering | $10 \le \text{length} \le 1024$ | Drops edge-case strings that are not valid, actionable URLs. |
| **3. Dedup** | Exact string match | Unique URLs only | Prevents class imbalance and data leakage from repeated domains. |
| **4. Feature Eng.**| 25+ Lexical features | Entropy, digit ratio, etc. | Converts strings to discrete numerical feature vectors for XGBoost. |
| **5. Outliers** | `StandardScaler` cap | $> 3\sigma$ standard deviations | **Prevents gradient explosion** in tree-based models by capping extreme feature outliers. |

---

## 📅 SECTION 2: Milestone M4 (Image Forensics) Deep-Dive & Timeline

> [!TIP]
> **Total Timeline:** 2.5 to 3.5 Weeks. This is the critical path for the Tier 1 Image Forensics pipeline.

### ⏱️ Weekly Execution Schedule

*   **🟢 Week 1: Data Curation & Baselines**
    *   Execute M0.5 data cleaning: `PIL.verify()`, pHash deduplication, train/val/test splitting.
    *   Implement frozen **CLIP ViT-L/14** zero-shot feature extraction baseline to establish a performance floor.
*   **🔵 Week 2: EfficientNet-B4 Fine-Tuning**
    *   Fine-tune on GenImage. Train on 2 generators (e.g., SD1.5 + Midjourney) and **hold out 2 generators** (e.g., GLIDE, ADM) for unseen evaluation.
    *   *Parameters:* BCE loss, AdamW optimizer ($\text{lr}=10^{-4}$), Cosine decay scheduling.
    *   Monitor validation loss to prevent overfitting.
*   **🟠 Week 3: Explainability Integration (Grad-CAM)**
    *   Attach PyTorch backward hooks to the final convolutional block (`features.8`).
    *   Generate activation heatmaps, upsample to $224 \times 224$, and apply OpenCV `JET` colormap.
    *   Conduct IoU (Intersection over Union) evaluation against manual bounding box rubrics to verify model attention.
*   **🟣 Week 4: Evaluation & Packaging**
    *   Cross-generator evaluation using the held-out **Synthbuster** dataset (strictly excluding overlapping subsets like SD1.4).
    *   Perform latency measurements (inference speed).
    *   Package inference outputs into the canonical `EvidenceRecord` JSON schema.

---

## ⚖️ SECTION 3: Industrial-Grade Dataset Sizing vs. "Toy Scale"

### 🚀 Transfer Learning vs. Scratch Training
Training a CNN from scratch requires millions of images, massive cluster compute, and months of time. **Authentix AI strictly uses Transfer Learning.** We utilize a pre-trained EfficientNet-B4 backbone (which already understands edges, textures, and objects from ImageNet) and *fine-tune* only the top layers to recognize synthetic noise. This reduces required data by 90% and local compute time to mere hours.

### 📊 Exact Sample Budgeting Table
*Sized to guarantee statistical significance ($p < 0.001$) while remaining locally computable on consumer hardware.*

| Modality | Total Target Size | Train Split | Validation | Test / Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Image (Tier 1)** | ~75,000 images | 50k (25k Real / 25k AI) | 10k | 15k (Unseen Synthbuster) |
| **Video (Tier 2)** | ~600 videos | 40k face crops | 10k | 10k |
| **URL (Tier 2)** | ~160,000 URLs | 120k (60k Phish / 60k Safe) | 20k | 20k |
| **Evidence Fusion**| ~15,000 vectors | 10k (via 5-Fold OOF) | 2.5k | 2.5k |

> [!IMPORTANT]
> **Compute Feasibility:** This transfer learning strategy allows the entire training pipeline to run on a single RTX 3090/4090 or cloud T4/A100, completing within **4 to 7 hours**.

---

## ☁️ SECTION 4: Production Inference Architecture & Deployment Reality

### 📦 Web Server Footprint (< 200 MB)
**Zero raw training datasets (160+ GB) are deployed to the web server.** The raw data is mathematically compressed into lightweight, compiled model weights during offline training:

*   📄 `efficientnet_b4_genimage.pt` (~75 MB)
*   📄 `dfdc_efficientnet_b4.pt` (~75 MB)
*   📄 `url_xgboost_model.json` (~2 MB)
*   📄 `fusion_xgboost.json` + `platt_params.json` (< 1 MB)

*Total API Server RAM/VRAM requirement: < 200 MB for model weights.*

### 🔄 Step-by-Step Inference Flow

1. **Client Upload:** User submits a file/URL via the React UI.
2. **API Gateway Sanitization:** FastAPI intercepts. Validates magic bytes via `python-magic` and enforces strict file size limits (e.g., 10MB image, 50MB video) to prevent DoS. SSRF guards block malicious network probes.
3. **In-Memory Decoding:** Payload decoded into tensors.
4. **Fast Forward Pass:** Tensors passed through resident loaded models (<100ms for images).
5. **Evidence Extraction:** Confidence bounds and feature maps extracted.
6. **Fusion & Calibration:** XGBoost meta-learner fuses raw scores; Platt Scaling converts logits to true calibrated probabilities.
7. **Explainability Generation:** Grad-CAM heatmaps and SHAP feature importances generated.
8. **Logging & Delivery:** Verdict and metadata logged to PostgreSQL; JSON response delivered to UI for report rendering.

---

## 🏗️ SECTION 5: Architectural Gap Findings & Decision Options

During the system audit, four critical gaps were identified. Below are the decision matrices providing multiple engineering solutions for the team.

### 1. Data Cleaning & Validation Architecture
*Gap: Raw data fed directly to models risks corrupted tensors and slow data loading.*
* **Option A: Standalone Offline Preprocessing Scripts (Recommended).** Run a one-time script that sanitizes, crops, and deduplicates all data, saving it to a clean `data/processed/` directory. *Pros: Fast training, predictable. Cons: Uses 2x disk space.*
* **Option B: Lazy Manifest-Driven PyTorch DataLoaders.** Clean metadata only (CSV/JSON), and apply OpenCV filters dynamically during the PyTorch `__getitem__` call. *Pros: Saves disk space. Cons: Severely bottlenecks GPU during training.*
* **Option C: Precomputed Feature/Embedding Stores.** Run data through frozen backbones and save `.pt` tensor embeddings. *Pros: Blazing fast XGBoost/Linear probing. Cons: Prevents fine-tuning the vision backbone.*

### 2. Fusion Training Data Generation & Leakage Elimination
*Gap: If the meta-model trains on the CNN's predictions of its own training data, it learns overconfidence (Stacking Leakage).*
* **Option A: 5-Fold Stratified Out-of-Fold (OOF) Stacking (Recommended).** Train the CNN 5 times on 80% splits, inferring on the remaining 20%. Concatenate the 20% chunks to train the fusion model. *Pros: Mathematically rigorous, maximizes data. Cons: 5x training cost.*
* **Option B: Dedicated 3-Way Independent Split.** Split data strictly into Train (CNN) / Fusion-Calibrate / Test. *Pros: Simplest to implement, fast. Cons: Reduces data available for the base CNN.*
* **Option C: Parametric Synthetic Evidence Generation.** Model CNN outputs as Gaussian distributions and sample synthetic evidence vectors to train the fusion layer. *Pros: Zero leakage, no retraining. Cons: High risk of distribution mismatch with real data.*

### 3. Document Forensics Scope vs. Timeline Risk
*Gap: Training custom YOLOv8 models for seal/signature detection requires massive manual annotation, risking the core timeline.*
* **Option A: DocTamper CNN + OCR + Metadata (Drop YOLO) (Recommended).** Focus entirely on pixel-level manipulation (DocTamper) and metadata/OCR forensics. Drop seal verification. *Pros: Achievable in V1 timeline. Cons: Less feature-complete.*
* **Option B: Zero-Shot Open-Vocabulary Object Grounding (Grounding-DINO).** Prompt a foundational model with "signature" or "official seal" without training. *Pros: No annotation needed. Cons: Slow inference, high VRAM, unpredictable accuracy.*
* **Option C: Mini YOLOv8 Custom Annotation.** Hand-annotate 500 documents for a lightweight YOLO model. *Pros: Highly accurate. Cons: Guaranteed 2-3 week delay for manual labeling.*

### 4. Production Model Execution & Server RAM/VRAM Management
*Gap: Loading multiple PyTorch models into a basic web server risks Out-Of-Memory (OOM) crashes under concurrent load.*
* **Option A: CPU-Optimized ONNX Runtime with INT8 Quantization (Recommended).** Export PyTorch models to ONNX and quantize to 8-bit integers. *Pros: Massive memory reduction, runs fast on cheap CPU servers. Cons: Slight drop in accuracy (1-2%).*
* **Option B: Resident GPU Model Pool.** Keep PyTorch models loaded in GPU VRAM and use asynchronous batching. *Pros: Fastest possible inference. Cons: Requires expensive cloud GPU hosting.*
* **Option C: Lazy Dynamic Model Loading with LRU Cache.** Load models from disk only when requested, evicting least-used models. *Pros: Lowest memory baseline. Cons: Terrible latency (cold starts) for the first request.*

---

## 🛡️ SECTION 6: Reviewer Q&A Defense Matrix

| Skeptical Reviewer Question | Your Authoritative Defense |
| :--- | :--- |
| **"Why not just build a single end-to-end multimodal transformer model?"** | End-to-end models suffer from catastrophic forgetting and are nearly impossible to debug or interpret. Our modular ensemble allows isolated updates, hardware scaling, and explicit feature-level explainability (SHAP/Grad-CAM) per modality, which is critical for a forensic tool. |
| **"What happens if a generative AI model is completely unseen (e.g., Midjourney v7)?"** | This is exactly why we use a dual-model approach and Platt Scaling. If the CNN fails on a novel texture, it outputs low confidence. Platt Scaling calibrates this into a true probability. If the risk score falls between $0.30$ and $0.70$, the system safely outputs `INCONCLUSIVE` rather than making a false, overconfident assertion. |
| **"How do you mathematically prove your fusion model is not leaking training data?"** | We prevent stacking leakage by utilizing **Out-of-Fold (OOF) predictions**. The base classifiers generate training data for the fusion model strictly on hold-out validation sets that they did not see during their own backpropagation. Thus, the fusion model only learns how the base models behave on *unseen* data. |
| **"How does this differ from commercial black-box detectors?"** | Commercial tools output a binary verdict with zero transparency. Authentix AI acts as an *assistive forensic tool*. It provides mathematically calibrated uncertainty bounds, the exact Evidence Vector (metadata, lexical, pixel), and visual tracebacks (Grad-CAM), empowering the human rather than acting as an absolute judge. |
| **"How are you testing on live web data securely?"** | Our URL extraction pipeline utilizes strict SSRF (Server-Side Request Forgery) guards. We use DNS resolution checks to block private IP ranges (`10.x.x.x`, `127.0.0.1`), preventing attackers from probing our internal server network via the URL analysis endpoint. |
