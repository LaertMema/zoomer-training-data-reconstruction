# Zoomer: White-Box Model Reconstruction and Training-Set Recovery

Code-only research showcase for my solution to the **Zoomer Challenge** in the
PINE 2026 seminar. The task was to analyze an executable used in an
authentication-system challenge, recover its embedded machine-learning model,
and submit exactly 18 images that were semantically close to the model's hidden
training samples.

The challenge score was the mean cosine similarity between the submitted and
hidden samples in a separate model's embedding space. Classifier acceptance was
therefore only a filter; semantic similarity to the hidden set was the actual
objective. The final approach combined natural-image retrieval with ablation as
an image-selection strategy and achieved a reported score of **0.6657**.

## Approach

```text
challenge executable
        |
        v
probe images + dynamic analysis
        |
        v
reconstructed ResNet-18 classifier
        |
        v
public-image retrieval and positive-logit filtering
        |
        v
CLIP-guided semantic refinement
        |
        v
anchor-based single-swap ablation
        |
        v
final set of 18 candidates
```

### 1. Reconstruct the embedded model

Controlled black and white `.ras` probes and dynamic inspection with GDB exposed
the preprocessing pipeline: `[0, 1]` scaling followed by channel-wise ImageNet
Z-score normalization. In practical terms, the pipeline is:

```text
OpenCV BGR load -> resize shorter side to 256 -> center-crop 224
-> BGR to RGB -> ImageNet normalization -> NCHW float32
```

Inspection of `get_weights(const char *)`, a `weight_index` table, and a
contiguous float32 weight blob revealed a ResNet-18 backbone with a
`Linear(512, 1)` binary head. The recovered parameter names, offsets, and counts
were then mapped into a PyTorch state dictionary.

### 2. Retrieve natural-image candidates

The initial hypothesis was that the hidden samples came from public image
collections. Images from COCO, Visual Genome, Places365, and Open Images V7
were processed in batches with the reconstructed classifier. Positive-logit
images formed the retrieval pool, from which three 18-image strategies were
tested:

- highest classifier logits;
- greedy diversity with optional logit weighting; and
- one high-logit representative from each of 18 feature-space clusters.

The highest-logit strategy scored `0.5620`; the diversity strategy scored
`0.5614`. These results indicated that broad retrieval alone selected too many
unrelated visual themes.

### 3. Refine the semantic region

CLIP image and text embeddings were used to query the pool with natural-language
descriptions and focus the search on the strongest emerging theme. DINOv2
image-to-image features served as a comparison. CLIP-guided retrieval improved
the score to `0.6613`; the best reported DINOv2 result on COCO was `0.5744`.

### 4. Optimize the final set with ablation

The best semantic set became an 18-image anchor. Weak anchor slots and possible
replacements were ranked with the report's proxy score:

```text
s(x) = anchor_similarity(x)
     + 0.25 * theme_similarity(x)
     + 0.05 * normalized_classifier_logit(x)
```

Single-image swaps paired weak slots with both high-ranked *strong fillers* and
middle-ranked *neutral fillers*. Challenge feedback acted as an oracle: an
improving swap became the new anchor and the search continued until the score
plateaued. This final stage reached `0.6657`.

## Reported results

| Version | Method | Mean cosine similarity |
|---|---|---:|
| Initial | Natural-image retrieval | 0.5620 |
| Improved | CLIP-guided semantic retrieval | 0.6613 |
| Final | Anchored ablation and local search | **0.6657** |

These are hidden challenge scores, not classification accuracy. The result
shows that public-image retrieval can recover useful themes and semantically
related candidates; it does not establish exact recovery of all hidden images.

## Repository layout

```text
scripts/
  image_pipeline.py                 ImageNet preprocessing and Sun raster I/O
  reconstruct_model.py              Rebuild ResNet-18 from the flat weight blob
  preprocess_compare.py             Compare Python and executable tensors
  retrieve_natural_candidates.py    Score and select natural-image candidates
  select_retrieval_variants.py      Logit, diversity, and cluster selection
  semantic_retrieval_variants.py    CLIP/DINOv2 semantic variants
  scene_prior_retrieval.py          Text-guided scene retrieval
  openimages_strict_and_fetch.py    Strict multi-label Open Images retrieval
  mix_coco_openimages_domain.py     Controlled cross-source candidate mixing
  generate_single_swap_tests.py     Single-image neighborhood search
  generate_anchor_ablation_tests.py Strong/neutral-filler ablation generation
  score_candidates.py               Local classifier scoring and contact sheets
  feature_select_candidates.py      Feature-space selection utilities
  validate_candidates.py            Original-binary output validation
tests/
  test_image_pipeline.py            Deterministic preprocessing and raster tests
```

The original executable, recovered weights, downloaded datasets, generated
candidate images, and private challenge-evaluator integration are deliberately
not included. They are not required to review the implementation and may carry
separate ownership or access restrictions.

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Large model and dataset artifacts should remain outside version control. The
examples below use `artifacts/`, `data/`, and `outputs/`, all ignored by Git.

## Example workflow

Reconstruct the classifier from a locally recovered float32 weight blob:

```bash
python scripts/reconstruct_model.py \
  --weights artifacts/resnet18_weights.bin \
  --checkpoint artifacts/reconstructed_resnet18.pth
```

Build a positive-logit retrieval pool and a first diverse set. The
`--no-strict-audit` option skips validation against the original executable,
which is not distributed here:

```bash
python scripts/retrieve_natural_candidates.py \
  --image-root data/images \
  --checkpoint artifacts/reconstructed_resnet18.pth \
  --pool-dir outputs/retrieval_pool \
  --selected-dir outputs/retrieval_selected_18 \
  --no-strict-audit
```

Generate CLIP variants from that pool:

```bash
python scripts/semantic_retrieval_variants.py \
  --pool-dir outputs/retrieval_pool \
  --encoder clip \
  --output-root outputs/semantic_variants
```

Generate strong- and neutral-filler ablations around an 18-image anchor:

```bash
python scripts/generate_anchor_ablation_tests.py \
  --anchor-dir outputs/current_anchor \
  --replacement-roots outputs/replacement_candidates \
  --replacement-include clip \
  --output-root outputs/anchor_ablations
```

Run the lightweight local tests:

```bash
python -m unittest discover -s tests -v
```

## Limitations and future work

The method depends on coverage in public datasets. If the classifier was
trained on private images, natural-image retrieval may identify broad themes
without recovering the underlying samples. The final set was treated as the
best set found within the available retrieval pool, not as a global optimum.

Synthetic recovery was explored only briefly because of time constraints.
DeepInversion-style generation is the clearest next direction when public-data
coverage is insufficient.

## References

- K. He et al., *Deep Residual Learning for Image Recognition*, 2016.
- A. Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, 2021.
- M. Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023.
- H. H. Hoos and T. Stutzle, *Stochastic Local Search: Foundations and Applications*, 2004.
- H. Yin et al., *Dreaming to Distill: Data-Free Knowledge Transfer via DeepInversion*, 2020.

## Author

Laert Mema
