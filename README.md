# Defect Detector — Engine Part Inspection App

A Databricks App that performs AI-powered visual inspection of engine fan parts using a **two-stage pipeline**: fast CLIP embedding similarity search followed by detailed Vision LLM structural analysis.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DEFECT DETECTOR PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐    ┌────────────────────┐    ┌───────────────────────┐
  │              │    │   STAGE 1: CLIP     │    │   STAGE 2: Vision    │
  │ Input Image  │───▶│   Embedding +       │───▶│   LLM Structural     │
  │              │    │   Similarity Search  │    │   Comparison         │
  └──────────────┘    └────────────────────┘    └───────────────────────┘
        │                     │                          │
        │              Top-K matches              PASS / FAIL
        │              with scores              + detailed report
        ▼                     ▼                          ▼
  ┌──────────────┐    ┌────────────────────┐    ┌───────────────────────┐
  │  UC Volume   │    │  Reference Library  │    │  databricks-claude-   │
  │  /inspect/   │    │  (in-memory cache)  │    │  sonnet-4             │
  └──────────────┘    └────────────────────┘    └───────────────────────┘
```

| Layer        | Technology                          | Purpose                                   |
|-------------|-------------------------------------|-------------------------------------------|
| Frontend    | Vue 3 (single-file SPA)             | Upload UI, mode toggle, results display   |
| Backend     | FastAPI (app.py)                    | API orchestration, image processing       |
| Stage 1     | CLIP ViT-B/32 (local inference)     | 512-dim embedding similarity search       |
| Stage 2     | databricks-claude-sonnet-4 (served) | Multi-modal structural comparison         |
| Storage     | UC Volumes via Databricks SDK       | Reference + inspection image persistence  |

---

## File Structure

```
engine-part-inspector/
├── app.py              # FastAPI backend (534 lines)
├── app.yaml            # Uvicorn server configuration
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── static/
    └── index.html      # Vue 3 frontend (single-page app)
```

---

## Implementation Steps — Detailed Walkthrough

### Step 1: App Startup — Reference Library Initialization

**File:** `app.py` lines 161–169 (`startup_event`), 106–155 (`_ensure_reference_library`)

When the app starts, the `@app.on_event("startup")` handler eagerly loads the reference library:

1. **Clear stale cache** — `shutil.rmtree("/tmp/defect-detector/references")` removes any leftover files from previous deployments to prevent using corrupted or outdated images.

2. **List Volume contents** — `w.files.list_directory_contents()` queries the UC Volume at `/Volumes/ppdemo/cv_model/manufacturing_parts/engine_fan/` to discover all `.jpg`, `.jpeg`, `.png` reference images.

3. **Download via SDK** — Each image is downloaded using `w.files.download(path)` which handles Databricks Apps' managed OAuth authentication automatically. The SDK returns a `DownloadResponse` whose `.contents` attribute is a `BinaryIO`-like stream. Files are written to `/tmp/defect-detector/references/`.

4. **File size validation** — Downloaded files under 100 bytes are rejected as likely auth/path failures.

5. **Compute CLIP embeddings** — Each downloaded image is processed through `_get_image_embedding()` (see Step 3) and the resulting 512-dim vector is stored in a local `library` list.

6. **Atomic promotion** — Only if at least one image loads successfully is the `library` list promoted to the global `_reference_library`. If all downloads fail, the global stays `None` so the next request retries rather than silently returning empty results.

### Step 2: Image Upload and Persistence

**File:** `app.py` lines 423–466 (`inspect_part` endpoint)

When a user uploads an image via `POST /api/inspect`:

1. **Local save** — The uploaded file bytes are written to `/tmp/defect-detector/inspect/{filename}`.

2. **Volume persistence (non-fatal)** — The file is also uploaded to `/Volumes/ppdemo/cv_model/manufacturing_parts/inspect/{filename}` via `w.files.upload()`. This is wrapped in a try/except so Volume write failures don't block the inspection.

3. **Mode parameter** — The `mode` Form field (`"standard"` or `"enhanced"`) determines which Stage 2 analysis runs.

### Step 3: Stage 1 — CLIP Embedding Similarity Search

**File:** `app.py` lines 175–186 (`_get_image_embedding`), 233–247 (`_find_similar_references`)

This is the fast, inexpensive screening step:

1. **Image preprocessing** — `PIL.Image.open().convert("RGB")` loads the image, then `CLIPProcessor` resizes and normalizes it to the required 224×224 input tensor.

2. **Embedding extraction** — To avoid version-dependent behavior in `transformers`, the embedding is computed manually:
   ```
   vision_model(pixel_values) → pooler_output → visual_projection → L2 normalize
   ```
   This explicitly chains the CLIP vision encoder with its linear projection head, producing a **512-dimensional unit vector** in CLIP's shared embedding space.

   > **Why not `get_image_features()`?** In some `transformers` versions, this method returns a `BaseModelOutputWithPooling` wrapper instead of a raw tensor. Our manual path avoids this inconsistency.

3. **Cosine similarity ranking** — The sample embedding is compared against every reference embedding using `np.dot(a, b) / (norm(a) * norm(b))`. The top 3 matches are returned, ranked by score.

4. **Threshold check** — The best match's similarity is compared against `SIMILARITY_THRESHOLD` (0.75). Below threshold signals the part may be a completely different type or severely damaged.

5. **Output** — All 3 ranked matches are included in the `stage1.all_matches` response for display, but only the **best match** is forwarded to Stage 2.

### Step 4: Stage 2 — Vision LLM Analysis (Standard Mode)

**File:** `app.py` lines 253–296 (`_vision_llm_compare`)

The standard 5-point structural inspection:

1. **Image encoding** — Both the sample and best-match reference are resized to max 1500px and encoded as base64 JPEG (quality 85%).

2. **Multi-modal prompt construction** — A structured prompt instructs the LLM to evaluate:
   - **Physical Condition** — cracks, dents, corrosion, discolouration
   - **Structural Integrity** — blade count, shape, hub condition
   - **Dimensional Conformity** — size/proportions vs reference
   - **Component Completeness** — all sub-components present
   - **Wear Assessment** — new through damaged

3. **Tolerant PASS/FAIL rules** — The prompt explicitly states minor cosmetic differences (scratches, light wear) are PASS. Only major structural issues trigger FAIL.

4. **API call** — The payload (text + images as `data:image/jpeg;base64,...` URLs) is sent to `databricks-claude-sonnet-4` via `w.api_client.do("POST", ...)`. This method handles the OAuth token exchange required by Databricks Apps (standard bearer tokens return 401).

5. **JSON parsing** — The LLM response is expected as raw JSON. A fallback strips ```json fences if present. On parse failure, a `REVIEW_NEEDED` result is returned.

6. **Structured output** — The LLM returns:
   ```json
   {
     "decision": "PASS",
     "confidence": 0.95,
     "condition": "new",
     "defects_found": [],
     "similarity_to_reference": "high",
     "reasoning": "...",
     "recommendations": "..."
   }
   ```

### Step 5: Stage 2 — Vision LLM Analysis (Enhanced Mode)

**File:** `app.py` lines 469–481 (`_run_enhanced_inspection`), 302–345 (`_vision_llm_crack_inspect`)

The enhanced mode adds quadrant cropping for micro-defect detection:

1. **Full-resolution encoding** — The sample image is encoded at 2000px (vs 1500px standard).

2. **Quadrant cropping** — `_crop_quadrants()` splits the image into 4 overlapping regions:
   ```
   ┌───────────┬───────────┐
   │ top_left  │ top_right │
   │     ↕ 10% overlap     │
   ├───────────┼───────────┤
   │bottom_left│bottom_rgt │
   └───────────┴───────────┘
   ```
   Each quadrant extends 10% past the midpoint in both directions, ensuring edge defects aren't missed at boundaries. Quadrants are saved as JPEG (quality 92%) to `/tmp/defect-detector/quadrants/`.

3. **6-image payload** — The LLM receives:
   - 1 full-view image (2000px)
   - 4 quadrant zooms (1500px each)
   - 1 reference image (2000px)

4. **Crack-focused prompt** — A specialized prompt instructs the model to look for hairline cracks, stress marks, pitting, heat discolouration, and asymmetry. "Even a single hairline crack means FAIL."

5. **Enhanced output fields** — In addition to standard fields, enhanced mode returns:
   ```json
   {
     "crack_detected": true,
     "crack_details": "Hairline crack on blade 3, top-left quadrant, ~2mm, oriented radially",
     "quadrant_analysis": {
       "top_left": "Clean, no defects detected",
       "top_right": "Minor surface scratch, cosmetic only",
       "bottom_left": "Hairline crack visible on blade edge",
       "bottom_right": "No anomalies"
     }
   }
   ```

### Step 6: Response Assembly

**File:** `app.py` lines 484–530 (`_build_response`, `_build_stage2`)

The final JSON response combines both stages:

```json
{
  "mode": "standard",
  "uploaded_image": { "filename": "...", "volume_path": "...", "image_url": "..." },
  "stage1": {
    "description": "CLIP Embedding Similarity Search",
    "best_match": { "image_name": "...", "similarity_score": 0.95, "image_url": "..." },
    "all_matches": [ { "rank": 1, "image_name": "...", "similarity_score": 0.95 }, ... ],
    "threshold": 0.75,
    "above_threshold": true
  },
  "stage2": {
    "description": "Vision LLM 5-Point Structural Inspection",
    "decision": "PASS",
    "confidence": 0.95,
    "condition": "new",
    "defects_found": [],
    "similarity_to_reference": "high",
    "reasoning": "...",
    "recommendations": "..."
  },
  "final_decision": "PASS"
}
```

### Step 7: Frontend Display

**File:** `static/index.html`

The Vue 3 single-page app provides:

1. **Upload zone** — Drag-and-drop or file browser with image preview.
2. **Mode toggle** — Standard / Enhanced (Quadrant) button group.
3. **Progress indicator** — Animated step dots showing Stage 1 → Stage 2 transitions.
4. **Decision banner** — Large PASS/FAIL/REVIEW_NEEDED with confidence, condition, and reference similarity pills.
5. **Stage 1 results** — Side-by-side uploaded vs best match images + ranked similarity table with color-coded score pills.
6. **Stage 2 results** — Confidence %, condition, defects tags, reasoning, recommendations.
7. **Enhanced-only sections** — Crack alert banner (red) and per-quadrant analysis grid.

---

## API Reference

| Endpoint             | Method | Description                                         |
|---------------------|--------|-----------------------------------------------------|
| `/`                 | GET    | Serves the Vue 3 frontend                           |
| `/api/health`       | GET    | CLIP/reference load status and image names           |
| `/api/debug`        | GET    | Detailed diagnostics: local files, startup errors    |
| `/api/references`   | GET    | List reference images with serving URLs              |
| `/api/image?path=`  | GET    | Serve a cached image from /tmp                       |
| `/api/inspect`      | POST   | Run inspection (multipart: `file` + `mode`)          |

### POST /api/inspect

| Parameter | Type      | Default      | Description                          |
|-----------|-----------|-------------|--------------------------------------|
| `file`    | UploadFile| required    | Image file (JPG, JPEG, PNG)           |
| `mode`    | Form str  | `"standard"` | `"standard"` or `"enhanced"`         |

---

## Configuration

| Constant                | Value                                                         | Description                    |
|------------------------|---------------------------------------------------------------|--------------------------------|
| `VOLUME_REFERENCE_PATH` | `/Volumes/ppdemo/cv_model/manufacturing_parts/engine_fan`     | Reference images location      |
| `VOLUME_INSPECT_PATH`   | `/Volumes/ppdemo/cv_model/manufacturing_parts/inspect`        | Upload destination             |
| `SIMILARITY_THRESHOLD`  | `0.75`                                                        | CLIP cosine similarity cutoff  |
| `VISION_MODEL`          | `databricks-claude-sonnet-4`                                  | Foundation Model endpoint      |

---

## Unity Catalog Permissions

The app's service principal requires:

```sql
GRANT USE CATALOG ON CATALOG ppdemo TO `<service-principal-uuid>`;
GRANT USE SCHEMA ON SCHEMA ppdemo.cv_model TO `<service-principal-uuid>`;
GRANT READ VOLUME ON VOLUME ppdemo.cv_model.manufacturing_parts TO `<service-principal-uuid>`;
GRANT WRITE VOLUME ON VOLUME ppdemo.cv_model.manufacturing_parts TO `<service-principal-uuid>`;
```

The serving endpoint `databricks-claude-sonnet-4` must be added as an app resource:
```json
{"resources": [{"name": "vision-llm", "serving_endpoint": {"name": "databricks-claude-sonnet-4", "permission": "CAN_QUERY"}}]}
```

---

## Dependencies

```
fastapi
uvicorn[standard]
transformers
torch
pillow
numpy
requests
databricks-sdk
python-multipart
```

---

## Deployment

```python
import requests

host = "<workspace-url>"
token = "<pat-token>"
resp = requests.post(
    f"https://{host}/api/2.0/apps/defect-detector/deployments",
    headers={"Authorization": f"Bearer {token}"},
    json={"source_code_path": "/Workspace/Users/praveen.ponna@databricks.com/engine-part-inspector"}
)
print(resp.json())
```

---

## Key Design Decisions

1. **No FUSE mounts** — Databricks Apps don't have `/Volumes` FUSE access. All Volume I/O uses the Databricks SDK (`w.files.download()`, `w.files.upload()`).

2. **SDK `api_client.do()` for LLM calls** — Standard bearer tokens get 401 on foundation model endpoints in Apps. The SDK's `api_client.do()` handles OAuth token exchange automatically.

3. **Manual CLIP projection** — `vision_model → pooler_output → visual_projection` avoids `transformers` version inconsistencies where `get_image_features()` returns wrapper objects instead of raw tensors.

4. **Atomic reference loading** — The global `_reference_library` stays `None` until at least one image loads successfully. This prevents empty-library lockout and enables automatic retry on the next request.

5. **Cache clearing on startup** — `shutil.rmtree()` removes stale `/tmp` files from previous deployments, preventing use of corrupted downloads.

---

## Troubleshooting

| Symptom                        | Diagnosis                                                   | Fix                                    |
|-------------------------------|-------------------------------------------------------------|----------------------------------------|
| Best Match = N/A              | Reference library failed to load                            | Check `/api/debug` for `startup_error` |
| Reference Library (0)          | Volume download auth failure                                | Verify UC permissions for service principal |
| Stage 2 returns REVIEW_NEEDED | LLM response wasn't valid JSON                              | Check app logs for raw LLM response    |
| 401 on Vision LLM              | Serving endpoint not added as app resource                  | PATCH app resources with CAN_QUERY     |
| Low similarity for same image  | Missing `visual_projection` step or stale /tmp cache        | Ensure manual projection path is used  |
