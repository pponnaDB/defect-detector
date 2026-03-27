import os
import io
import json
import base64
import logging
import traceback

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import requests as http_requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VOLUME_REFERENCE_PATH = "/Volumes/ppdemo/cv_model/manufacturing_parts/engine_fan"
VOLUME_INSPECT_PATH = "/Volumes/ppdemo/cv_model/manufacturing_parts/inspect"
SIMILARITY_THRESHOLD = 0.75
VISION_MODEL = "databricks-claude-sonnet-4"

LOCAL_REF_CACHE = "/tmp/defect-detector/references"
LOCAL_INSPECT_CACHE = "/tmp/defect-detector/inspect"
LOCAL_QUADRANT_CACHE = "/tmp/defect-detector/quadrants"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("defect-detector")

app = FastAPI(title="Defect Detector")

# ---------------------------------------------------------------------------
# Lazy-loaded globals
# ---------------------------------------------------------------------------
_clip_model = None
_clip_processor = None
_reference_library = None  # None = not yet attempted
_workspace_client = None
_startup_error = None  # Capture startup errors for /api/debug


def _get_workspace_client():
    global _workspace_client
    if _workspace_client is None:
        from databricks.sdk import WorkspaceClient
        _workspace_client = WorkspaceClient()
    return _workspace_client


# ---------------------------------------------------------------------------
# Volume I/O helpers  (SDK-based, no FUSE dependency)
# ---------------------------------------------------------------------------
def _list_volume_images(volume_dir):
    w = _get_workspace_client()
    names = []
    for entry in w.files.list_directory_contents(volume_dir):
        if entry.name and entry.name.lower().endswith((".jpg", ".jpeg", ".png")):
            names.append(entry.name)
    return sorted(names)


def _download_from_volume(volume_path, local_path):
    """Download a file from a UC Volume using the SDK (handles OAuth)."""
    w = _get_workspace_client()
    resp = w.files.download(volume_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    # resp.contents is a BinaryIO-like object
    if hasattr(resp, "contents"):
        data = resp.contents.read()
    elif hasattr(resp, "read"):
        data = resp.read()
    else:
        data = b"".join(resp)
    with open(local_path, "wb") as f:
        f.write(data)
    file_size = os.path.getsize(local_path)
    logger.info(f"Downloaded {volume_path} -> {local_path} ({file_size:,} bytes)")
    if file_size < 100:
        raise ValueError(
            f"Downloaded file is suspiciously small ({file_size} bytes) "
            f"- possible auth or path issue: {volume_path}"
        )
    return local_path


def _upload_to_volume(local_path, volume_path):
    w = _get_workspace_client()
    with open(local_path, "rb") as f:
        w.files.upload(volume_path, f, overwrite=True)


# ---------------------------------------------------------------------------
# CLIP model loading (lazy)
# ---------------------------------------------------------------------------
def _ensure_clip_loaded():
    global _clip_model, _clip_processor
    if _clip_model is None:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        logger.info("Loading CLIP model...")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
        logger.info("CLIP model loaded.")


def _ensure_reference_library():
    """Download reference images from Volume, compute CLIP embeddings.

    Only promotes to the global on success.  If anything fails the
    global stays None so the next request retries.
    """
    global _reference_library, _startup_error
    if _reference_library is not None:
        return
    _ensure_clip_loaded()

    import shutil
    if os.path.isdir(LOCAL_REF_CACHE):
        shutil.rmtree(LOCAL_REF_CACHE)
    os.makedirs(LOCAL_REF_CACHE, exist_ok=True)

    try:
        image_names = _list_volume_images(VOLUME_REFERENCE_PATH)
        logger.info(f"Found {len(image_names)} reference images: {image_names}")
    except Exception as e:
        _startup_error = f"list_volume_images failed: {traceback.format_exc()}"
        logger.error(_startup_error)
        return

    if not image_names:
        _startup_error = f"Volume directory is empty: {VOLUME_REFERENCE_PATH}"
        logger.error(_startup_error)
        return

    library = []
    for fname in image_names:
        try:
            local_path = os.path.join(LOCAL_REF_CACHE, fname)
            _download_from_volume(f"{VOLUME_REFERENCE_PATH}/{fname}", local_path)
            emb = _get_image_embedding(local_path)
            library.append(
                {"image_name": fname, "image_path": local_path, "embedding": emb}
            )
            logger.info(f"Indexed: {fname} (dim={len(emb)})")
        except Exception as e:
            _startup_error = f"Failed to process {fname}: {traceback.format_exc()}"
            logger.error(_startup_error)

    if library:
        _reference_library = library
        _startup_error = None
        logger.info(f"Reference library ready: {len(library)} images")
    else:
        _startup_error = "No reference images loaded successfully."
        logger.error(_startup_error)


# ---------------------------------------------------------------------------
# Startup: eagerly load references so errors surface in app logs immediately
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("=== App startup: loading reference library ===")
    try:
        _ensure_reference_library()
    except Exception as e:
        global _startup_error
        _startup_error = f"Startup exception: {traceback.format_exc()}"
        logger.error(_startup_error)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def _get_image_embedding(image_path):
    import torch
    from PIL import Image
    _ensure_clip_loaded()
    image = Image.open(image_path).convert("RGB")
    inputs = _clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        vision_outputs = _clip_model.vision_model(pixel_values=inputs["pixel_values"])
        pooled_output = vision_outputs.pooler_output
        embedding = _clip_model.visual_projection(pooled_output)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.squeeze().numpy().tolist()


def _cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _encode_image_base64(image_path, max_size=1500):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _encode_image_base64_hires(image_path, max_size=4000):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _crop_quadrants(image_path, overlap=0.1):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    ow, oh = int(w * overlap), int(h * overlap)
    mid_w, mid_h = w // 2, h // 2
    quadrants = {
        "top_left":     img.crop((0,          0,          mid_w + ow, mid_h + oh)),
        "top_right":    img.crop((mid_w - ow, 0,          w,          mid_h + oh)),
        "bottom_left":  img.crop((0,          mid_h - oh, mid_w + ow, h)),
        "bottom_right": img.crop((mid_w - ow, mid_h - oh, w,          h)),
    }
    os.makedirs(LOCAL_QUADRANT_CACHE, exist_ok=True)
    paths = {}
    for name, crop_img in quadrants.items():
        tmp = os.path.join(LOCAL_QUADRANT_CACHE, f"quadrant_{name}.jpg")
        crop_img.save(tmp, format="JPEG", quality=92)
        paths[name] = tmp
    return paths


def _find_similar_references(sample_embedding, top_k=3):
    _ensure_reference_library()
    if not _reference_library:
        logger.error("Reference library is empty.")
        return []
    similarities = []
    for ref in _reference_library:
        sim = _cosine_similarity(sample_embedding, ref["embedding"])
        similarities.append({
            "image_name": ref["image_name"],
            "image_path": ref["image_path"],
            "similarity": round(sim, 4),
        })
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_k]


# ---------------------------------------------------------------------------
# Vision LLM — Standard 5-point inspection
# ---------------------------------------------------------------------------
def _vision_llm_compare(sample_image_path, reference_image_paths):
    w = _get_workspace_client()
    prompt_text = (
        "You are an expert manufacturing quality inspector specialising in engine fan parts.\n\n"
        "TASK: Compare the SAMPLE image (first image) against the REFERENCE images "
        "(subsequent images) from our approved parts library.\n\n"
        "Analyse the following aspects:\n"
        "1. **Physical Condition** - cracks, dents, corrosion, discolouration, surface damage\n"
        "2. **Structural Integrity** - blade count, blade shape, hub condition, mounting points\n"
        "3. **Dimensional Conformity** - size / proportions relative to references\n"
        "4. **Component Completeness** - all expected sub-components present (blades, hub, shroud)\n"
        "5. **Wear Assessment** - new, light wear, moderate wear, heavy wear, damaged\n\n"
        "IMPORTANT RULES:\n"
        "- Minor cosmetic differences (small scratches, light surface marks, slight wear, "
        "minor discolouration) are ACCEPTABLE and should result in PASS.\n"
        "- Only return FAIL if there are SIGNIFICANT, NOTICEABLE differences such as:\n"
        "  * Missing or broken components (blades, hub, shroud)\n"
        "  * Major structural damage (large cracks, severe deformation, bent blades)\n"
        "  * Completely different part type or wrong orientation\n"
        "  * Heavy corrosion or extensive surface damage\n\n"
        "Respond with JSON only (no markdown fences):\n"
        "{\n"
        "  \"decision\": \"PASS\" or \"FAIL\",\n"
        "  \"confidence\": 0.0 to 1.0,\n"
        "  \"condition\": \"new\" or \"light_wear\" or \"moderate_wear\" or \"heavy_wear\" or \"damaged\",\n"
        "  \"defects_found\": [\"list of specific defects or empty list\"],\n"
        "  \"similarity_to_reference\": \"high\" or \"medium\" or \"low\",\n"
        "  \"reasoning\": \"Detailed explanation\",\n"
        "  \"recommendations\": \"Maintenance or follow-up recommendations\"\n"
        "}"
    )
    content_parts = [{"type": "text", "text": prompt_text}]
    content_parts.append({"type": "text", "text": "SAMPLE IMAGE (to inspect):"})
    content_parts.append({"type": "image_url", "image_url": {
        "url": f"data:image/jpeg;base64,{_encode_image_base64(sample_image_path)}"
    }})
    for i, rp in enumerate(reference_image_paths):
        content_parts.append({"type": "text", "text": f"REFERENCE IMAGE {i + 1} (approved part):"})
        content_parts.append({"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{_encode_image_base64(rp)}"
        }})
    payload = {"messages": [{"role": "user", "content": content_parts}], "max_tokens": 1000, "temperature": 0.1}
    response = w.api_client.do("POST", path=f"/serving-endpoints/{VISION_MODEL}/invocations", body=payload)
    return _parse_llm_json(response["choices"][0]["message"]["content"])


# ---------------------------------------------------------------------------
# Vision LLM — Enhanced crack-focused inspection
# ---------------------------------------------------------------------------
def _vision_llm_crack_inspect(image_b64_list, labels, reference_b64):
    w = _get_workspace_client()
    prompt_text = (
        "You are an expert manufacturing quality inspector with specialised training "
        "in detecting cracks, fractures, and micro-defects in metal engine components.\n\n"
        "TASK: Carefully examine the SAMPLE images for ANY of the following defects:\n"
        "- Hairline cracks or fractures (even very thin lines)\n"
        "- Stress marks, fatigue lines, or surface discontinuities\n"
        "- Chipping, pitting, or micro-erosion\n"
        "- Discolouration that may indicate heat stress or material fatigue\n"
        "- Deformation, warping, or asymmetry relative to the REFERENCE image\n\n"
        "You will receive:\n"
        "1. A FULL VIEW of the sample part\n"
        "2. ZOOMED QUADRANT views (top-left, top-right, bottom-left, bottom-right) for detail\n"
        "3. A REFERENCE image of an approved part\n\n"
        "Examine EVERY quadrant carefully. Even a single hairline crack means FAIL.\n\n"
        "Respond with JSON only (no markdown fences):\n"
        "{\n"
        "  \"decision\": \"PASS\" or \"FAIL\",\n"
        "  \"confidence\": 0.0 to 1.0,\n"
        "  \"condition\": \"new\" or \"light_wear\" or \"moderate_wear\" or \"heavy_wear\" or \"damaged\",\n"
        "  \"defects_found\": [\"list EVERY defect with its location\"],\n"
        "  \"crack_detected\": true or false,\n"
        "  \"crack_details\": \"Describe any cracks: location, orientation, estimated length, severity\",\n"
        "  \"quadrant_analysis\": {\n"
        "    \"top_left\": \"findings\",\n"
        "    \"top_right\": \"findings\",\n"
        "    \"bottom_left\": \"findings\",\n"
        "    \"bottom_right\": \"findings\"\n"
        "  },\n"
        "  \"similarity_to_reference\": \"high\" or \"medium\" or \"low\",\n"
        "  \"reasoning\": \"Detailed explanation\",\n"
        "  \"recommendations\": \"Maintenance or follow-up recommendations\"\n"
        "}"
    )
    content = [{"type": "text", "text": prompt_text}]
    for b64, label in zip(image_b64_list, labels):
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": "REFERENCE IMAGE (approved part):"})   
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{reference_b64}"}})
    payload = {"messages": [{"role": "user", "content": content}], "max_tokens": 1500, "temperature": 0.1}
    response = w.api_client.do("POST", path=f"/serving-endpoints/{VISION_MODEL}/invocations", body=payload)
    return _parse_llm_json(response["choices"][0]["message"]["content"])


def _parse_llm_json(text):
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {"raw_response": text, "decision": "REVIEW_NEEDED"}


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "clip_loaded": _clip_model is not None,
        "references_loaded": _reference_library is not None,
        "reference_count": len(_reference_library) if _reference_library else 0,
        "reference_names": [r["image_name"] for r in (_reference_library or [])],
    }


@app.get("/api/debug")
async def debug():
    """Detailed diagnostic info for troubleshooting."""
    ref_files = []
    if os.path.isdir(LOCAL_REF_CACHE):
        ref_files = [
            {"name": f, "size": os.path.getsize(os.path.join(LOCAL_REF_CACHE, f))}
            for f in sorted(os.listdir(LOCAL_REF_CACHE))
        ]
    return {
        "clip_loaded": _clip_model is not None,
        "reference_library": _reference_library is not None,
        "reference_count": len(_reference_library) if _reference_library else 0,
        "reference_names": [r["image_name"] for r in (_reference_library or [])],
        "local_ref_files": ref_files,
        "startup_error": _startup_error,
        "volume_path": VOLUME_REFERENCE_PATH,
    }


@app.get("/api/references")
async def get_references():
    try:
        _ensure_reference_library()
    except Exception as e:
        logger.error(f"Failed to load references: {e}")
        return {"references": [], "count": 0, "error": str(e)}
    refs = []
    for ref in (_reference_library or []):
        refs.append({"image_name": ref["image_name"], "image_url": f"/api/image?path={ref['image_path']}"})
    return {"references": refs, "count": len(refs)}


@app.get("/api/image")
async def get_image(path: str):
    allowed = [LOCAL_REF_CACHE, LOCAL_INSPECT_CACHE, LOCAL_QUADRANT_CACHE]
    if not any(path.startswith(p) for p in allowed):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    ext = path.lower().rsplit(".", 1)[-1]
    media = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
    return FileResponse(path, media_type=media)


@app.post("/api/inspect")
async def inspect_part(
    file: UploadFile = File(...),
    mode: str = Form("standard"),
):
    try:
        filename = file.filename or "uploaded_part.jpg"
        os.makedirs(LOCAL_INSPECT_CACHE, exist_ok=True)
        local_path = os.path.join(LOCAL_INSPECT_CACHE, filename)
        file_content = await file.read()
        with open(local_path, "wb") as f:
            f.write(file_content)
        logger.info(f"Saved: {local_path} ({len(file_content):,} bytes)")

        # Persist to Volume (non-fatal)
        try:
            _upload_to_volume(local_path, f"{VOLUME_INSPECT_PATH}/{filename}")
        except Exception as e:
            logger.warning(f"Volume upload failed (non-fatal): {e}")

        # Stage 1: CLIP similarity
        logger.info("Stage 1: CLIP embedding...")
        sample_emb = _get_image_embedding(local_path)
        similar_refs = _find_similar_references(sample_emb, top_k=3)
        best_ref = similar_refs[0] if similar_refs else None
        best_sim = best_ref["similarity"] if best_ref else 0
        logger.info(f"Stage 1 done: best={best_ref['image_name'] if best_ref else 'N/A'} sim={best_sim}")

        # Stage 2: Vision LLM
        if mode == "enhanced":
            logger.info("Stage 2: Enhanced crack inspection...")
            llm_result = _run_enhanced_inspection(local_path, best_ref)
        else:
            logger.info("Stage 2: Standard 5-point inspection...")
            ref_paths = [best_ref["image_path"]] if best_ref else []
            llm_result = _vision_llm_compare(local_path, ref_paths)
        logger.info(f"Stage 2 done: decision={llm_result.get('decision', 'N/A')}")

        return JSONResponse(content=_build_response(
            filename, local_path, similar_refs, best_ref, best_sim, llm_result, mode
        ))
    except Exception as e:
        logger.error(f"Inspection failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


def _run_enhanced_inspection(sample_path, best_ref):
    full_b64 = _encode_image_base64(sample_path, max_size=2000)
    quadrant_paths = _crop_quadrants(sample_path, overlap=0.1)
    ref_path = best_ref["image_path"] if best_ref else None
    ref_b64 = _encode_image_base64(ref_path, max_size=2000) if ref_path else ""
    image_b64s = [full_b64]
    labels = ["FULL VIEW \u2014 Sample part under inspection:"]
    for qname in ["top_left", "top_right", "bottom_left", "bottom_right"]:
        image_b64s.append(_encode_image_base64(quadrant_paths[qname], max_size=1500))
        labels.append(f"ZOOMED QUADRANT \u2014 {qname.replace('_', ' ').upper()}:")
    total_kb = sum(len(b) for b in image_b64s) * 3 / 4 / 1024
    logger.info(f"Enhanced: {len(image_b64s)} images (~{total_kb:.0f} KB) + 1 reference")
    return _vision_llm_crack_inspect(image_b64s, labels, ref_b64)


def _build_response(filename, local_path, similar_refs, best_ref, best_sim, llm_result, mode):
    return {
        "mode": mode,
        "uploaded_image": {
            "filename": filename,
            "volume_path": f"{VOLUME_INSPECT_PATH}/{filename}",
            "image_url": f"/api/image?path={local_path}",
        },
        "stage1": {
            "description": "CLIP Embedding Similarity Search",
            "best_match": {
                "image_name": best_ref["image_name"] if best_ref else "N/A",
                "image_path": best_ref["image_path"] if best_ref else "N/A",
                "similarity_score": best_sim,
                "image_url": f"/api/image?path={best_ref['image_path']}" if best_ref else "",
            },
            "all_matches": [
                {"rank": i + 1, "image_name": r["image_name"], "similarity_score": r["similarity"],
                 "image_url": f"/api/image?path={r['image_path']}"}
                for i, r in enumerate(similar_refs)
            ],
            "threshold": SIMILARITY_THRESHOLD,
            "above_threshold": best_sim >= SIMILARITY_THRESHOLD,
        },
        "stage2": _build_stage2(llm_result, mode),
        "final_decision": llm_result.get("decision", "REVIEW_NEEDED"),
    }


def _build_stage2(llm_result, mode):
    s2 = {
        "description": "Enhanced Crack-Focused Inspection" if mode == "enhanced" else "Vision LLM 5-Point Structural Inspection",
        "decision": llm_result.get("decision", "REVIEW_NEEDED"),
        "confidence": llm_result.get("confidence", 0),
        "condition": llm_result.get("condition", "unknown"),
        "defects_found": llm_result.get("defects_found", []),
        "similarity_to_reference": llm_result.get("similarity_to_reference", "unknown"),
        "reasoning": llm_result.get("reasoning", ""),
        "recommendations": llm_result.get("recommendations", ""),
    }
    if mode == "enhanced":
        s2["crack_detected"] = llm_result.get("crack_detected", False)
        s2["crack_details"] = llm_result.get("crack_details", "")
        s2["quadrant_analysis"] = llm_result.get("quadrant_analysis", {
            "top_left": "N/A", "top_right": "N/A", "bottom_left": "N/A", "bottom_right": "N/A"
        })
    return s2


app.mount("/static", StaticFiles(directory="static"), name="static")
