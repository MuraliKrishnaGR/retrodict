"""SAM 3.1 open weights, run locally with Hugging Face transformers (CPU is fine).
The image is encoded once; each extra concept phrase only runs the light decoder.
Results are cached on disk, so the same photo and phrase are never computed twice."""
import io

import numpy as np
from PIL import Image

from app import config
from app.cache import DiskCache, make_key


def clean_box(box, mask) -> list:
    """A box that always lies inside the image with left <= right and top <= bottom.
    SAM's own box can poke slightly outside the photo (e.g. a jar touching the top edge gives a
    negative top), so the tight box around the mask is used, falling back to the clamped SAM box."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs):
        return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    l, t, r, b = (float(v) for v in box)
    l, r = sorted((min(max(l, 0.0), w), min(max(r, 0.0), w)))
    t, b = sorted((min(max(t, 0.0), h), min(max(b, 0.0), h)))
    return [round(l, 1), round(t, 1), round(r, 1), round(b, 1)]


class LocalSAM:
    def __init__(self, cache: DiskCache | None = None, model=None, processor=None,
                 model_id: str = config.SAM_MODEL_ID):
        self.cache = cache or DiskCache(config.CACHE_DIR)
        self.model_id = model_id
        self._model, self._processor = model, processor
        self._vision_key, self._vision = None, None
        self.cache_hits = 0
        self.computed = 0

    def _load(self):
        if self._model is None:
            import torch
            from transformers import Sam3Model, Sam3Processor
            print(f"Loading SAM 3.1 ({self.model_id}). First time downloads ~1.7 GB...", flush=True)
            torch.set_grad_enabled(False)
            self._processor = Sam3Processor.from_pretrained(self.model_id)
            self._model = Sam3Model.from_pretrained(self.model_id, dtype=torch.float32).eval()
        return self._model, self._processor

    def _vision_features(self, jpeg: bytes, img: Image.Image):
        key = make_key("vision", jpeg)
        if key != self._vision_key:
            model, processor = self._load()
            inputs = processor(images=img, return_tensors="pt")
            self._vision = model.get_vision_features(pixel_values=inputs["pixel_values"])
            self._vision_key = key
        return self._vision

    def segment(self, jpeg: bytes, concept: str) -> list:
        """One concept phrase -> [{'object_id','bbox':[l,t,r,b],'score','mask': bool (H, W)}]."""
        key = make_key("sam-local", self.model_id, concept, jpeg,
                       config.SAM_SCORE_THRESHOLD, config.SAM_MASK_THRESHOLD)
        meta = self.cache.get_json("sam", key)
        if meta is not None:
            self.cache_hits += 1
            if not meta["items"]:
                return []
            arr = np.load(self.cache.path_for("sam", key, "npz"))["masks"]
            return [dict(it, mask=arr[i], bbox=clean_box(it["bbox"], arr[i]))
                    for i, it in enumerate(meta["items"])]

        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        model, processor = self._load()
        vision = self._vision_features(jpeg, img)
        text_inputs = processor(text=concept, return_tensors="pt")
        outputs = model(vision_embeds=vision, input_ids=text_inputs["input_ids"],
                        attention_mask=text_inputs["attention_mask"])
        res = processor.post_process_instance_segmentation(
            outputs, threshold=config.SAM_SCORE_THRESHOLD, mask_threshold=config.SAM_MASK_THRESHOLD,
            target_sizes=[(img.height, img.width)])[0]
        self.computed += 1

        masks = np.asarray(res["masks"]).astype(bool)
        boxes = np.asarray(res["boxes"], dtype=float)
        scores = np.asarray(res["scores"], dtype=float)
        order = np.argsort(-scores)[:config.SAM_MAX_INSTANCES]
        items = [{"object_id": str(i), "bbox": clean_box(boxes[j], masks[j]),
                  "score": round(float(scores[j]), 4), "mask": masks[j]}
                 for i, j in enumerate(order)]

        self.cache.put_json("sam", key, {"concept": concept, "items": [
            {k: v for k, v in it.items() if k != "mask"} for it in items]})
        if items:
            np.savez_compressed(self.cache.path_for("sam", key, "npz"),
                                masks=np.stack([it["mask"] for it in items]))
        return items
