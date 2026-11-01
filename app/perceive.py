"""Pipeline step 1: photo -> object categories (Muse Spark Open) -> one clean mask per object (SAM 3.1).

SAM 3.1 segments every instance of a *concept*, so "brown jar" can also match the other jars.
To get exactly one mask per physical object we:
  1. ask for plain, generic nouns plus a count ("jar" x3, "packet" x1),
  2. retry a phrase that finds nothing with its last word ("rusk bag" -> "bag"),
  3. assign every mask to at most one object: highest score first (fallback matches last), skip
     masks that overlap an accepted one (IoU >= 0.5), and stop each concept at its count,
  4. then keep any extra mask SAM is very sure about (score >= 0.8, no overlap): SAM's detections
     are evidence, the language model's count is only a hint (e.g. it said "pot" but SAM calls it a jar).
"""
import json
from pathlib import Path

import numpy as np

from app.imaging import overlay, prepare_image, save_mask_png
from app.schemas import ObjectList, SceneObject

LIST_PROMPT = """You are preparing a photo for an investigator who will reason about what happened BEFORE it was taken.
List the distinct kinds of physical objects in this photo whose state could reveal recent activity
(examples: mug, plate, chair, drawer, towel, laptop, packet, jar, glass, kettle, spoon, door).
Rules:
- 'name' is the plainest generic noun a segmentation model can find: ONE or TWO words, singular,
  no colours or brands (good: "jar", "packet", "mug", "plate"; bad: "brown jar", "rusk bag").
- If several objects share a name, list the name ONCE and give their 'count'.
- 'desc' tells them apart in a few words (colour, material, lid, contents).
- 'why' is one short sentence on what their state might reveal.
- At most {n} entries. Do not list walls, floor, tiles, ceiling or people.
Return ONLY JSON: {{"objects": [{{"name": "...", "count": 1, "desc": "...", "why": "..."}}]}}"""

IOU_DUPLICATE = 0.5
FALLBACK_RANK = 0.5   # fallback matches rank below every direct match: they only take leftovers
EXTRA_MIN_SCORE = 0.8 # an uncounted mask is kept only if SAM is this sure


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter) / float(np.logical_or(a, b).sum())


def merge_concepts(entries) -> list:
    """Normalise names, merge duplicates (summing counts). Returns [(name, count, desc)]."""
    merged = {}
    for o in entries:
        name = " ".join(o.name.strip().lower().split())
        if not name:
            continue
        if name in merged:
            merged[name][0] += o.count
            if o.desc:
                merged[name][1].append(o.desc)
        else:
            merged[name] = [o.count, [o.desc] if o.desc else []]
    return [(n, c, "; ".join(d)) for n, (c, d) in merged.items()]


def assign_masks(candidates: list, counts: dict, min_area: int) -> list:
    """candidates: [{'concept','score','mask','bbox'}]. Greedy, highest score first:
    a mask is kept only if it is big enough, its concept still has room, and it does not
    duplicate a mask already kept."""
    ranked = sorted(candidates, key=lambda c: -c["score"] * (FALLBACK_RANK if c.get("fallback") else 1.0))
    kept, used = [], {}

    def free(c):
        return c["mask"].sum() >= min_area and not any(
            mask_iou(c["mask"], k["mask"]) >= IOU_DUPLICATE for k in kept)

    for c in ranked:                                    # pass 1: within each concept's count
        if used.get(c["concept"], 0) < counts.get(c["concept"], 1) and free(c):
            kept.append(c)
            used[c["concept"]] = used.get(c["concept"], 0) + 1
    for c in ranked:                                    # pass 2: confident objects nobody counted
        if not c.get("fallback") and c["score"] >= EXTRA_MIN_SCORE and free(c):
            kept.append(dict(c, extra=True))
    # stable, readable order: top-to-bottom, then left-to-right
    return sorted(kept, key=lambda k: (round(k["bbox"][1] / 40), k["bbox"][0]))


async def perceive(models, photo_path, out_dir, max_objects: int = 6, min_area: int = 150):
    out_dir = Path(out_dir)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    img, jpeg = prepare_image(photo_path)
    (out_dir / "input.jpg").write_bytes(jpeg)

    listing = await models.spark_json(LIST_PROMPT.format(n=max_objects), ObjectList, jpeg=jpeg, tag="list_objects")
    concepts = merge_concepts(listing.objects)[:max_objects]
    counts = {name: count for name, count, _ in concepts}

    candidates, searched = [], []
    for name, _, _ in concepts:
        items = await models.sam_segment(jpeg, img.width, img.height, name)
        used_phrase = name
        if not items and " " in name:                      # too specific: try the last word
            used_phrase = name.split()[-1]
            items = await models.sam_segment(jpeg, img.width, img.height, used_phrase)
        searched.append({"name": name, "phrase": used_phrase, "found": len(items)})
        for it in items:
            candidates.append({"concept": name, "phrase": used_phrase, "score": float(it.get("score", 0.0)),
                               "fallback": used_phrase != name, "mask": it["mask"], "bbox": it["bbox"]})

    kept = assign_masks(candidates, counts, min_area)

    totals = {}
    for k in kept:
        totals[k["concept"]] = totals.get(k["concept"], 0) + 1
    found, drawn, per_concept = [], [], {}
    for k in kept:
        per_concept[k["concept"]] = per_concept.get(k["concept"], 0) + 1
        label = k["concept"] if totals[k["concept"]] == 1 else f"{k['concept']} {per_concept[k['concept']]}"
        oid = f"o{len(found) + 1}"
        mask_path = out_dir / "masks" / f"{oid}.png"
        save_mask_png(k["mask"], mask_path)
        found.append(SceneObject(id=oid, label=label, bbox=k["bbox"],
                                 mask_png=mask_path.relative_to(out_dir).as_posix(),
                                 area=int(k["mask"].sum()), score=round(k["score"], 4), concept=k["phrase"]))
        drawn.append({"id": oid, "label": label, "mask": k["mask"], "bbox": k["bbox"]})

    overlay(img, drawn).save(out_dir / "overlay.png")
    missing = [n for n, _, _ in concepts if n not in totals]
    result = {"photo": str(photo_path), "size": [img.width, img.height], "not_found": missing,
              "concepts": [{"name": n, "count": c, "desc": d} for n, c, d in concepts],
              "search": searched,
              "objects": [f.model_dump() for f in found]}
    (out_dir / "objects.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
