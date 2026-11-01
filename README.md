# Retrodict

Retrodict reasons backward from **one photo** to what most likely happened before it was taken:
a timeline of events with evidence, plus rendered earlier versions of the scene.

Built during the Meta Global AI Developer Hackathon (indie track, Push the Frontier) entirely on
**Meta's open-weight models**:
- **Muse Spark Open** (reasoning and vision), served through Baseten partner inference
- **SAM 3.1** (segmentation), run locally on CPU with Hugging Face `transformers`

## Status
- Step 1 done: photo -> object names (Muse Spark Open) -> pixel masks (SAM 3.1).
  Every model result is cached on disk, and every paid call passes a spend guard.

## Run
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python scripts/set_key.py                          # paste your Baseten API key (hidden)
python -m pytest -q                                # offline tests, no key needed
python scripts/check_setup.py path\to\photo.jpg    # step-by-step model check
python scripts/day1_run.py path\to\photo.jpg
```
Results go to `out/<photo name>/` (`overlay.png`, `objects.json`, `masks/`).
The first SAM run downloads the SAM 3.1 weights (~1.7 GB). Re-running the same photo is instant.

## Cost safety
Every paid LLM call passes a spend guard (`MAX_SPEND_USD`, `MAX_RUN_USD` in `.env`) and is logged in
`.cache/calls.log`. SAM 3.1 runs locally and costs nothing. Prices are estimates; the host's billing page is
the real source of truth.
