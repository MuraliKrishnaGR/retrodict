# Retrodict

Retrodict reasons backward from **one photo** to what most likely happened before it was taken:
a timeline of events with evidence, plus rendered earlier versions of the scene.

Built during the Meta Global AI Developer Hackathon (indie track, Push the Frontier) entirely on
**Meta's open-weight models**:
- **Muse Spark Open** (reasoning and vision), served through Baseten partner inference
- **SAM 3.1** (segmentation), run locally on CPU with Hugging Face `transformers`

## How it works
1. **Perceive**: Muse Spark Open names the kinds of objects; SAM 3.1 gives one pixel mask per object
   (concept matches are de-duplicated, and SAM's confident detections are kept even if uncounted).
2. **Read**: Muse Spark Open reads each object's current state from its own close-up crop
   and names the physical traces it can see (steam, wet ring, torn wrapper...).
3. **Investigate**: an agent built on Muse Spark Open works like a detective with tools. Each turn it
   chooses one move: **zoom** (a sharp close-up from the full-resolution photo), **search** (SAM 3.1 looks
   for something not yet found, such as a spoon or crumbs, which then becomes citable evidence), or
   **clock** (the timing rule of a physical trace). Then it gives a backward timeline in which every event
   cites evidence visible now and says how a rewind should show the scene before it.
   **Trace clocks** (physical rules such as "visible steam lasts minutes") fix or narrow each event's time
   range, so times rest on physics, not on the model's guess. If the agent fails, a single reasoning call
   takes over.
4. **Verify**: an independent check sees only the plain claims, not the story, and confirms or rejects
   each piece of evidence. Events left with no support are rejected and shown as such.

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
python scripts/run.py path\to\photo.jpg          # full pipeline: timeline with evidence
```
Results go to `out/<photo name>/` (`result.json`, `sheet.jpg`, `overlay.png`, `masks/`).
The first SAM run downloads the SAM 3.1 weights (~1.7 GB). Re-running the same photo is instant.

## Cost safety
Every paid LLM call passes a spend guard (`MAX_SPEND_USD`, `MAX_RUN_USD` in `.env`) and is logged in
`.cache/calls.log`. SAM 3.1 runs locally and costs nothing. Prices are estimates; the host's billing page is
the real source of truth.
