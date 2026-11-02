# Retrodict

Retrodict reasons backward from **one photo** to what most likely happened before it was taken: a timeline
of events, each with a time **range**, the visible evidence behind it and an independent check of that
evidence, plus a short **rewind clip** made from the real photo.

Built during the Meta Global AI Developer Hackathon (indie track, Push the Frontier) entirely on
**Meta's open-weight models**:
- **Muse Spark Open** (reasoning and vision), served through Baseten partner inference
- **SAM 3.1** (segmentation), run locally on CPU with Hugging Face `transformers`

It is a best guess from the clues in the photo, not proof: every result says so, and every time is a range
("up to 30 minutes before"), never an exact moment.

## How it works
1. **Perceive.** Muse Spark Open names the kinds of objects; SAM 3.1 gives one pixel mask per object.
2. **Read.** Each object is read on its own, from a sharp close-up cut from the original photo and shown
   in six views: as photographed, shadows lifted (faint steam, film), glare dimmed (what reflections hide),
   and three views of the object alone on black: a whiteness map (a white spoon in brown sugar) and the two
   light edits, out of a bank of eight, that measurably reveal the most detail inside that object. The
   model names the physical traces it sees from a fixed list, each defined by how it looks and what it is
   confused with ("bubbles are not a skin").
3. **Investigate.** An agent on Muse Spark Open works like a detective with tools. Each turn it picks one
   move: **zoom** on an object, **search** with SAM 3.1 (on the whole photo, or as a *microscope* inside one
   object's full-resolution close-up, where crumbs and drips are big enough to segment), or **clock** (the
   timing rule of a trace it can see). It connects objects (a pan with milk on its wall held the milk now in
   the glass) and reasons about unclear shapes ("something white in the sugar is most likely a spoon").
4. **Verify.** An independent checker sees sharp close-ups and plain claims, never the story. Each claim
   is confirmed, partly right, can't tell, or contradicted. Contradicted events are rejected or marked, and
   the summary people read keeps only what held up.
5. **Rewind.** The real photo, walked back through the events, each step the opposite of its event: an
   object that arrived (or was moved here) is erased, a drink poured into a cup is erased so the cup stands
   empty, crumbs left by eating are erased, steam is drawn and labelled as an illustration, and anything
   that cannot be shown honestly (a wrapper resealed) is marked, not painted. Erasing uses **LaMa**, an
   open-source inpainting model (Apache-2.0), as a helper on the CPU: SAM's masks decide what goes, the
   timeline decides when, and other objects are protected. If it is unavailable, a simple fill is used.

## Honest timing
- **Trace clocks:** physical rules ("visible steam lasts minutes", "a skin forms on milk as it cools")
  date events, not the model's guess.
- **Clock gate:** a clock dates an event only if the independent checker confirms the trace is really
  there. (A model once read bubbles as a skin and dated a 3-minute-old drink to "10 to 90 minutes".)
- **No clue, no minimum:** without a confirmed trace an event is "up to X minutes before"; a missing trace
  proves nothing (no visible steam does not mean a drink is old).
- **Chains:** steps on the same object share their times (milk poured, then drunk from the same glass).

## Web page
```powershell
python scripts/serve.py          # http://127.0.0.1:8000
python scripts/serve.py --lan    # also from a phone on the same Wi-Fi ("Take a photo" opens the camera)
```
Upload or take a photo and watch the investigation live: each step, each agent move and why. The time
ruler shows every event as a range before the photo; click one for its evidence, the checker's verdicts,
the physics behind its time and the six views the model saw. The same photo is never paid for twice.
Uploaded photos stay on your computer.

## Command line
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python scripts/set_key.py                          # paste your Baseten API key (hidden)
python -m pytest -q                                # offline tests, no key needed
python scripts/run.py path\to\photo.jpg            # full run: timeline, evidence, rewind clip
python scripts/rewind.py out\<photo name>          # remake the rewind clip (free)
python scripts/microscope.py out\<photo name> o1 spoon   # SAM search inside one object (free)
```
Results go to `out/<photo name>/`: `result.json`, `rewind.mp4`, `detail_<object>.jpg` (exactly what the
model was shown for each object), `evidence.jpg` (what the checker saw), `overlay.png`, `masks/`.
The first SAM run downloads the SAM 3.1 weights (~1.7 GB). Photos from a chat app are low in detail;
Retrodict says so, and the original from the phone's camera works better.

## Cost safety
Every paid call passes a spend guard (`MAX_SPEND_USD`, `MAX_RUN_USD` in `.env`), is cached on disk and is
logged in `.cache/calls.log`. A photo costs about 10 to 20 US cents. SAM 3.1 runs locally and costs
nothing. Prices are estimates; the host's billing page is the source of truth.
