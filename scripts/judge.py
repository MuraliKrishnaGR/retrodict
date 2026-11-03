"""Blind-judge one run against its answer key (one Muse Spark Open call, about 1 US cent, cached).

  python scripts/judge.py out/salad_photo eval/keys/salad_01.json
  python scripts/judge.py out/baselines/salad_photo eval/keys/salad_01.json

Saves judge.json in the run folder; scripts/score.py then uses it instead of the word patterns."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.models import Models        # noqa: E402
from app.cost import BudgetExceeded          # noqa: E402
from app.judge import judge                  # noqa: E402


async def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: python scripts/judge.py out/<run> eval/keys/<video>.json")
    run, key_path = Path(sys.argv[1]), Path(sys.argv[2])
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    models = Models()
    try:
        j = await judge(models, key, result)
    except BudgetExceeded as e:
        raise SystemExit(f"Stopped by spend guard: {e}")
    finally:
        await models.aclose()
    (run / "judge.json").write_text(json.dumps(j, indent=2), encoding="utf-8")
    texts = {e["id"]: e["description"] for e in result.get("events", [])}
    for eid, ks in j["covers"].items():
        flag = "  WRONG" if eid in j["wrong"] else ""
        print(f"  {eid}: {', '.join(ks) or '(nothing)'}{flag}\n      {texts.get(eid, '')}\n      judge: {j['notes'].get(eid, '')}")
    print(f"\nSaved {run / 'judge.json'}. Now: python scripts/score.py {run} {key_path}")


if __name__ == "__main__":
    asyncio.run(main())
