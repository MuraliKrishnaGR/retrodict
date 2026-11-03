"""Physical order rules, in code: some steps can only happen in one order, whatever a model says.

  the same fruit is peeled before it is cut       (peel banana -> slice banana)
  food is cut before it goes into the bowl         (chop apple -> fruit into the bowl)
  everything is in the bowl before it is mixed     (into the bowl -> mix)
  a bottle is opened before anything is poured     (cap off -> water poured)

Events that break a rule are moved, keeping the model's order everywhere else, and each move is noted.
The same words also link events about the same food, so a clue that dates one step can bound the steps
before and after it on that food (see reason.propagate_order)."""
import re

FOODS = {"banana": "banana", "apple": "apple", "orange": "orange", "mandarin": "orange", "tangerine": "orange",
         "citrus": "orange", "clementine": "orange", "lemon": "lemon", "lime": "lime", "cucumber": "cucumber",
         "tomato": "tomato", "onion": "onion", "carrot": "carrot", "potato": "potato", "mango": "mango",
         "pear": "pear", "grape": "grape", "melon": "melon", "watermelon": "melon", "pineapple": "pineapple",
         "papaya": "papaya", "guava": "guava", "pomegranate": "pomegranate", "kiwi": "kiwi", "bread": "bread",
         "egg": "egg", "cheese": "cheese"}

PEEL = re.compile(r"\b(peel(ed|ing|s)?|skinn?(ed|ing)|unpeel\w*)\b")
CUT = re.compile(r"\b(cut|cuts|cutting|slic\w+|chopp?\w*|dic(e|ed|ing)|halv\w+|cubed?|quarter(ed)?|segment(ed)?)\b")
INTO = re.compile(r"\b(put|puts|placed|added|gathered|transferred|moved|dropped|tipped|tossed|combined|collected)\b")
MIX = re.compile(r"\b(mix\w*|stirr?\w*|toss\w*|combin\w*)\b")
OPEN = re.compile(r"\b(open\w*|unscrew\w*|uncapp?\w*)\b|\b(cap|lid)\b.*\b(off|removed|aside|taken)\b")
CONTAINER = re.compile(r"\b(bottle|can|jar|carton|cap|lid|flask|jug|tumbler)s?\b")
POUR = re.compile(r"\b(pour\w*|filled|refilled)\b")


def foods(text: str) -> set:
    t = text.lower()
    return {v for k, v in FOODS.items() if re.search(rf"\b{k}(e?s)?\b", t)}


def stage(text: str) -> dict:
    """{'food': 0 peel | 1 cut | 2 into bowl | 3 mix, 'drink': 0 open | 1 pour} for the steps it mentions;
    an event that mentions several food steps counts as its latest one."""
    t = text.lower()
    s = {}
    if MIX.search(t):
        s["food"] = 3
    elif "bowl" in t and INTO.search(t):
        s["food"] = 2
    elif CUT.search(t):
        s["food"] = 1
    elif PEEL.search(t):
        s["food"] = 0
    if POUR.search(t):
        s["drink"] = 1
    elif OPEN.search(t) and CONTAINER.search(t):
        s["drink"] = 0
    return s


def must_precede(a: dict, b: dict) -> str:
    """The rule that puts event a before event b, or ''."""
    sa, sb = stage(a["description"]), stage(b["description"])
    fa, fb = foods(a["description"]), foods(b["description"])
    if "food" in sa and "food" in sb and sa["food"] < sb["food"]:
        if sa["food"] == 0 and sb["food"] == 1:
            if fa & fb:
                return f"the {sorted(fa & fb)[0]} is peeled before it is cut"
        elif sb["food"] == 2:
            if not fa or not fb or fa & fb:
                return "food is cut before it goes into the bowl"
        elif sb["food"] == 3:
            return "everything goes into the bowl before it is mixed"
    if sa.get("drink") == 0 and sb.get("drink") == 1:
        return "a bottle is opened before anything is poured"
    return ""


def order_by_rules(events: list) -> list:
    """Events earliest first, moved only where a physical rule demands it (stable otherwise). Events that
    were moved get 'order_rule'."""
    n = len(events)
    before = {j: [(i, r) for i in range(n) if i != j and (r := must_precede(events[i], events[j]))]
              for j in range(n)}
    placed, out = set(), []
    while len(out) < n:
        ready = [j for j in range(n) if j not in placed and all(i in placed for i, _ in before[j])]
        j = ready[0] if ready else min(k for k in range(n) if k not in placed)   # a cycle: keep model order
        placed.add(j)
        out.append(j)
    result = []
    for j in out:
        late = [r for i, r in before[j] if i > j]     # a step listed after it had to come first
        result.append(dict(events[j], order_rule=late[0]) if late else events[j])
    return result


def links(e: dict) -> set:
    """What ties events together for timing: the objects they cite and the foods they name."""
    return {ev["object_id"] for ev in e.get("evidence", [])} | {f"food:{f}" for f in foods(e["description"])}
