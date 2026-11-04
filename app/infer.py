"""Name the action, not the leftover: what a used utensil standing in food says happened.

The investigator sometimes reports a leftover as the event ("the spoon was left in the bowl after use").
A spoon standing in a bowl of mixed food was used to mix it; the action is what happened before the photo,
the spoon is only its trace. The rule rewrites such an event into the action, keeps the original wording
in 'read_as', and keeps the same evidence (the checker already confirmed it)."""
import re

UTENSIL = r"(spoon|fork|ladle|whisk|spatula|chopsticks?|stirrer)"
CONTAINER = r"(bowl|pot|pan|mug|cup|plate|jar|glass|tumbler)"
LEFT_IN = re.compile(rf"\b(?:the\s+)?(?:\w+\s+)?{UTENSIL}\b.*\b(?:left|standing|resting|lying|remain\w*|sitting|stuck)\b"
                     rf".*\b(?:in|inside|into)\s+(?:the\s+)?(?:\w+\s+)?{CONTAINER}\b", re.I)
ACTION = re.compile(r"\b(mix\w*|stirr?\w*|toss\w*|serv\w*|scoop\w*|eat\w*|ate)\b", re.I)


def name_actions(events: list) -> list:
    out = []
    for e in events:
        d = e["description"]
        m = LEFT_IN.search(d)
        if m and not ACTION.search(d):
            utensil, container = m.group(1).lower(), m.group(2).lower()
            e = dict(e, read_as=d, description=f"the food in the {container} was most likely mixed with the "
                                               f"{utensil}, which was left standing in it")
        out.append(e)
    return out


BOTTLE = re.compile(r"\b(bottle|jug|flask|carton|kettle)\b", re.I)
OPENED = re.compile(r"\b(open\w*|unscrew\w*|uncapp?\w*)\b|\b(cap|lid|top)\b.*\b(off|removed|taken|aside|detached)\b",
                    re.I)
PART_FULL = re.compile(r"\b(below the (open )?rim|well below|half|third|quarter|partly|partially|part[- ]full|"
                       r"not full|low level|level dropped)\b", re.I)
VESSEL = re.compile(r"\b(cup|glass|tumbler|mug|beaker)\b", re.I)
POURED = re.compile(r"\b(pour\w*|filled|refilled|drank|drunk)\b", re.I)


CAP = re.compile(r"\b(cap|lid|stopper)\b", re.I)


def _centre(o):
    l, t, r, b = o["bbox"]
    return ((l + r) / 2, (t + b) / 2)


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def infer_pour(events: list, states: dict, labels: dict, objects: list = ()) -> list:
    """A liquid moves from one container to another. If the story says a bottle was opened and a cup or glass
    stands in the scene, and either the bottle is now only partly full or its cap lies by the cup (it was
    opened over the cup, far from where the bottle now lies), the drink was most likely poured into the cup.
    The new event cites what is visible (the level, the cup, the cap) so the checker can test each fact."""
    if any(POURED.search(e["description"]) for e in events):
        return events
    opened = next((e for e in events if OPENED.search(e["description"]) and BOTTLE.search(e["description"])), None)
    if not opened:
        return events
    bottle = next((ev["object_id"] for ev in opened["evidence"]
                   if BOTTLE.search(f"{labels.get(ev['object_id'], '')} {states.get(ev['object_id'], {}).get('what', '')}")),
                  None)
    if not bottle:
        return events
    level = next((o for o in states.get(bottle, {}).get("observations", []) if PART_FULL.search(o)), None)
    cups = [oid for oid, lab in labels.items() if oid != bottle and VESSEL.search(lab) and "holder" not in lab.lower()]
    if not cups:
        return events
    cup = cups[0]
    pos = {o["id"]: _centre(o) for o in objects if o.get("bbox")}
    cap = next((oid for oid, lab in labels.items() if CAP.search(lab) and oid in pos and cup in pos and bottle in pos
                and _dist(pos[oid], pos[cup]) < _dist(pos[oid], pos[bottle]) / 2), None)
    if not level and not cap:
        return events
    what = f"{labels.get(bottle, '')} {states.get(bottle, {}).get('what', '')}".lower()
    drink = "water" if ("water" in what or "clear liquid" in what) else "the drink"
    evidence = ([{"object_id": bottle, "observation": level}] if level else []) + \
        [{"object_id": cup, "observation": "a cup is in the scene"}] + \
        ([{"object_id": cap, "observation": "the cap lies beside the cup, far from the bottle"}] if cap else [])
    event = {"id": f"e{len(events) + 1}", "description": f"{drink} was most likely poured from the "
             f"{labels.get(bottle, 'bottle')} into the {labels.get(cup, 'cup')}",
             "minutes_before_min": 0.0, "minutes_before_max": opened["minutes_before_max"],
             "model_range": [0.0, opened["minutes_before_max"]], "clock": "no_clock", "clock_rule": None,
             "trace": None, "confidence": 0.5, "alternatives": ["someone drank straight from the bottle"],
             "undo": "mark", "undo_object": cup,
             "source": "inferred: an opened bottle, a cup" + (", the cap left by the cup" if cap else ""),
             "evidence": evidence}
    return events + [event]


FOOD = re.compile(r"\b(biscuits?|cookies?|crackers?|wafers?|chips|crisps|chocolates?|snacks?|cakes?|rusks?)\b", re.I)
PACKET = re.compile(r"\b(packet|package|wrapper|pack|bag|box|foil)\b", re.I)
TORN = re.compile(r"\b(open\w*|torn|tore|ripp\w*|cut open|unwrapp\w*)\b", re.I)
DRINK = re.compile(r"\b(tea|coffee|milk|chai|cocoa|drink)\b", re.I)
EATEN = re.compile(r"\b(eat\w*|ate|eaten|dipp?\w*|dunk\w*|bit\w*|consum\w*)\b", re.I)


DIPPED = re.compile(r"\b(dipp?\w*|dunk\w*|soak\w*)\b", re.I)
ATE = re.compile(r"\b(eat\w*|ate|eaten|bit\w*|consum\w*)\b", re.I)


def infer_snack(events: list, states: dict, labels: dict) -> list:
    """A packet of food is opened to eat from it. If the story says a food packet was torn open, some was most
    likely eaten; with a drink standing by (tea, coffee, milk), a biscuit is most likely dipped in it first. If
    the story already has the eating but not the dipping, only the dipping is added. The new event cites the
    open packet and the drink so the checker tests both."""
    if any(DIPPED.search(e["description"]) for e in events):
        return events
    text = lambda oid: f"{labels.get(oid, '')} {states.get(oid, {}).get('what', '')}"
    eaten = next((e for e in events if ATE.search(e["description"]) and FOOD.search(e["description"])), None)
    opened = next((e for e in events if PACKET.search(e["description"]) and TORN.search(e["description"])), None)
    source = opened or eaten
    if not source:
        return events
    packet = next((ev["object_id"] for e in (opened, eaten) if e for ev in e["evidence"]
                   if PACKET.search(text(ev["object_id"])) or FOOD.search(text(ev["object_id"]))), None)
    if not packet:
        return events
    m = FOOD.search(f"{text(packet)} {source['description']}")
    food = (m.group(1).lower().rstrip("s") if m else "piece")
    cup = next((oid for oid in labels if oid != packet and VESSEL.search(labels[oid] + " " + text(oid))
                and DRINK.search(states.get(oid, {}).get("what", ""))), None)
    if eaten and not cup:
        return events                                   # eaten is already told; nothing to dip it in
    evidence = [{"object_id": packet, "observation": "the packet is torn open with food still inside"}]
    if cup:
        evidence.append({"object_id": cup, "observation": "the cup holds a drink"})
        desc = (f"the {food} was most likely dipped into the drink in the {labels.get(cup, 'cup')} before it was eaten"
                if eaten else f"a {food} was most likely dipped into the drink in the {labels.get(cup, 'cup')} and eaten")
    else:
        desc = f"a {food} was most likely eaten from the opened packet"
    event = {"id": f"e{len(events) + 1}", "description": desc, "minutes_before_min": 0.0,
             "minutes_before_max": source["minutes_before_max"], "model_range": [0.0, source["minutes_before_max"]],
             "clock": "no_clock", "clock_rule": None, "trace": None, "confidence": 0.5,
             "alternatives": ["the packet was opened and nothing was eaten yet"], "undo": "mark", "undo_object": packet,
             "source": "inferred: an opened food packet" + (" and a drink" if cup else ""), "evidence": evidence}
    return events + [event]
