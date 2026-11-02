"""Trace clocks: physical traces that fade or change at a roughly known rate.

Muse Spark Open only *names* the trace it sees (from a fixed list). This table, not the model,
turns the trace into a time range, so the time estimate rests on physics we can state and test.
Ranges are rules of thumb for an indoor room at 20-32 C, in minutes between the event that made
the trace and the photo. They are deliberately wide, and the evaluation checks them.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Clock:
    min_minutes: float
    max_minutes: float
    rule: str


CLOCKS = {
    "steam":             Clock(0, 10, "visible steam from a hot drink lasts only minutes"),
    "liquid_film":       Clock(10, 90, "a skin forms on milk or tea only after it has cooled"),
    "ice_whole":         Clock(0, 10, "ice still whole means it went in recently"),
    "ice_partly_melted": Clock(5, 45, "ice part-melted in a drink at room temperature"),
    "condensation":      Clock(0, 30, "droplets on a cold glass build up soon after it is poured"),
    "wet_spill":         Clock(0, 45, "a thin spill on a hard surface dries within about an hour"),
    "wet_ring":          Clock(0, 45, "a wet ring under a cup has not dried yet"),
    "dry_ring_stain":    Clock(30, 1440, "a dried ring means the liquid sat long enough to dry"),
    "damp_cloth":        Clock(0, 180, "a damp cloth dries over a few hours"),
}

NO_CLOCK = {"crumbs", "open_package", "partially_eaten", "lid_off", "utensil_used",
            "door_or_drawer_open", "displaced_furniture", "residue", "level_dropped"}   # show WHAT happened, not WHEN


# What each trace looks like, and what it is often confused with. Used when reading the photo and when
# the verifier confirms a trace before its clock may date anything (run 2: bubbles were read as a skin).
LOOKS = {
    "steam":             "a soft white haze rising from a hot drink (not glare or reflections)",
    "liquid_film":       "a matte, wrinkled skin covering the surface of milk or tea "
                         "(small bubbles, froth or shine are NOT a skin)",
    "ice_whole":         "ice cubes with sharp edges in a drink",
    "ice_partly_melted": "ice pieces with rounded edges in a drink",
    "condensation":      "small water droplets on the OUTSIDE of a cold glass",
    "wet_spill":         "a shiny puddle or wet patch of liquid on a surface",
    "wet_ring":          "a still-wet ring of liquid where a cup stands or stood",
    "dry_ring_stain":    "a dried ring-shaped stain where a cup stood",
    "damp_cloth":        "a cloth darker or shiny from water",
    "crumbs":            "small loose bits of food on a surface",
    "open_package":      "a wrapper or box that is torn or open",
    "partially_eaten":   "food with bites or pieces missing",
    "lid_off":           "a container with its lid removed",
    "utensil_used":      "a spoon, knife or fork with food or liquid on it, or left out after use",
    "door_or_drawer_open": "a door or drawer left open",
    "displaced_furniture": "a chair or furniture pushed out of its usual place",
    "residue":           "drips, film or leftovers inside or on a container after its contents were poured or used",
    "level_dropped":     "a wet film or band on the INSIDE wall of a cup ABOVE the current drink level, all the way "
                         "round: it was fuller before, so some was drunk or poured out (a few scattered drops are "
                         "splashes from pouring, not this)",
}


def traces_guide() -> str:
    return "\n".join(f"  - {k}: {v}" for k, v in LOOKS.items())


def clock_for(trace):
    return CLOCKS.get(trace) if trace else None


def apply_clock(lo: float, hi: float, trace):
    """Clamp a model's time range to the trace clock. Returns (lo, hi, status)."""
    c = clock_for(trace)
    if c is None:
        return lo, hi, "no_clock"
    if hi < lo:
        lo, hi = hi, lo
    new_lo, new_hi = max(lo, c.min_minutes), min(hi, c.max_minutes)
    if new_lo <= new_hi:
        return new_lo, new_hi, ("clock_agrees" if (new_lo, new_hi) == (lo, hi) else "narrowed_by_clock")
    return c.min_minutes, c.max_minutes, "replaced_by_clock"   # model contradicted physics


def clock_table_text(seen=None) -> str:
    """The clock rules for the traces actually SEEN (all of them if seen is None). Listing traces that
    are not in the photo only tempts the model to talk about what is missing."""
    keys = [k for k in CLOCKS if seen is None or k in seen]
    lines = [f"- {k}: {CLOCKS[k].min_minutes:g} to {CLOCKS[k].max_minutes:g} min ({CLOCKS[k].rule})" for k in keys]
    undated = sorted(t for t in NO_CLOCK if seen is None or t in seen)
    if undated:
        lines.append("- " + ", ".join(undated) + ": no clock (they show what happened, not when)")
    return "\n".join(lines) if lines else "(no physical traces with known timing were seen)"
