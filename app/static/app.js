// Retrodict page: earlier photos, a live investigation, the time ruler and the evidence for each event.
const $ = (id) => document.getElementById(id);
const STEP_NAMES = { perceive: "Find the objects", propose: "Propose what happened", gather: "Find the evidence",
                     read: "Read and name each object", investigate: "Investigate", second_look: "Second look",
                     verify: "Check every claim", rewind: "Make the rewind" };
const VERDICT = { yes: "confirmed", partly: "partly right", unclear: "can't tell", no: "contradicted" };
let current = null;

function esc(s) { return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function fmt(m) { return m < 120 ? `${+m.toFixed(1)} min` : `${(m / 60).toFixed(1)} h`; }
const DATED = ["clock_agrees", "narrowed_by_clock", "replaced_by_clock"];
// A time is shown only when a clue in the photo gave one; otherwise the model's guess is not a time.
function isDated(e) { return "dated" in e ? e.dated : DATED.includes(e.clock); }
function whenText(e) {
  const lo = e.minutes_before_min, hi = e.minutes_before_max;
  if (!isDated(e)) return lo > 0 ? `at least ${fmt(lo)} before` : "time unknown";
  if (lo === 0) return `up to ${fmt(hi)} before`;
  return hi < 120 ? `${+lo.toFixed(1)} to ${+hi.toFixed(1)} min before` : `${fmt(lo)} to ${fmt(hi)} before`;
}

async function loadRuns(selected) {
  const runs = await (await fetch("/api/runs")).json();
  $("runs").innerHTML = runs.map((r) =>
    `<button data-name="${esc(r.name)}" title="${esc(r.summary)}" aria-current="${r.name === selected}">
       <img src="/files/${encodeURIComponent(r.name)}/input.jpg" alt="${esc(r.summary || r.name)}" loading="lazy"></button>`).join("");
  $("runs").querySelectorAll("button").forEach((b) => b.onclick = () => openRun(b.dataset.name));
  if (!selected && runs.length) openRun(runs[0].name);
}

async function openRun(name) {
  const r = await fetch(`/api/runs/${encodeURIComponent(name)}`);
  if (!r.ok) return;
  const { result, files } = await r.json();
  current = { name, result, files };
  document.querySelectorAll("#runs button").forEach((b) => b.setAttribute("aria-current", b.dataset.name === name));
  render();
}

function labelOf(oid) {
  const res = current.result;
  const o = (res.objects || []).find((x) => x.id === oid);
  return o ? o.label : oid;
}

function plainIds(text) {           // the agent writes "o4"; people read "the mug"
  return String(text ?? "")
    .replace(/\b(o\d+) (\w+)\b/g, (m, id, w) => (w.toLowerCase() === labelOf(id).toLowerCase() ? `the ${w}` : m))
    .replace(/\b(\w+) (o\d+)\b/g, (m, w, id) => (w.toLowerCase() === labelOf(id).toLowerCase() ? `${w}` : `${w} the ${labelOf(id)}`))
    .replace(/\b(o\d+)\b/g, (_, id) => `the ${labelOf(id)}`)
    .replace(/\bthe the\b/gi, "the");
}

function moveText(m) {
  const on = m.object_id ? labelOf(m.object_id) : "";
  if (m.move === "zoom") return `Looked closely at the ${esc(on)}`;
  if (m.move === "search") {
    const found = m.result.startsWith("found");
    return `Searched for “${esc(m.phrase)}”${on ? ` in and around the ${esc(on)}` : ""}: ${found ? "found it" : "nothing new"}`;
  }
  if (m.move === "clock") return `Checked how long a trace lasts: ${esc(m.result)}`;
  if (m.move === "finish") return `Wrote the timeline (${esc(m.result)})`;
  return esc(m.result);
}

function renderSteps(trace, running) {
  const done = Object.fromEntries((trace || []).map((s) => [s.step, s]));
  const order = ["perceive", "propose", "gather", "read", "investigate", "second_look", "verify", "rewind"]
    .filter((k) => !done.verify || done[k] || k === "rewind");      // a finished run lists only its own steps
  $("steps").innerHTML = order.map((k) => {
    const s = done[k];
    const cls = s ? "done" : (running === k ? "running" : "");
    return `<li class="${cls}"><span>${STEP_NAMES[k]}</span>${s && s.seconds !== "" ? `<span class="t">${s.seconds}s</span>` : ""}</li>`;
  }).join("");
}

function renderMoves(moves) {
  $("moves").innerHTML = (moves || []).map((m) =>
    `<li><div class="what">${moveText(m)}</div>${m.thought ? `<div class="why">${esc(plainIds(m.thought))}</div>` : ""}</li>`).join("");
}

function setView(view) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.view === view));
  const base = `/files/${encodeURIComponent(current.name)}/`;
  const video = $("rewind"), img = $("photo");
  const clip = view === "rewind" ? "rewind.mp4" : view === "proof" ? "proof_rewind.mp4" : null;
  if (clip && current.files.includes(clip)) {
    img.hidden = true; video.hidden = false; video.src = base + clip; video.play().catch(() => {});
  } else {
    video.pause(); video.hidden = true; img.hidden = false;
    img.src = base + (view === "objects" && current.files.includes("overlay.png") ? "overlay.png" : "input.jpg");
  }
}

function render() {
  const res = current.result;
  $("empty").hidden = true; $("stage").hidden = false;
  document.querySelector('.tabs [data-view="rewind"]').hidden = !current.files.includes("rewind.mp4");
  document.querySelector('.tabs [data-view="proof"]').hidden = !current.files.includes("proof_rewind.mp4");
  setView(current.files.includes("rewind.mp4") ? "rewind" : "photo");   // the clip plays as soon as a run opens
  $("note").textContent = res.photo_note || "";
  $("summary").textContent = res.summary || "";
  renderSteps(res.trace.concat(current.files.includes("rewind.mp4") ? [{ step: "rewind", seconds: "" }] : []));
  renderMoves(res.moves);
  renderRuler(res);
  loadCheck(current.name);
}

function renderRuler(res) {
  const events = res.events || [];
  const dated = events.filter(isDated);
  const maxM = Math.max(10, ...dated.map((e) => e.minutes_before_max));
  const narrow = $("ruler").clientWidth < 560;
  const ticks = (narrow ? [0, 5, 15, 60, 240, 1440] : [0, 1, 2, 5, 10, 15, 30, 60, 120, 240, 480, 1440])
    .filter((t) => t <= maxM * 1.05);
  if (ticks[ticks.length - 1] < maxM) ticks.push(Math.ceil(maxM));
  const top = ticks[ticks.length - 1];
  // square-root scale: the last minutes get room, a long past stays on screen; now is on the right
  const x = (m) => 100 - (Math.sqrt(m) / Math.sqrt(top)) * 100;
  const rows = events.map((e, i) => {
    let mark;
    if (isDated(e)) {
      const left = x(e.minutes_before_max), right = x(e.minutes_before_min);
      const cls = [e.status === "contested" ? "contested" : "", e.status === "partly_verified" ? "partly" : "",
                   e.clock && DATED.includes(e.clock) ? "clocked" : ""].join(" ");
      mark = `<span class="band ${cls}" style="left:${left}%;width:${Math.max(0.8, right - left)}%" title="${esc(whenText(e))}"></span>`;
    } else {
      mark = `<span class="unknown">${esc(whenText(e))}</span>`;
    }
    const label = `<span class="label"><span class="num">${i + 1}</span>${esc(e.description)}</span>`;
    if (!dated.length)                 // nothing dated: an ordered list, no empty time axis
      return `<button class="row" data-i="${i}" aria-pressed="false">${label}<span class="chip">${esc(whenText(e))}</span></button>`;
    return `<button class="row" data-i="${i}" aria-pressed="false">${label}
      <span class="track ${isDated(e) ? "" : "undated"}">${mark}</span></button>`;
  }).join("");
  const axis = ticks.map((t) => `<span class="tick ${t === 0 ? "now" : ""}" style="left:${x(t)}%">${t === 0 ? "photo" : fmt(t)}</span>`).join("");
  const foot = dated.length ? `<div class="axis">${axis}</div>`
    : events.length ? `<p class="order-note">Nothing in this photo shows how long ago these happened, so they are shown in order only, oldest first.</p>` : "";
  $("ruler").classList.toggle("orderonly", !dated.length);
  $("ruler").innerHTML = (rows || `<p>No sign of recent activity.</p>`) + foot;
  $("ruler").querySelectorAll(".row").forEach((b) => b.onclick = () => select(+b.dataset.i));
  if (events.length) select(0); else $("detail").hidden = true;
}

// How it did: score.json from scripts/score.py (the answer key from the real video + V-JEPA 2 watching it)
async function loadCheck(name) {
  $("check").hidden = true;
  if (!current.files.includes("score.json")) return;
  const r = await fetch(`/files/${encodeURIComponent(name)}/score.json`);
  if (!r.ok || !current || current.name !== name) return;
  renderCheck(await r.json());
}

function shortText(t, limit = 110) {
  t = plainIds(t).trim().replace(/\.$/, "");
  if (t.length <= limit) return t;
  const cut = t.slice(0, limit), at = Math.max(cut.lastIndexOf(", "), cut.lastIndexOf(" "));
  return cut.slice(0, at > 40 ? at : limit).replace(/,$/, "") + "…";
}

function renderCheck(data) {
  const s = data.score, w = data.witness, scene = data.scene || {};
  $("check").hidden = false;
  $("c-scene").textContent = scene.split === "dev"
    ? "Practice scene: it was used while building Retrodict, so this is not a fair test." : "";
  const parts = [`Found ${s.found} of ${s.total} things that really happened`];
  if (s.order_pairs) parts.push(`${s.order_agree} of ${s.order_pairs} pairs in the right order`);
  if (s.matched_by === "blind judge") {
    const n = (s.false_claims || []).length;
    parts.push(n === 1 ? "1 claim judged wrong" : `${n} claims judged wrong`);
  } else {
    parts.push(s.unsupported.length === 1 ? "1 thing said that didn't happen" : `${s.unsupported.length} things said that didn't happen`);
  }
  $("c-sum").textContent = parts.join(" · ") + ".";

  const wit = Object.fromEntries(((w && w.events) || []).map((r) => [r.id, r]));
  const real = [...s.events];              // the key lists events in the order they happened
  const retro = s.retro || [];
  const left = real.map((e, i) => {
    const v = wit[e.id];
    const seen = !v ? "" : !v.can_see ? "Video model: has no word for this"
      : v.seen ? `Video model saw “${v.plain}” at ${v.at[0]}–${v.at[1]} s` : "Video model: didn't see it";
    return `<li class="${e.found ? "" : "missed"}" data-key="${esc(e.id)}"><span class="num">${i + 1}</span>
      <span><b>${esc(e.what.charAt(0).toUpperCase() + e.what.slice(1))}</b>
      ${e.found ? "" : `<span class="status missed">Missed</span>`}
      ${seen ? `<span class="witness" title="${esc(v.label || "")}">${esc(seen)}</span>` : ""}</span></li>`;
  }).join("");
  const right = retro.map((r, i) =>
    `<li class="${r.key && !r.wrong ? "" : "made-up"}" data-keys="${esc((r.keys && r.keys.length ? r.keys : [r.key || ""]).join(" "))}"><span class="num">${i + 1}</span>
      <span>${esc(shortText(r.text))}${r.wrong ? `<span class="status missed">Judged wrong</span>`
        : r.key ? "" : `<span class="status other">Not in the answer key</span>`}</span></li>`).join("");
  $("c-order").innerHTML = `<div class="col-head">What really happened (video)</div><div></div>
    <div class="col-head">What Retrodict said (photo only)</div>
    <ol class="real">${left}</ol><svg class="links" aria-hidden="true"></svg><ol class="said">${right}</ol>`;
  drawLinks();

  const [lo, hi] = s.real_span_min || [0, 0];
  let when = `When: in the video, all of this happened ${+lo.toFixed(1)} to ${+hi.toFixed(1)} min before the photo. `;
  when += s.dated ? `Retrodict dated ${s.dated} of them from clues; ${s.covered} of those times contain the real time`
                    + (s.median_width_min != null ? `, about ${fmt(s.median_width_min)} wide.` : ".")
                  : "Retrodict found no clue in the photo that dates them, so it says “time unknown” instead of guessing.";
  $("c-when").textContent = when;
  $("c-extra").innerHTML = w ? `<li>V-JEPA 2, Meta's video model, watched the real video as a second witness: it saw
    ${w.seen} of the ${w.seeable} events it has words for, and ${w.stray_windows} of its ${w.action_windows} action
    moments matched nothing that really happened. It only knows hand actions, never object names.</li>` : "";
}

// Lines from each real event to the Retrodict step that describes it; a line that crosses another is an order mistake.
function drawLinks() {
  const box = $("c-order"), svg = box.querySelector("svg.links");
  if (!svg || box.closest("[hidden]")) return;
  const sb = svg.getBoundingClientRect();
  const leftOf = Object.fromEntries([...box.querySelectorAll("ol.real li")].map((li, i) => [li.dataset.key, { li, i }]));
  const pairs = [...box.querySelectorAll("ol.said li")].flatMap((li, j) =>      // one claim may cover several events
    (li.dataset.keys || "").split(" ").filter((k) => k && leftOf[k]).map((k) => ({ j, li, l: leftOf[k] })));
  const mid = (el) => { const r = el.getBoundingClientRect(); return r.top + Math.min(r.height / 2, 14) - sb.top; };
  svg.setAttribute("viewBox", `0 0 ${sb.width} ${sb.height}`);
  svg.innerHTML = pairs.map((p) => {
    const crosses = pairs.some((q) => q !== p && (q.l.i - p.l.i) * (q.j - p.j) < 0);
    const y1 = mid(p.l.li), y2 = mid(p.li);
    return `<path class="${crosses ? "wrong" : "right"}" d="M2 ${y1} C ${sb.width / 2} ${y1}, ${sb.width / 2} ${y2}, ${sb.width - 2} ${y2}"/>`;
  }).join("");
}
window.addEventListener("resize", () => { if (current) drawLinks(); });

function select(i) {
  const e = current.result.events[i];
  document.querySelectorAll(".row").forEach((b) => b.setAttribute("aria-pressed", +b.dataset.i === i));
  $("detail").hidden = false;
  $("d-when").textContent = whenText(e);
  $("d-desc").textContent = e.description;
  const why = [e.clock_rule && `Physics: ${e.clock_rule}.`, e.bounded_by && `Also ${e.bounded_by}.`,
               e.trace_check && e.trace_check.verdict !== "yes" && `The checker did not confirm the ${String(e.trace_check.trace).replace(/_/g, " ")}, so no clock was used.`,
               !isDated(e) && "Nothing in the photo shows when this happened, so no time is given."]
              .filter(Boolean).join(" ");
  $("d-physics").hidden = !why; $("d-physics").textContent = why;
  $("d-evidence").innerHTML = e.evidence.map((ev) =>
    `<li><span class="verdict ${esc(ev.verdict)}">${VERDICT[ev.verdict] || esc(ev.verdict)}</span>
       <span><b>${esc(labelOf(ev.object_id))}</b>: ${esc(ev.observation)}${ev.note ? `<div class="note">Checker: ${esc(plainIds(ev.note))}</div>` : ""}</span></li>`).join("");
  $("d-alts").innerHTML = (e.alternatives || []).map((a) => `<li>${esc(a)}</li>`).join("");
  const oid = e.undo_object || e.evidence[0]?.object_id;
  const file = `detail_${oid}.jpg`;
  $("d-views").parentElement.hidden = !current.files.includes(file);
  $("d-views").src = `/files/${encodeURIComponent(current.name)}/${file}`;
}

async function upload(file) {
  if (!file) return;
  const body = new FormData();
  body.append("photo", file);
  $("empty").hidden = true; $("stage").hidden = false;
  $("summary").textContent = "Working backward from the photo…";
  $("ruler").innerHTML = ""; $("detail").hidden = true; $("moves").innerHTML = ""; $("note").textContent = "";
  $("photo").hidden = false; $("rewind").hidden = true; $("photo").src = URL.createObjectURL(file);
  const r = await fetch("/api/run", { method: "POST", body });
  if (!r.ok) { $("summary").innerHTML = `<span class="error">${esc((await r.json()).detail)}</span>`; return; }
  const { job, name } = await r.json();
  const trace = [], moves = [];
  renderSteps(trace, "perceive");
  const es = new EventSource(`/api/jobs/${job}/events`);
  es.onmessage = async (msg) => {
    const { kind, data } = JSON.parse(msg.data);
    if (kind === "step_started") renderSteps(trace, data.step);
    if (kind === "step_done") { trace.push(data); renderSteps(trace, null); }
    if (kind === "move") { moves.push(data); current = current && current.name === name ? current : { name, result: { objects: [] }, files: [] }; renderMoves(moves); }
    if (kind === "queued") $("summary").textContent = data.message;
    if (kind === "error") { es.close(); $("summary").innerHTML = `<span class="error">${esc(data.message)}</span> ${esc(data.fix || "")}`; }
    if (kind === "done") { es.close(); await loadRuns(name); await openRun(name); }
  };
}

document.querySelectorAll(".tabs button").forEach((b) => b.onclick = () => setView(b.dataset.view));
$("pick").onchange = (e) => upload(e.target.files[0]);
$("snap").onchange = (e) => upload(e.target.files[0]);
loadRuns(null);
