// Retrodict page: earlier photos, a live investigation, the time ruler and the evidence for each event.
const $ = (id) => document.getElementById(id);
const STEP_NAMES = { perceive: "Find the objects", read: "Read each object", investigate: "Investigate",
                     verify: "Check every claim", rewind: "Make the rewind" };
const VERDICT = { yes: "confirmed", partly: "partly right", unclear: "can't tell", no: "contradicted" };
let current = null;

function esc(s) { return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function fmt(m) { return m < 120 ? `${+m.toFixed(1)} min` : `${(m / 60).toFixed(1)} h`; }
function whenText(e) {
  const lo = e.minutes_before_min, hi = e.minutes_before_max;
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
  const order = ["perceive", "read", "investigate", "verify", "rewind"];
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
  if (view === "rewind" && current.files.includes("rewind.mp4")) {
    img.hidden = true; video.hidden = false; video.src = base + "rewind.mp4"; video.play().catch(() => {});
  } else {
    video.pause(); video.hidden = true; img.hidden = false;
    img.src = base + (view === "objects" && current.files.includes("overlay.png") ? "overlay.png" : "input.jpg");
  }
}

function render() {
  const res = current.result;
  $("empty").hidden = true; $("stage").hidden = false;
  document.querySelector('.tabs [data-view="rewind"]').hidden = !current.files.includes("rewind.mp4");
  setView(current.files.includes("rewind.mp4") ? "rewind" : "photo");   // the clip plays as soon as a run opens
  $("note").textContent = res.photo_note || "";
  $("summary").textContent = res.summary || "";
  renderSteps(res.trace.concat(current.files.includes("rewind.mp4") ? [{ step: "rewind", seconds: "" }] : []));
  renderMoves(res.moves);
  renderRuler(res);
}

function renderRuler(res) {
  const events = res.events || [];
  const maxM = Math.max(10, ...events.map((e) => e.minutes_before_max));
  const narrow = $("ruler").clientWidth < 560;
  const ticks = (narrow ? [0, 5, 15, 60, 240, 1440] : [0, 1, 2, 5, 10, 15, 30, 60, 120, 240, 480, 1440])
    .filter((t) => t <= maxM * 1.05);
  if (ticks[ticks.length - 1] < maxM) ticks.push(Math.ceil(maxM));
  const top = ticks[ticks.length - 1];
  // square-root scale: the last minutes get room, a long past stays on screen; now is on the right
  const x = (m) => 100 - (Math.sqrt(m) / Math.sqrt(top)) * 100;
  const rows = events.map((e, i) => {
    const left = x(e.minutes_before_max), right = x(e.minutes_before_min);
    const cls = [e.status === "contested" ? "contested" : "", e.status === "partly_verified" ? "partly" : "",
                 e.clock && !["no_clock", "clock_not_applied"].includes(e.clock) ? "clocked" : ""].join(" ");
    return `<button class="row" data-i="${i}" aria-pressed="false">
      <span class="label">${esc(e.description)}</span>
      <span class="track"><span class="band ${cls}" style="left:${left}%;width:${Math.max(0.8, right - left)}%"
            title="${esc(whenText(e))}"></span></span></button>`;
  }).join("");
  const axis = ticks.map((t) => `<span class="tick ${t === 0 ? "now" : ""}" style="left:${x(t)}%">${t === 0 ? "photo" : fmt(t)}</span>`).join("");
  $("ruler").innerHTML = (rows || `<p>No sign of recent activity.</p>`) + `<div class="axis">${axis}</div>`;
  $("ruler").querySelectorAll(".row").forEach((b) => b.onclick = () => select(+b.dataset.i));
  if (events.length) select(0); else $("detail").hidden = true;
}

function select(i) {
  const e = current.result.events[i];
  document.querySelectorAll(".row").forEach((b) => b.setAttribute("aria-pressed", +b.dataset.i === i));
  $("detail").hidden = false;
  $("d-when").textContent = whenText(e);
  $("d-desc").textContent = e.description;
  const why = [e.clock_rule && `Physics: ${e.clock_rule}.`, e.bounded_by && `Also ${e.bounded_by}.`,
               e.trace_check && e.trace_check.verdict !== "yes" && `The checker did not confirm the ${String(e.trace_check.trace).replace(/_/g, " ")}, so no clock was used.`]
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
