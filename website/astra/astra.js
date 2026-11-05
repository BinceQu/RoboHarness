(() => {
  const base = document.body.dataset.media;
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const title = (name) => name.replace(/_/g, " ");
  const band = (e) => e.q === null ? "q-none" : e.q >= 1 ? "q-full" : e.q > 0 ? "q-mid" : "q-zero";
  const qText = (e) => e.q === null ? "—" : e.q.toFixed(e.q % 1 ? 2 : 1);
  const time = (s) => s >= 60 ? `${Math.floor(s / 60)}m ${String(Math.round(s % 60)).padStart(2, "0")}s` : `${Math.round(s)}s`;
  const FILTERS = [["all", "All", () => true], ["full", "Full success", (e) => e.q >= 1],
    ["mid", "Partial", (e) => e.q > 0 && e.q < 1], ["zero", "Zero", (e) => e.q === 0], ["none", "Unscored", (e) => e.q === null]];
  const SPEEDS = [1, 2, 4, 8, 16];
  let data, shown = [], current = -1, filter = "all", speed = 4;

  fetch("episodes.json").then((r) => r.json()).then((d) => { data = d; render(); });

  function render() {
    const s = data.summary;
    $("stats").innerHTML = [["key", s.mean_q.toFixed(3), `mean Q · ${s.scored} scored tasks`], ["", s.tasks, "tasks attempted"],
      ["", s.full_success, "full success"], ["", s.partial, "partial score"], ["", s.zero, "zero score"]]
      .map(([c, b, t]) => `<div class="stat ${c}"><b>${b}</b><span>${t}</span></div>`).join("");
    $("strip").innerHTML = data.episodes.map((e) =>
      `<button type="button" class="${band(e)}" data-id="${e.id}" title="${e.id.toUpperCase()} ${esc(title(e.task))} · Q ${qText(e)}"></button>`).join("");
    $("strip").addEventListener("click", (ev) => ev.target.dataset.id && openId(ev.target.dataset.id));
    $("chips").innerHTML = FILTERS.map(([k, label, f]) =>
      `<button type="button" class="chip${k === filter ? " on" : ""}" data-k="${k}">${label}<small>${data.episodes.filter(f).length}</small></button>`).join("");
    $("chips").addEventListener("click", (ev) => {
      const b = ev.target.closest(".chip"); if (!b) return;
      filter = b.dataset.k; document.querySelectorAll(".chip").forEach((c) => c.classList.toggle("on", c === b)); layout();
    });
    $("settings").innerHTML = Object.entries(data.settings).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
    $("v-settings").innerHTML = $("settings").innerHTML;
    $("v-speed").insertAdjacentHTML("beforeend", SPEEDS.map((x) => `<button type="button" data-x="${x}">${x}×</button>`).join(""));
    $("v-speed").addEventListener("click", (ev) => ev.target.dataset.x && setSpeed(+ev.target.dataset.x));
    $("search").addEventListener("input", layout);
    $("sort").addEventListener("change", layout);
    layout();
    if (location.hash.length > 1) openId(location.hash.slice(1));
  }

  function layout() {
    const q = $("search").value.trim().toLowerCase().replace(/\s+/g, "_");
    const keep = FILTERS.find(([k]) => k === filter)[2];
    shown = data.episodes.filter((e) => keep(e) && (!q || e.task.includes(q) || e.id.includes(q)));
    const order = $("sort").value, val = (e) => e.q === null ? -1 : e.q;
    if (order === "q-desc") shown.sort((a, b) => val(b) - val(a) || a.task_id - b.task_id);
    if (order === "q-asc") shown.sort((a, b) => val(a) - val(b) || a.task_id - b.task_id);
    if (order === "len") shown.sort((a, b) => b.steps - a.steps);
    if (order === "id") shown.sort((a, b) => a.task_id - b.task_id);
    $("grid").innerHTML = shown.map((e, i) => `
      <button type="button" class="tile" data-i="${i}" aria-label="Open ${esc(title(e.task))}">
        <div class="media"><img loading="lazy" src="${base}${e.media.posters.file}" alt="">
          <span class="tid">${e.id.toUpperCase()}</span><span class="q ${band(e)}">Q ${qText(e)}</span>
          <span class="bar" style="width:${Math.min(100, 100 * e.steps / e.max_ticks).toFixed(1)}%"></span></div>
        <div class="body"><h4>${esc(title(e.task))}</h4>
          <div class="meta">${e.steps.toLocaleString()} / ${e.max_ticks.toLocaleString()} ticks · ${time(e.video_seconds)}</div></div>
      </button>`).join("");
    $("empty").hidden = shown.length > 0;
  }

  // Grid previews: time-lapse on hover / focus.
  const grid = $("grid");
  const startPreview = (tile) => {
    if (!tile || tile.querySelector("video")) return;
    const e = shown[+tile.dataset.i], v = document.createElement("video");
    Object.assign(v, {muted: true, loop: true, playsInline: true, src: base + e.media.previews.file});
    v.addEventListener("playing", () => tile.classList.add("playing"));
    tile.querySelector(".media").insertBefore(v, tile.querySelector(".tid"));
    v.play().catch(() => {});
  };
  const stopPreview = (tile) => { const v = tile && tile.querySelector("video"); if (v) { v.remove(); tile.classList.remove("playing"); } };
  grid.addEventListener("mouseover", (ev) => startPreview(ev.target.closest(".tile")));
  grid.addEventListener("mouseout", (ev) => { const t = ev.target.closest(".tile"); if (t && !t.contains(ev.relatedTarget)) stopPreview(t); });
  grid.addEventListener("focusin", (ev) => startPreview(ev.target.closest(".tile")));
  grid.addEventListener("focusout", (ev) => stopPreview(ev.target.closest(".tile")));
  grid.addEventListener("click", (ev) => { const t = ev.target.closest(".tile"); if (t) open(+t.dataset.i); });

  // Viewer.
  const dlg = $("viewer"), video = $("v-video");
  function setSpeed(x) {
    speed = x; video.playbackRate = x;
    $("v-speed").querySelectorAll("button").forEach((b) => b.classList.toggle("on", +b.dataset.x === x));
  }
  function openId(id) {
    let i = shown.findIndex((e) => e.id === id);
    if (i < 0) { filter = "all"; $("search").value = ""; document.querySelectorAll(".chip").forEach((c) => c.classList.toggle("on", c.dataset.k === "all")); layout(); i = shown.findIndex((e) => e.id === id); }
    if (i >= 0) open(i);
  }
  function open(i) {
    current = (i + shown.length) % shown.length;
    const e = shown[current];
    $("v-id").textContent = `${e.id.toUpperCase()} · INSTANCE ${e.instance} · ROLLOUT ${e.rollout}`;
    $("v-title").textContent = title(e.task);
    video.poster = base + e.media.posters.file;
    video.src = base + e.media.videos.file;
    setSpeed(e.video_seconds > 120 ? 8 : e.video_seconds > 40 ? 4 : 2);
    video.play().catch(() => {});
    const m = [[qText(e), e.q === null ? "no official score" : e.success ? "Q · task succeeded" : "Q · official q-score"],
      [`${e.steps.toLocaleString()} / ${e.max_ticks.toLocaleString()}`, "ticks used / budget"],
      [time(e.sim_seconds), "simulated time"], [e.bash_calls ?? "—", "Bash calls by the agent"],
      [title(e.scene), "scene"]];
    $("v-metrics").innerHTML = m.map(([b, t]) => `<div class="m"><b>${esc(b)}</b><span>${esc(t)}</span></div>`).join("") +
      (e.note ? `<div class="m note">${esc(e.note)}</div>` : "");
    $("v-prompt").textContent = e.prompt;
    if (!dlg.open) dlg.showModal();
    dlg.scrollTop = 0;
    history.replaceState(null, "", "#" + e.id);
  }
  function close() { video.pause(); video.removeAttribute("src"); video.load(); dlg.close(); }
  dlg.addEventListener("close", () => { video.pause(); history.replaceState(null, "", location.pathname + location.search); });
  dlg.addEventListener("click", (ev) => { if (ev.target === dlg) close(); });
  video.addEventListener("loadedmetadata", () => { video.playbackRate = speed; });
  $("v-close").onclick = close;
  $("v-prev").onclick = () => open(current - 1);
  $("v-next").onclick = () => open(current + 1);
  document.addEventListener("keydown", (ev) => {
    if (!dlg.open || ev.target.tagName === "INPUT") return;
    if (ev.key === "ArrowLeft") open(current - 1);
    if (ev.key === "ArrowRight") open(current + 1);
  });
  $("v-copy").onclick = (ev) => navigator.clipboard.writeText($("v-prompt").textContent)
    .then(() => { ev.target.textContent = "Copied"; setTimeout(() => (ev.target.textContent = "Copy"), 1500); });
})();
