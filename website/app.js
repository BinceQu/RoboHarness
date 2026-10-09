"use strict";

(() => {
  const $ = id => document.getElementById(id);
  const tasks = JSON.parse($("archive-data").textContent);
  const byId = new Map(tasks.map(task => [task.id, task]));
  const select = $("task-select");
  const runSelect = $("run-task");
  const chart = $("score-chart");
  const explorer = $("archive-explorer");
  const video = $("rollout-video");
  const panel = $("trace-panel");
  const message = $("trace-message");
  const motion = matchMedia("(prefers-reduced-motion: reduce)");
  const narrow = matchMedia("(max-width: 680px)");
  const query = new URLSearchParams(location.search);
  const views = ["below", "beside", "cards"];
  const traceCache = new Map();
  let view = views.includes(query.get("trace")) ? query.get("trace") : "beside";
  let activeTask = null;
  let traceSteps = [];
  let traceIndex = -2;
  let traceRequest = null;
  let generation = 0;
  let animationFrame = 0;
  let targetMessage = "";
  let typeStarted = 0;
  let playbackRate = 4;

  runSelect.replaceChildren(...Array.from(select.options, option => option.cloneNode(true)));
  runSelect.disabled = false;
  video.autoplay = !motion.matches;

  const timeLabel = seconds => {
    const whole = Math.max(0, Math.floor(seconds));
    return `${String(Math.floor(whole / 60)).padStart(2, "0")}:${String(whole % 60).padStart(2, "0")}`;
  };
  const family = tool => /gripper/.test(tool) ? "gripper" :
    /^(track_|move_|spin_)/.test(tool) ? "geometry" :
    /^(adjust_|set_arm|exec_|reset_)/.test(tool) ? "motion" : "observation";

  function shareSelection() {
    const url = new URL(location.href);
    if (view === "beside") url.searchParams.delete("trace");
    else url.searchParams.set("trace", view);
    if (activeTask.id === "task01") url.searchParams.delete("task");
    else url.searchParams.set("task", activeTask.id);
    history.replaceState(null, "", url);
  }

  function placeTrace() {
    explorer.dataset.view = view;
    $(view === "beside" && !narrow.matches ? "trace-beside" : "trace-below").append(panel);
    document.querySelectorAll(".view-buttons button").forEach(button => {
      button.setAttribute("aria-pressed", String(button.dataset.view === view));
    });
  }

  function streamMessage(now) {
    if (!message.classList.contains("is-typing")) return;
    const length = Math.min(targetMessage.length, Math.floor((now - typeStarted) / 1000 * 160));
    message.textContent = targetMessage.slice(0, length);
    if (length === targetMessage.length) message.classList.remove("is-typing");
  }

  function finishMessage() {
    message.textContent = targetMessage;
    message.classList.remove("is-typing");
  }

  function historyCards(index) {
    const cards = traceSteps.slice(Math.max(0, index - 2), index).map(step => {
      const card = document.createElement("div");
      card.className = "trace-card";
      const heading = document.createElement("div");
      heading.className = "trace-card-heading";
      const label = document.createElement("span");
      label.textContent = "Agent message";
      const time = document.createElement("time");
      time.textContent = timeLabel(step.t);
      heading.append(label, time);
      const text = document.createElement("p");
      text.textContent = step.message || "";
      const chip = document.createElement("span");
      chip.className = "tool-chip";
      chip.dataset.family = family(step.tool);
      chip.textContent = step.tool;
      card.append(heading, chip, text);
      return card;
    });
    $("trace-history").replaceChildren(...cards);
  }

  function renderTrace(force = false) {
    if (!traceSteps.length) return;
    const t = video.currentTime;
    $("trace-time").textContent = timeLabel(t);
    let lo = 0, hi = traceSteps.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (traceSteps[mid].t <= t + 0.001) lo = mid + 1;
      else hi = mid;
    }
    const index = lo - 1;
    if (index === traceIndex) {
      if (force) finishMessage();
      return;
    }
    const forward = index > traceIndex;
    traceIndex = index;
    panel.dataset.step = index;
    const step = traceSteps[index];
    const text = step?.message || (step ? "Recorded tool call" : "Agent messages appear as the rollout plays.");
    const animate = forward && !force && !video.paused && !video.seeking && !motion.matches;
    if (targetMessage !== text) {
      targetMessage = text;
      message.scrollTop = 0;
      if (animate) {
        typeStarted = performance.now();
        message.textContent = "";
        message.classList.add("is-typing");
      } else finishMessage();
    } else if (force) finishMessage();
    $("trace-tool").hidden = !step;
    $("trace-tool").textContent = step?.tool || "";
    $("trace-tool").dataset.family = family(step?.tool || "");
    $("trace-params").textContent = step ? Object.entries(step.params).map(([key, value]) =>
      `${key}: ${typeof value === "string" ? value : JSON.stringify(value)}`).join(" · ") : "";
    historyCards(Math.max(0, index));
    if (animate) {
      $("trace-card").getAnimations().forEach(animation => animation.cancel());
      $("trace-card").animate([{opacity: .5, transform: "translateY(7px)"},
        {opacity: 1, transform: "translateY(0)"}], {duration: 260, easing: "ease-out"});
    }
  }

  function tick(now) {
    animationFrame = 0;
    renderTrace();
    streamMessage(now);
    if (!video.paused && !video.ended && activeTask?.rollout) animationFrame = requestAnimationFrame(tick);
  }

  function startTick() {
    if (!animationFrame) animationFrame = requestAnimationFrame(tick);
  }

  async function loadTrace(task, token) {
    try {
      let steps = traceCache.get(task.id);
      if (!steps) {
        const controller = new AbortController();
        traceRequest = controller;
        const response = await fetch(task.rollout.trace, {signal: controller.signal});
        if (!response.ok) throw new Error("Trace unavailable");
        const data = await response.json();
        if (data.task !== task.id || data.run !== task.rollout.run || !Array.isArray(data.steps)) throw new Error("Trace mismatch");
        steps = data.steps;
        traceCache.set(task.id, steps);
      }
      if (token !== generation) return;
      traceSteps = steps;
      traceIndex = -2;
      renderTrace(true);
      if (!video.paused) startTick();
    } catch (error) {
      if (token !== generation || error.name === "AbortError") return;
      targetMessage = "The recorded trace could not load. Open the trace above to view it.";
      finishMessage();
    }
  }

  function updateRollout(task) {
    generation += 1;
    const token = generation;
    traceRequest?.abort();
    traceRequest = null;
    video.pause();
    cancelAnimationFrame(animationFrame);
    animationFrame = 0;
    traceSteps = [];
    traceIndex = -2;
    panel.dataset.step = "";
    targetMessage = "Loading recorded agent messages…";
    finishMessage();
    $("trace-history").replaceChildren();
    $("trace-tool").hidden = true;
    $("trace-params").textContent = "";
    $("trace-time").textContent = "00:00";
    $("rollout-error").hidden = true;
    $("rollout-figure").hidden = !task.rollout;
    $("rollout-empty").hidden = Boolean(task.rollout);
    panel.hidden = !task.rollout;
    $("reasoning-views").hidden = !task.rollout;
    video.dataset.task = task.id;
    if (!task.rollout) {
      video.removeAttribute("src");
      video.removeAttribute("poster");
      video.load();
      return;
    }
    const rollout = task.rollout;
    $("rollout-label").textContent = `Head camera · Instance ${rollout.instance}`;
    $("rollout-score").textContent = rollout.q == null ? "Partial · No final Q" : `Recorded Q ${rollout.q.toFixed(4)}`;
    $("rollout-note").hidden = rollout.q == null || !rollout.missingTailSeconds;
    $("rollout-note").textContent = rollout.missingTailSeconds ?
      `Recording ends ${rollout.missingTailSeconds.toFixed(1)} s before the evaluation.` : "";
    $("rollout-download").href = rollout.video;
    $("trace-source").href = rollout.trace;
    video.poster = rollout.poster;
    video.src = rollout.video;
    video.setAttribute("aria-label", `${task.name}, recorded head-camera rollout, instance ${rollout.instance}`);
    video.load();
    loadTrace(task, token);
    if (!motion.matches) video.play().catch(() => {});
  }

  function updateTask(id, share = false) {
    const task = byId.get(id);
    if (!task) return;
    const changed = activeTask?.id !== task.id;
    activeTask = task;
    select.value = task.id;
    runSelect.value = task.id;
    $("selected-mean").textContent = task.mean.toFixed(4);
    $("manifest-link").href = `https://github.com/BinceQu/RoboHarness/blob/main/tasks/${task.id}.json`;
    if (changed) updateRollout(task);
    chart.setAttribute("aria-label", `${task.name}. Archived Q-scores: ` +
      task.cases.map(item => `instance ${item.instance}: ${item.q.toFixed(4)}`).join("; "));
    chart.replaceChildren(...task.cases.map(item => {
      const column = document.createElement("div");
      column.className = "chart-column";
      column.setAttribute("aria-hidden", "true");
      const bar = document.createElement("div");
      bar.className = "chart-bar";
      bar.style.height = `${item.q * 100}%`;
      const value = document.createElement("span");
      value.className = "chart-value";
      value.textContent = item.q.toFixed(4);
      const label = document.createElement("span");
      label.className = "chart-label";
      label.textContent = item.instance;
      bar.append(value);
      column.append(bar, label);
      return column;
    }));
    $("run-command").textContent = `./scripts/reproduce_task.sh ${task.id} --gpu 0`;
    $("copy-feedback").textContent = "";
    if (share) shareSelection();
  }

  select.addEventListener("change", () => updateTask(select.value, true));
  runSelect.addEventListener("change", () => updateTask(runSelect.value, true));
  document.querySelectorAll(".view-buttons button").forEach(button => button.addEventListener("click", () => {
    view = button.dataset.view;
    placeTrace();
    shareSelection();
  }));
  narrow.addEventListener("change", placeTrace);
  motion.addEventListener("change", () => {
    video.autoplay = !motion.matches;
    if (motion.matches) { video.pause(); finishMessage(); }
  });
  $("playback-speed").addEventListener("change", event => {
    playbackRate = Number(event.target.value);
    video.playbackRate = playbackRate;
  });
  video.addEventListener("loadedmetadata", () => {
    video.defaultPlaybackRate = playbackRate;
    video.playbackRate = playbackRate;
  });
  video.addEventListener("playing", startTick);
  video.addEventListener("pause", () => { renderTrace(true); finishMessage(); });
  video.addEventListener("timeupdate", () => renderTrace(video.paused || video.seeking));
  video.addEventListener("seeked", () => renderTrace(true));
  video.addEventListener("error", () => {
    if (activeTask?.rollout && video.error) $("rollout-error").hidden = false;
  });
  $("copy-command").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("run-command").textContent);
      $("copy-feedback").textContent = "Command copied.";
    } catch {
      const range = document.createRange();
      range.selectNodeContents($("run-command"));
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      $("copy-feedback").textContent = "Select and copy the highlighted command.";
    }
  });

  placeTrace();
  updateTask(byId.has(query.get("task")) ? query.get("task") : "task01");
  explorer.hidden = false;
  $("copy-command").hidden = false;
})();
