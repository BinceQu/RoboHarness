"use strict";

(() => {
  const tasks = JSON.parse(document.getElementById("archive-data").textContent);
  const cards = tasks.map(task => document.getElementById(`card-${task.id}`));
  const carousel = document.getElementById("task-carousel");
  const command = document.getElementById("setup-command");
  const feedback = document.getElementById("copy-feedback");
  const motion = matchMedia("(prefers-reduced-motion: reduce)");
  const rates = [1, 2, 4, 8];
  let playbackRate = 4;
  let current = -1;

  function updateRate(card) {
    const video = card.querySelector("video");
    const button = card.querySelector(".speed-toggle");
    if (!video || !button) return;
    video.defaultPlaybackRate = playbackRate;
    video.playbackRate = playbackRate;
    button.textContent = `${playbackRate}×`;
    button.dataset.rate = playbackRate;
    button.setAttribute("aria-label", `Playback speed: ${playbackRate} times. Change playback speed.`);
  }

  function shareTask() {
    const url = new URL(location.href);
    url.searchParams.delete("trace");
    if (tasks[current].id === "task01") url.searchParams.delete("task");
    else url.searchParams.set("task", tasks[current].id);
    history.replaceState(null, "", url);
  }

  function showTask(index, step = 0, focus = false) {
    index = (index + tasks.length) % tasks.length;
    if (index === current) return;
    if (current >= 0) {
      const old = cards[current];
      const video = old.querySelector("video");
      if (video) {
        video.pause();
        video.preload = "none";
        video.removeAttribute("src");
        video.load();
      }
      old.hidden = true;
    }
    current = index;
    const task = tasks[current];
    const card = cards[current];
    card.hidden = false;
    carousel.dataset.task = task.id;
    document.getElementById("task-announcement").textContent = `${current + 1} of ${tasks.length}: ${task.title}`;
    const video = card.querySelector("video");
    if (video) {
      card.querySelector(".rollout-error").hidden = true;
      video.autoplay = !motion.matches;
      video.preload = "metadata";
      video.src = video.dataset.src;
      video.load();
      updateRate(card);
      if (!motion.matches) video.play().catch(() => {});
    }
    if (step && !motion.matches) {
      card.getAnimations().forEach(animation => animation.cancel());
      card.animate([{opacity: .3, transform: `translateX(${step > 0 ? 12 : -12}px)`},
        {opacity: 1, transform: "translateX(0)"}], {duration: 200, easing: "ease-out"});
    }
    if (focus) {
      card.querySelector(`.task-arrow[data-step="${step}"]`).focus({preventScroll: true});
      carousel.scrollIntoView({block: "nearest", behavior: motion.matches ? "instant" : "smooth"});
    }
  }

  carousel.addEventListener("click", event => {
    const arrow = event.target.closest(".task-arrow");
    if (arrow) {
      const step = Number(arrow.dataset.step);
      showTask(current + step, step, true);
      shareTask();
      return;
    }
    if (event.target.closest(".speed-toggle")) {
      playbackRate = rates[(rates.indexOf(playbackRate) + 1) % rates.length];
      updateRate(cards[current]);
    }
  });
  carousel.addEventListener("keydown", event => {
    if (event.target.closest("video, .speed-toggle, a") || event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const step = event.key === "ArrowRight" ? 1 : -1;
    showTask(current + step, step, true);
    shareTask();
  });
  cards.forEach(card => {
    const video = card.querySelector("video");
    if (!video) return;
    video.addEventListener("loadedmetadata", () => updateRate(card));
    video.addEventListener("error", () => {
      if (!card.hidden && video.getAttribute("src") && video.error) card.querySelector(".rollout-error").hidden = false;
    });
  });
  motion.addEventListener("change", () => {
    const video = cards[current].querySelector("video");
    if (!video) return;
    video.autoplay = !motion.matches;
    if (motion.matches) video.pause();
  });
  document.getElementById("copy-command").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(command.textContent);
      feedback.textContent = "Commands copied.";
    } catch {
      const range = document.createRange();
      range.selectNodeContents(command);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      feedback.textContent = "Select and copy the highlighted commands.";
    }
  });
  addEventListener("popstate", () => {
    const id = new URLSearchParams(location.search).get("task") || "task01";
    const index = tasks.findIndex(task => task.id === id);
    showTask(index < 0 ? tasks.findIndex(task => task.id === "task01") : index);
  });

  const query = new URLSearchParams(location.search);
  const initial = tasks.findIndex(task => task.id === (query.get("task") || "task01"));
  showTask(initial < 0 ? tasks.findIndex(task => task.id === "task01") : initial);
  carousel.hidden = false;
  document.getElementById("copy-command").hidden = false;
  if (query.has("trace")) shareTask();
})();
