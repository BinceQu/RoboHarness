"use strict";

(() => {
  const tasks = JSON.parse(document.getElementById("archive-data").textContent);
  const byId = new Map(tasks.map(task => [task.id, task]));
  const select = document.getElementById("task-select");
  const runSelect = document.getElementById("run-task");
  const chart = document.getElementById("score-chart");
  const feedback = document.getElementById("copy-feedback");
  const copyButton = document.getElementById("copy-command");
  const code = document.getElementById("run-command");

  runSelect.replaceChildren(...Array.from(select.options, option => option.cloneNode(true)));
  runSelect.disabled = false;

  function updateTask(id) {
    const task = byId.get(id);
    if (!task) return;
    select.value = task.id;
    runSelect.value = task.id;
    document.getElementById("selected-mean").textContent = task.mean.toFixed(4);
    document.getElementById("task-budget").textContent =
      `${task.maxSteps.toLocaleString("en-US")} simulation steps · Challenge 2025 ×2`;
    document.getElementById("manifest-link").href =
      `https://github.com/cbq349/RoboHarness/blob/main/tasks/${task.id}.json`;
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
    code.textContent = `./scripts/reproduce_task.sh ${task.id} --gpu 0`;
    feedback.textContent = "";
  }

  select.addEventListener("change", () => updateTask(select.value));
  runSelect.addEventListener("change", () => updateTask(runSelect.value));
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(code.textContent);
      feedback.textContent = "Command copied.";
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      feedback.textContent = "Select and copy the highlighted command.";
    }
  });

  updateTask("task01");
  document.getElementById("archive-explorer").hidden = false;
  copyButton.hidden = false;
})();
