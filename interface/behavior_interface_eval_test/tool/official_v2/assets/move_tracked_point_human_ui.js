(() => {
  // Bump this whenever the coordinate grammar or row serializer changes.
  // The value is also used by the host page to invalidate an already-open
  // interface tab that still has an older widget installed.
  // Supersedes tracked_target_rows_v14_spatial_axial_order while retaining
  // the same grouped-constraint UI contract.
  const BUILD = "tracked_target_rows_v17_inequalities";
  const WIDGET = "tracked_target_points";
  const AXES = ["x", "y", "z"];
  const VARIABLE_PATTERN = /^[A-Za-z][A-Za-z0-9_]*$/;
  // The server parses the expression with the same safe affine grammar.  The
  // browser only performs a lexical check here so valid parentheses, scalar
  // coefficients, division, and spaces reach that authoritative validator.
  const EXPRESSION_PATTERN = /^(?:\?|[A-Za-z0-9_().+*/\-\s]{1,256})$/;
  const COORDINATE_SEPARATOR = /[\s,]+/;

  function runtimeReady() {
    return typeof V2 !== "undefined"
      && typeof renderV2Args === "function"
      && typeof _v2CollectArgs === "function"
      && typeof _v2ValidateBody === "function"
      && typeof escapeHtml === "function";
  }

  function install() {
    if (!runtimeReady()) return false;
    if (V2.moveTrackedPointUiBuild === BUILD) return true;

  const targetSpec = tool => ((tool || V2.tool)?.args || []).find(
    arg => arg.widget === WIDGET
  ) || null;
  const isTargetTool = () => !!targetSpec(V2.tool);
  const html = value => escapeHtml(String(value ?? ""));
    const blankRow = () => ({name: "", coordinates: "", role: "on_hand"});

    function normalizedRow(row) {
      const source = row && typeof row === "object" ? row : {};
      const legacyCoordinates = AXES.map(axis => source[axis] ?? "").join(" ").trim();
      return {
        name: String(source.name ?? ""),
        coordinates: String(source.coordinates ?? legacyCoordinates),
        role: ["on_hand", "off_hand"].includes(String(source.role || ""))
          ? String(source.role) : "on_hand",
      };
    }

  const migratedTargetRows = Array.isArray(V2.trackedTargetRows)
      ? V2.trackedTargetRows.map(normalizedRow)
    : [];

  const RELATION_TYPES = [
    "common_plane",
    "oriented_plane_normal",
    "align_vector",
  ];
  // Typed relations remain supported by the backend/API, while the human
  // widget exposes the safer named preset-group editor. Discard hidden state
  // from older tabs so an invisible legacy relation cannot affect a request.
  V2.trackedTargetRelations = [];
  const QUICK_TYPES = [
    "",
    "touch",
    "flatwise",
    "plane_parallel",
    "collinear",
    "line_vertical_to_plane",
    "vertical_to_ground",
    "faceto",
    "reverse_faceto",
  ];
  const MAX_QUICK_GROUPS = 6;
  const MAX_INEQUALITIES = 6;
  const namesText = value => Array.isArray(value)
    ? value.map(item => String(item || "").trim()).filter(Boolean).join(", ")
    : String(value || "");
  const groupNameList = raw => String(raw || "").split(/[\s,]+/)
    .map(item => item.trim()).filter(Boolean);
  const normalizedQuickGroup = group => {
    const source = group && typeof group === "object" ? group : {};
    const type = QUICK_TYPES.includes(String(source.type || ""))
      ? String(source.type || "") : "collinear";
    return {
      type: type || "collinear",
      onHandNames: namesText(source.onHandNames ?? source.on_hand_points),
      offHandNames: namesText(source.offHandNames ?? source.off_hand_points),
    };
  };
  const migratedQuickType = QUICK_TYPES.includes(String(V2.trackedQuickType || ""))
    ? String(V2.trackedQuickType || "") : "";
  const migratedQuickGroups = Array.isArray(V2.trackedQuickGroups)
    ? V2.trackedQuickGroups.slice(0, MAX_QUICK_GROUPS).map(normalizedQuickGroup)
    : (migratedQuickType ? [normalizedQuickGroup({type: migratedQuickType})] : []);
  const normalizedConstraintBlock = block => {
    const source = block && typeof block === "object" ? block : {};
    const quickType = QUICK_TYPES.includes(String(source.quickType || ""))
      ? String(source.quickType || "") : "";
    const rows = Array.isArray(source.rows)
      ? source.rows.slice(0, 6).map(normalizedRow) : [];
    return {quickType, rows: rows.length ? rows : [blankRow()]};
  };
  const legacyRowByName = new Map(
    migratedTargetRows.map(row => [String(row.name || "").trim(), row])
  );
  const migratedReferencedNames = new Set();
  const migratedBlocks = migratedQuickGroups.map(group => {
    const onNames = groupNameList(group.onHandNames);
    const offNames = groupNameList(group.offHandNames);
    const rows = [
      ...onNames.map(name => ({name, role: "on_hand"})),
      ...offNames.map(name => ({name, role: "off_hand"})),
    ].map(seed => {
      migratedReferencedNames.add(seed.name);
      const existing = legacyRowByName.get(seed.name);
      return normalizedRow(existing ? {...existing, role: seed.role} : seed);
    });
    return {
      quickType: group.type,
      rows: rows.length ? rows : [blankRow()],
    };
  });
  const unreferencedRows = migratedTargetRows.filter(
    row => !migratedReferencedNames.has(String(row.name || "").trim())
  );
  if (unreferencedRows.length) {
    migratedBlocks.push({quickType: "", rows: unreferencedRows});
  }
  V2.trackedConstraintBlocks = Array.isArray(V2.trackedConstraintBlocks)
    ? V2.trackedConstraintBlocks.slice(0, MAX_QUICK_GROUPS)
      .map(normalizedConstraintBlock)
    : migratedBlocks.slice(0, MAX_QUICK_GROUPS).map(normalizedConstraintBlock);
  if (!V2.trackedConstraintBlocks.length) {
    V2.trackedConstraintBlocks.push({
      quickType: migratedQuickType,
      rows: migratedTargetRows.length ? migratedTargetRows : [blankRow()],
    });
  }
  // Retain no hidden copy of the superseded global-row / point-name-list UI.
  V2.trackedTargetRows = [];
  V2.trackedQuickGroups = [];
  V2.trackedQuickType = "";
  const EXECUTION_MODES = ["exec", "plan"];
  V2.trackedExecutionMode = EXECUTION_MODES.includes(
    String(V2.trackedExecutionMode || "")
  ) ? String(V2.trackedExecutionMode) : "exec";
  const blankInequality = () => ({lhs: "", op: ">", rhs: "0"});
  const normalizedInequality = raw => {
    const source = raw && typeof raw === "object" ? raw : {};
    return {
      lhs: String(source.lhs ?? ""),
      op: [">", "<"].includes(String(source.op || ""))
        ? String(source.op) : ">",
      rhs: String(source.rhs ?? "0"),
    };
  };
  V2.trackedInequalities = Array.isArray(V2.trackedInequalities)
    ? V2.trackedInequalities.slice(0, MAX_INEQUALITIES)
      .map(normalizedInequality)
    : [];

  const baseRenderArgs = renderV2Args;
  const baseCollectArgs = _v2CollectArgs;
  const baseValidateBody = _v2ValidateBody;

  function ensureInitialBlock() {
    if (!V2.trackedConstraintBlocks.length) {
      V2.trackedConstraintBlocks.push({quickType: "", rows: [blankRow()]});
    }
  }

  function targetRowsMarkup(block, blockIndex) {
    return block.rows.length
      ? block.rows.map((row, rowIndex) => `
          <div class="v2-tracked-target-row">
            <span class="v2-tracked-target-index">#${rowIndex + 1}</span>
            <label>
              <span>name</span>
              <input type="text" maxlength="128" value="${html(row.name)}"
                     data-v2-block-index="${blockIndex}"
                     data-v2-target-index="${rowIndex}" data-v2-target-field="name"
                     placeholder="point_name" autocomplete="off"
                     list="v2-predefined-eef-point-names"
                     title="Tracked name or predefined EEF point" />
            </label>
            <label>
              <span>role</span>
              <select data-v2-block-index="${blockIndex}"
                      data-v2-target-index="${rowIndex}" data-v2-target-field="role">
                <option value="on_hand" ${row.role === "on_hand" ? "selected" : ""}>on hand</option>
                <option value="off_hand" ${row.role === "off_hand" ? "selected" : ""}>off hand</option>
              </select>
            </label>
            <label>
              <span>XYZ coordinates</span>
              <input type="text" value="${html(row.coordinates)}"
                     data-v2-block-index="${blockIndex}"
                     data-v2-target-index="${rowIndex}"
                     data-v2-target-field="coordinates"
                     placeholder="optional: x y z (use ? for free)"
                     autocomplete="off" spellcheck="false" />
            </label>
            <button type="button" data-v2-block-index="${blockIndex}"
                    data-v2-target-remove="${rowIndex}"
                    ${block.rows.length <= 1 ? "disabled" : ""}
                    title="Remove this target point">&times;</button>
          </div>`).join("")
      : '<span class="v2-tracked-target-empty">No target point</span>';
  }

  function renderConstraintBlocks() {
    const spec = targetSpec(V2.tool);
    const list = document.getElementById("v2-tracked-constraint-block-list");
    const count = document.getElementById("v2-tracked-constraint-block-count");
    const add = document.getElementById("v2-tracked-constraint-block-add");
    const maximum = Number(spec?.max_points || 6);
    if (count) {
      count.textContent = `${V2.trackedConstraintBlocks.length}/${MAX_QUICK_GROUPS}`;
    }
    if (add) {
      add.disabled = V2.trackedConstraintBlocks.length >= MAX_QUICK_GROUPS;
    }
    if (!list) return;
    list.innerHTML = V2.trackedConstraintBlocks.map((block, blockIndex) => `
      <section class="v2-tracked-constraint-block" data-v2-block="${blockIndex}">
        <div class="v2-tracked-constraint-block-header">
          <span class="v2-tracked-constraint-block-index">C${blockIndex + 1}</span>
          <label>quick preset
            <select data-v2-block-index="${blockIndex}" data-v2-block-field="quickType">
              <option value="" ${block.quickType === "" ? "selected" : ""}>none</option>
              <option value="touch" ${block.quickType === "touch" ? "selected" : ""}>touch (1 on + 1 off)</option>
              <option value="flatwise" ${block.quickType === "flatwise" ? "selected" : ""}>flatwise (3 on)</option>
              <option value="plane_parallel" ${block.quickType === "plane_parallel" ? "selected" : ""}>plane-parallel (3 on + 3 off)</option>
              <option value="collinear" ${block.quickType === "collinear" ? "selected" : ""}>collinear (2 on + 1/2 off)</option>
              <option value="line_vertical_to_plane" ${block.quickType === "line_vertical_to_plane" ? "selected" : ""}>line vertical to plane (2 on + 3 off)</option>
              <option value="vertical_to_ground" ${block.quickType === "vertical_to_ground" ? "selected" : ""}>vertical to ground (2/3 on)</option>
              <option value="faceto" ${block.quickType === "faceto" ? "selected" : ""}>faceto (3 on; clockwise to head camera)</option>
              <option value="reverse_faceto" ${block.quickType === "reverse_faceto" ? "selected" : ""}>reverse faceto (3 on; counterclockwise)</option>
            </select>
          </label>
          <button type="button" data-v2-block-remove="${blockIndex}"
                  ${V2.trackedConstraintBlocks.length <= 1 ? "disabled" : ""}
                  title="Remove this entire constraint block">&times;</button>
        </div>
        <div class="v2-tracked-target-toolbar"
             data-v2-complete-block-point-toolbar="${blockIndex}">
          <button type="button" data-v2-block-add-point="${blockIndex}"
                  ${block.rows.length >= maximum ? "disabled" : ""}
                  title="Add a point to constraint C${blockIndex + 1}">+</button>
          <span>${block.rows.length}/${maximum}</span>
        </div>
        <div class="v2-tracked-target-list"
             data-v2-complete-block-point-list="${blockIndex}">
          ${targetRowsMarkup(block, blockIndex)}
        </div>
      </section>`).join("");

    list.querySelectorAll("[data-v2-target-field]").forEach(input => {
      const eventName = input.tagName === "SELECT" ? "change" : "input";
      input.addEventListener(eventName, event => {
        const blockIndex = Number(event.currentTarget.dataset.v2BlockIndex);
        const rowIndex = Number(event.currentTarget.dataset.v2TargetIndex);
        const field = String(event.currentTarget.dataset.v2TargetField || "");
        const row = V2.trackedConstraintBlocks[blockIndex]?.rows[rowIndex];
        if (!row || !["name", "coordinates", "role"].includes(field)) return;
        row[field] = event.currentTarget.value;
      });
    });
    list.querySelectorAll("[data-v2-target-remove]").forEach(button => {
      button.addEventListener("click", event => {
        const blockIndex = Number(event.currentTarget.dataset.v2BlockIndex);
        const rowIndex = Number(event.currentTarget.dataset.v2TargetRemove);
        V2.trackedConstraintBlocks[blockIndex]?.rows.splice(rowIndex, 1);
        renderConstraintBlocks();
      });
    });
    list.querySelectorAll("[data-v2-block-field]").forEach(select => {
      select.addEventListener("change", event => {
        const blockIndex = Number(event.currentTarget.dataset.v2BlockIndex);
        const value = String(event.currentTarget.value || "");
        const block = V2.trackedConstraintBlocks[blockIndex];
        if (block && QUICK_TYPES.includes(value)) block.quickType = value;
      });
    });
    list.querySelectorAll("[data-v2-block-add-point]").forEach(button => {
      button.addEventListener("click", event => {
        const blockIndex = Number(event.currentTarget.dataset.v2BlockAddPoint);
        const block = V2.trackedConstraintBlocks[blockIndex];
        if (!block || block.rows.length >= maximum) return;
        block.rows.push(blankRow());
        renderConstraintBlocks();
        requestAnimationFrame(() => {
          document.querySelector(
            `[data-v2-block-index="${blockIndex}"][data-v2-target-index="${block.rows.length - 1}"]`
          )?.focus();
        });
      });
    });
    list.querySelectorAll("[data-v2-block-remove]").forEach(button => {
      button.addEventListener("click", event => {
        if (V2.trackedConstraintBlocks.length <= 1) return;
        V2.trackedConstraintBlocks.splice(
          Number(event.currentTarget.dataset.v2BlockRemove), 1
        );
        renderConstraintBlocks();
      });
    });
  }

  function renderInequalities() {
    const list = document.getElementById("v2-tracked-inequality-list");
    const count = document.getElementById("v2-tracked-inequality-count");
    const add = document.getElementById("v2-tracked-inequality-add");
    if (count) count.textContent = `${V2.trackedInequalities.length}/${MAX_INEQUALITIES}`;
    if (add) add.disabled = V2.trackedInequalities.length >= MAX_INEQUALITIES;
    if (!list) return;
    list.innerHTML = V2.trackedInequalities.map((item, index) => `
      <section class="v2-tracked-inequality-block" data-v2-inequality="${index}">
        <span class="v2-tracked-inequality-index">I${index + 1}</span>
        <label>
          <span>variable / affine function</span>
          <input type="text" value="${html(item.lhs)}"
                 data-v2-inequality-index="${index}" data-v2-inequality-field="lhs"
                 placeholder="a-b" autocomplete="off" spellcheck="false" />
        </label>
        <label>
          <span>operator</span>
          <select data-v2-inequality-index="${index}" data-v2-inequality-field="op">
            <option value="&gt;" ${item.op === ">" ? "selected" : ""}>&gt;</option>
            <option value="&lt;" ${item.op === "<" ? "selected" : ""}>&lt;</option>
          </select>
        </label>
        <label>
          <span>constant [m]</span>
          <input type="number" step="any" value="${html(item.rhs)}"
                 data-v2-inequality-index="${index}" data-v2-inequality-field="rhs"
                 placeholder="0" />
        </label>
        <button type="button" data-v2-inequality-remove="${index}"
                title="Remove this inequality">&times;</button>
      </section>`).join("");

    list.querySelectorAll("[data-v2-inequality-field]").forEach(input => {
      const eventName = input.tagName === "SELECT" ? "change" : "input";
      input.addEventListener(eventName, event => {
        const index = Number(event.currentTarget.dataset.v2InequalityIndex);
        const field = String(event.currentTarget.dataset.v2InequalityField || "");
        const item = V2.trackedInequalities[index];
        if (!item || !["lhs", "op", "rhs"].includes(field)) return;
        item[field] = event.currentTarget.value;
      });
    });
    list.querySelectorAll("[data-v2-inequality-remove]").forEach(button => {
      button.addEventListener("click", event => {
        V2.trackedInequalities.splice(
          Number(event.currentTarget.dataset.v2InequalityRemove), 1
        );
        renderInequalities();
      });
    });
  }

  renderV2Args = function () {
    baseRenderArgs();
    const spec = targetSpec(V2.tool);
    if (!spec) return;
    ensureInitialBlock();
    const generated = document.querySelector(
      `#v2-args [data-arg="${spec.name}"]`
    );
    const cell = generated && generated.closest(".arg-cell");
    if (!cell) return;
    const generatedQuick = document.querySelector(
      '#v2-args [data-arg="quick_constraint"]'
    );
    const quickCell = generatedQuick && generatedQuick.closest(".arg-cell");
    if (quickCell) quickCell.style.display = "none";
    const generatedQuickGroups = document.querySelector(
      '#v2-args [data-arg="quick_constraints"]'
    );
    const quickGroupsCell = generatedQuickGroups
      && generatedQuickGroups.closest(".arg-cell");
    if (quickGroupsCell) quickGroupsCell.style.display = "none";
    const generatedExecutionMode = document.querySelector(
      '#v2-args [data-arg="execution_mode"]'
    );
    const executionModeCell = generatedExecutionMode
      && generatedExecutionMode.closest(".arg-cell");
    if (executionModeCell) executionModeCell.style.display = "none";
    const generatedMode = document.querySelector('#v2-args [data-arg="mode"]');
    const modeCell = generatedMode && generatedMode.closest(".arg-cell");
    if (modeCell) modeCell.style.display = "none";
    const generatedRelations = document.querySelector(
      '#v2-args [data-arg="relations"]'
    );
    const relationsCell = generatedRelations && generatedRelations.closest(".arg-cell");
    if (relationsCell) relationsCell.style.display = "none";
    const generatedInequalities = document.querySelector(
      '#v2-args [data-arg="inequalities"]'
    );
    const inequalitiesCell = generatedInequalities
      && generatedInequalities.closest(".arg-cell");
    if (inequalitiesCell) inequalitiesCell.style.display = "none";
    const required = spec.required ? "*" : "";
    cell.classList.add("v2-tracked-target-cell");
    cell.innerHTML = `
      <label class="v2-tracked-target-title">${html(spec.name)}${required}</label>
      <div data-arg="${html(spec.name)}" data-widget="${WIDGET}"
           class="v2-tracked-target-picker">
        <div class="v2-tracked-target-mode-row">
          <label>run mode
            <select id="v2-tracked-execution-mode" title="Plan previews the final EEF; exec moves immediately">
              <option value="exec" ${V2.trackedExecutionMode === "exec" ? "selected" : ""}>exec</option>
              <option value="plan" ${V2.trackedExecutionMode === "plan" ? "selected" : ""}>plan</option>
            </select>
          </label>
        </div>
        <div class="v2-tracked-constraint-blocks-header">
          <span>equality constraints</span>
          <button id="v2-tracked-constraint-block-add" type="button"
                  aria-label="Add equality constraint"
                  title="Add another equality constraint block">+</button>
          <span id="v2-tracked-constraint-block-count"></span>
          <span class="v2-tracked-constraint-kind-divider"></span>
          <span>inequalities</span>
          <button id="v2-tracked-inequality-add" type="button"
                  aria-label="Add inequality constraint"
                  title="Add a strict inequality">+</button>
          <span id="v2-tracked-inequality-count"></span>
        </div>
        <div id="v2-tracked-constraint-block-list"></div>
        <div id="v2-tracked-inequality-list"></div>
        <datalist id="v2-predefined-eef-point-names">
          <option value="left_finger_tip"></option>
          <option value="right_finger_tip"></option>
          <option value="gripper_slide_center"></option>
        </datalist>
      </div>`;
    document.getElementById("v2-tracked-execution-mode")?.addEventListener("change", event => {
      const value = String(event.currentTarget.value || "");
      V2.trackedExecutionMode = EXECUTION_MODES.includes(value) ? value : "exec";
    });
    document.getElementById("v2-tracked-constraint-block-add")?.addEventListener(
      "click",
      () => {
        if (V2.trackedConstraintBlocks.length >= MAX_QUICK_GROUPS) return;
        const previous = V2.trackedConstraintBlocks[
          V2.trackedConstraintBlocks.length - 1
        ] || {quickType: "", rows: [blankRow()]};
        V2.trackedConstraintBlocks.push({
          quickType: previous.quickType,
          rows: previous.rows.map(row => ({
            ...blankRow(),
            role: row.role === "off_hand" ? "off_hand" : "on_hand",
          })),
        });
        renderConstraintBlocks();
        requestAnimationFrame(() => {
          document.querySelector(
            `[data-v2-block-index="${V2.trackedConstraintBlocks.length - 1}"][data-v2-target-index="0"]`
          )?.focus();
        });
      },
    );
    document.getElementById("v2-tracked-inequality-add")?.addEventListener(
      "click",
      () => {
        if (V2.trackedInequalities.length >= MAX_INEQUALITIES) return;
        V2.trackedInequalities.push(blankInequality());
        renderInequalities();
        requestAnimationFrame(() => {
          document.querySelector(
            `[data-v2-inequality-index="${V2.trackedInequalities.length - 1}"][data-v2-inequality-field="lhs"]`
          )?.focus();
        });
      },
    );
    renderConstraintBlocks();
    renderInequalities();
  };

  function serializeCoordinate(raw) {
    const text = String(raw ?? "").trim();
    const numeric = Number(text);
    if (text && Number.isFinite(numeric)) return numeric;
    if (text === "?") return {free: true};
    if (VARIABLE_PATTERN.test(text)) return {var: text};
    return {expr: text};
  }

  function coordinateTokens(raw) {
    const text = String(raw ?? "").trim();
    if (!text) return [];
    // Commas make the three fields unambiguous. Without commas, combine
    // arithmetic/operator tokens and balanced parentheses so expressions such
    // as ``2 * (a + b)`` still count as one coordinate.
    if (text.includes(",")) {
      return text.split(",").map(item => item.trim()).filter(Boolean);
    }
    const rawTokens = text.split(COORDINATE_SEPARATOR).filter(Boolean);
    const tokens = [];
    for (let index = 0; index < rawTokens.length;) {
      let value = rawTokens[index++];
      let depth = (value.match(/\(/g) || []).length
        - (value.match(/\)/g) || []).length;
      while (index < rawTokens.length) {
        const next = rawTokens[index];
        const valueEndsWithOperator = /[+\-*/]$/.test(value);
        const nextStartsWithOperator = /^[+\-*/]/.test(next);
        const continuation = depth > 0
          || valueEndsWithOperator
          || nextStartsWithOperator;
        if (!continuation) break;
        value += next;
        index += 1;
        depth += (next.match(/\(/g) || []).length
          - (next.match(/\)/g) || []).length;
      }
      tokens.push(value);
    }
    return tokens;
  }

  function serializePointRow(row) {
    const coordinates = coordinateTokens(row.coordinates);
    const point = {
      name: String(row.name || "").trim(),
      role: row.role === "off_hand" ? "off_hand" : "on_hand",
    };
    if (coordinates.length) {
      point.target_xyz_m = Object.fromEntries(
        AXES.map((axis, index) => [
          axis,
          serializeCoordinate(coordinates[index]),
        ]),
      );
    }
    return point;
  }

  _v2CollectArgs = function () {
    const body = baseCollectArgs();
    if (!isTargetTool()) return body;
    delete body.mode;
    delete body.quick_constraint;
    delete body.quick_constraints;
    body.execution_mode = V2.trackedExecutionMode;
    const points = [];
    const pointByName = new Map();
    const mergeErrors = [];
    const quickGroups = [];
    V2.trackedConstraintBlocks.forEach((block, blockIndex) => {
      const blockPoints = block.rows.map(serializePointRow);
      blockPoints.forEach(point => {
        if (!point.name) {
          points.push(point);
          return;
        }
        const existing = pointByName.get(point.name);
        if (!existing) {
          pointByName.set(point.name, point);
          points.push(point);
          return;
        }
        if (existing.role !== point.role) {
          mergeErrors.push(
            `Point ${point.name} has different roles in multiple constraints`
          );
        }
        const existingTarget = existing.target_xyz_m;
        const incomingTarget = point.target_xyz_m;
        if (!existingTarget && incomingTarget) {
          existing.target_xyz_m = incomingTarget;
        } else if (existingTarget && incomingTarget
                   && JSON.stringify(existingTarget) !== JSON.stringify(incomingTarget)) {
          mergeErrors.push(
            `Point ${point.name} has different XYZ coordinates in multiple constraints`
          );
        }
      });
      if (!block.quickType) return;
      const onHand = blockPoints.filter(point => point.role === "on_hand")
        .map(point => point.name);
      const offHand = blockPoints.filter(point => point.role === "off_hand")
        .map(point => point.name);
      const constraint = {
        type: block.quickType,
        on_hand_points: onHand,
        off_hand_points: offHand,
      };
      if (block.quickType === "collinear") {
        constraint.axial_point_order = blockPoints.map(point => point.name);
      }
      quickGroups.push(constraint);
    });
    V2.trackedConstraintMergeErrors = mergeErrors;
    body.points = points;
    if (quickGroups.length) {
      body.mode = "quick_constraint";
      body.quick_constraints = quickGroups;
    }
    body.relations = [];
    body.inequalities = V2.trackedInequalities.map(item => ({
      lhs: String(item.lhs || "").trim(),
      op: [">", "<"].includes(String(item.op || "")) ? String(item.op) : ">",
      rhs: Number(item.rhs),
    }));
    return body;
  };

  function coordinateIsValid(value) {
    if (typeof value === "number") return Number.isFinite(value);
    if (!value || typeof value !== "object") return false;
    const keys = Object.keys(value);
    if (keys.length !== 1) return false;
    if (keys[0] === "free") return value.free === true;
    if (keys[0] === "var") {
      return VARIABLE_PATTERN.test(String(value.var || ""));
    }
    return keys[0] === "expr"
      && EXPRESSION_PATTERN.test(String(value.expr || ""));
  }

  _v2ValidateBody = function (tool, body) {
    const spec = targetSpec(tool);
    if (!spec) return baseValidateBody(tool, body);
    const base = baseValidateBody(tool, body);
    if (!base.ok) return base;
    if (!EXECUTION_MODES.includes(String(body.execution_mode || ""))) {
      return {ok: false, msg: "Run mode must be exec or plan"};
    }
    if (!Array.isArray(V2.trackedConstraintBlocks)
        || !V2.trackedConstraintBlocks.length) {
      return {ok: false, msg: "At least one constraint block is required"};
    }
    if (V2.trackedConstraintBlocks.length > MAX_QUICK_GROUPS) {
      return {ok: false, msg: `At most ${MAX_QUICK_GROUPS} constraint blocks are allowed`};
    }
    if (Array.isArray(V2.trackedConstraintMergeErrors)
        && V2.trackedConstraintMergeErrors.length) {
      return {ok: false, msg: V2.trackedConstraintMergeErrors[0]};
    }
    for (let blockIndex = 0;
         blockIndex < V2.trackedConstraintBlocks.length;
         blockIndex += 1) {
      const block = V2.trackedConstraintBlocks[blockIndex] || {};
      if (!QUICK_TYPES.includes(String(block.quickType || ""))) {
        return {ok: false, msg: `Constraint ${blockIndex + 1}: unsupported quick preset`};
      }
      if (!Array.isArray(block.rows) || !block.rows.length) {
        return {ok: false, msg: `Constraint ${blockIndex + 1}: add at least one point`};
      }
      const localNames = new Set();
      for (let rowIndex = 0; rowIndex < block.rows.length; rowIndex += 1) {
        const row = block.rows[rowIndex] || {};
        const name = String(row.name || "").trim();
        const label = `Constraint ${blockIndex + 1}, point ${rowIndex + 1}`;
        if (!name) return {ok: false, msg: `${label}: name is required`};
        if (localNames.has(name)) {
          return {ok: false, msg: `${label}: point names must be unique inside a constraint`};
        }
        localNames.add(name);
        if (!['on_hand', 'off_hand'].includes(String(row.role || ""))) {
          return {ok: false, msg: `${label}: choose on hand or off hand`};
        }
        const coordinateCount = coordinateTokens(row.coordinates).length;
        if (coordinateCount !== 0 && coordinateCount !== AXES.length) {
          return {ok: false, msg: `${label}: XYZ coordinates require exactly 3 values`};
        }
        if (coordinateCount === AXES.length) {
          const point = serializePointRow(row);
          for (const axis of AXES) {
            if (!coordinateIsValid(point.target_xyz_m?.[axis])) {
              return {
                ok: false,
                msg: `${label} ${axis}: enter a number, variable, affine expression (for example z+0.08, 2*z, or (a+b)/2), or ?`,
              };
            }
          }
        }
      }
    }
    const hasSingleQuick = body.quick_constraint !== undefined
      && body.quick_constraint !== null;
    const hasQuickGroups = body.quick_constraints !== undefined
      && body.quick_constraints !== null;
    if (hasSingleQuick && hasQuickGroups) {
      return {ok: false, msg: "Use either one legacy preset or constraint groups, not both"};
    }
    const minimum = Number(spec.min_points || 1);
    const maximum = Number(spec.max_points || 6);
    if (!Array.isArray(body.points) || body.points.length < minimum) {
      return {ok: false, msg: `At least ${minimum} target point is required`};
    }
    if (body.points.length > maximum) {
      return {ok: false, msg: `At most ${maximum} target points are allowed`};
    }
    const names = new Set();
    for (let index = 0; index < body.points.length; index += 1) {
      const point = body.points[index] || {};
      const name = String(point.name || "").trim();
      if (!name) return {ok: false, msg: `Point ${index + 1}: name is required`};
      if (name.length > Number(spec.name_max_length || 128)) {
        return {ok: false, msg: `Point ${index + 1}: name is too long`};
      }
      if (names.has(name)) {
        return {ok: false, msg: `Tracked names must be unique: ${name}`};
      }
      names.add(name);
      if (!["on_hand", "off_hand"].includes(String(point.role || ""))) {
        return {ok: false, msg: `Point ${index + 1}: choose on hand or off hand`};
      }
      if (point.target_xyz_m !== undefined) {
        for (const axis of AXES) {
          const value = point.target_xyz_m?.[axis];
          if (!coordinateIsValid(value)) {
            return {
              ok: false,
              msg: `Point ${index + 1} ${axis}: enter a number, variable, affine expression (for example z+0.08, 2*z, or (a+b)/2), or ?`,
            };
          }
        }
      }
    }
    const onHandNames = body.points.filter(point => point.role === "on_hand")
      .map(point => point.name);
    const offHandNames = body.points.filter(point => point.role === "off_hand")
      .map(point => point.name);
    if (!onHandNames.length) {
      return {ok: false, msg: "At least one point must be on hand"};
    }
    const quickGroups = hasQuickGroups
      ? body.quick_constraints
      : (hasSingleQuick ? [body.quick_constraint] : []);
    if (!Array.isArray(quickGroups)) {
      return {ok: false, msg: "Constraint groups must be an array"};
    }
    if (quickGroups.length > MAX_QUICK_GROUPS) {
      return {ok: false, msg: `At most ${MAX_QUICK_GROUPS} constraint groups are allowed`};
    }
    const roleByName = Object.fromEntries(
      body.points.map(point => [String(point.name), String(point.role)])
    );
    for (let groupIndex = 0; groupIndex < quickGroups.length; groupIndex += 1) {
      const constraint = quickGroups[groupIndex] || {};
      const type = String(constraint.type || "");
      const onHand = Array.isArray(constraint.on_hand_points)
        ? constraint.on_hand_points : [];
      const offHand = Array.isArray(constraint.off_hand_points)
        ? constraint.off_hand_points : [];
      const counts = {
        touch: [[1], [1]],
        flatwise: [[3], [0]],
        plane_parallel: [[3], [3]],
        collinear: [[2], [1, 2]],
        line_vertical_to_plane: [[2], [3]],
        vertical_to_ground: [[2, 3], [0]],
        faceto: [[3], [0]],
        reverse_faceto: [[3], [0]],
      }[type];
      if (!counts
          || !counts[0].includes(onHand.length) || !counts[1].includes(offHand.length)
          || onHand.some(name => !names.has(String(name)))
          || offHand.some(name => !names.has(String(name)))) {
        return {ok: false, msg: `Constraint ${groupIndex + 1}: preset requires the correct on-hand/off-hand point counts`};
      }
      if (new Set(onHand).size !== onHand.length
          || new Set(offHand).size !== offHand.length
          || onHand.some(name => offHand.includes(name))) {
        return {ok: false, msg: `Constraint ${groupIndex + 1}: point names must be unique and role-disjoint`};
      }
      if (onHand.some(name => roleByName[String(name)] !== "on_hand")
          || offHand.some(name => roleByName[String(name)] !== "off_hand")) {
        return {ok: false, msg: `Constraint ${groupIndex + 1}: selected names do not match their on/off-hand roles`};
      }
      if (type === "collinear") {
        const requiredNames = [...onHand, ...offHand];
        const suppliedOrder = Array.isArray(constraint.axial_point_order)
          ? constraint.axial_point_order : [];
        if (suppliedOrder.some(name => !requiredNames.includes(name))
            || new Set(suppliedOrder).size !== requiredNames.length
            || suppliedOrder.length !== requiredNames.length) {
          return {
            ok: false,
            msg: `Constraint ${groupIndex + 1}: collinear axial order must contain every row exactly once`,
          };
        }
      }
    }
    if (body.relations !== undefined) {
      if (!Array.isArray(body.relations)) {
        return {ok: false, msg: "Typed relations must be an array"};
      }
      for (let index = 0; index < body.relations.length; index += 1) {
        const relation = body.relations[index] || {};
        const type = String(relation.type || "");
        const relationPointNames = relation.point_names;
        if (!RELATION_TYPES.includes(type)) {
          return {ok: false, msg: `Relation ${index + 1}: unsupported type`};
        }
        if (!Array.isArray(relationPointNames) || relationPointNames.length < 2
            || relationPointNames.some(name => !names.has(String(name)))) {
          return {ok: false, msg: `Relation ${index + 1}: point names are invalid`};
        }
        if (relationPointNames.some(name => offHandNames.includes(String(name)))) {
          return {ok: false, msg: `Relation ${index + 1}: typed relations use on-hand points only`};
        }
        if (type === "common_plane" && (relationPointNames.length < 3 || relationPointNames.length > 4)) {
          return {ok: false, msg: `Relation ${index + 1}: common_plane needs 3 or 4 points`};
        }
        if (type === "oriented_plane_normal" && relationPointNames.length !== 3) {
          return {ok: false, msg: `Relation ${index + 1}: oriented_plane_normal needs 3 points`};
        }
        if (type === "align_vector" && relationPointNames.length !== 2) {
          return {ok: false, msg: `Relation ${index + 1}: align_vector needs 2 points`};
        }
      }
    }
    if (!Array.isArray(body.inequalities)) {
      return {ok: false, msg: "Inequalities must be an array"};
    }
    if (body.inequalities.length > MAX_INEQUALITIES) {
      return {ok: false, msg: `At most ${MAX_INEQUALITIES} inequalities are allowed`};
    }
    for (let index = 0; index < body.inequalities.length; index += 1) {
      const inequality = body.inequalities[index] || {};
      const lhs = String(inequality.lhs || "").trim();
      if (!lhs || lhs === "?" || !EXPRESSION_PATTERN.test(lhs)
          || !/[A-Za-z][A-Za-z0-9_]*/.test(lhs)) {
        return {
          ok: false,
          msg: `Inequality ${index + 1}: left side must be a variable or affine function`,
        };
      }
      if (![">", "<"].includes(String(inequality.op || ""))) {
        return {ok: false, msg: `Inequality ${index + 1}: choose > or <`};
      }
      if (typeof inequality.rhs !== "number" || !Number.isFinite(inequality.rhs)) {
        return {ok: false, msg: `Inequality ${index + 1}: constant must be finite`};
      }
    }
    return {ok: true, body};
  };

  setTimeout(() => {
    if (isTargetTool()) renderV2Args();
  }, 0);

    V2.moveTrackedPointUiBuild = BUILD;
    window.__OFFICIAL_MOVE_TRACKED_POINT_UI_BUILD__ = BUILD;
    return true;
  }

  function tryInstall() {
    try {
      return install();
    } catch (error) {
      console.error("move_tracked_point UI initialization failed", error);
      return false;
    }
  }

  if (tryInstall()) return;
  let attempts = 0;
  const timer = window.setInterval(() => {
    attempts += 1;
    if (tryInstall() || attempts >= 200) window.clearInterval(timer);
  }, 50);
})();
