(() => {
  if (typeof V2 === "undefined") return;

  const WIDGET = "named_multi_uv";
  const COLORS = [
    "#e63030", "#23aaeb", "#f5b423", "#be4bdc",
    "#2dbe69", "#f57323", "#4169e1", "#e15091",
    "#00bebe", "#a0782d", "#734bdc", "#50a550",
    "#eb5a41", "#2d7dbe", "#cd9114", "#7d7d7d",
  ];

  const namedSpec = () => (V2.tool && (V2.tool.args || []).find(
    arg => arg.widget === WIDGET
  )) || null;
  const fixedNames = () => {
    const names = namedSpec()?.fixed_names;
    return Array.isArray(names) ? names.map(name => String(name)) : [];
  };
  const isNamedTool = () => !!namedSpec();
  const html = value => escapeHtml(String(value ?? ""));
  const captureBindings = new Map();
  const captureBindingKey = (sessionId, imageId) =>
    `${String(sessionId || "").trim()}::${String(imageId || "").trim()}`;
  const currentCaptureImageId = () => {
    const image = document.getElementById("v2-img");
    const imageId = String(image?.dataset.loadedImageId || "").trim();
    const mediaKind = String(image?.dataset.loadedMediaKind || "").trim();
    if (!imageId || mediaKind !== "head") return "";
    const binding = captureBindings.get(
      captureBindingKey(V2.session, imageId)
    );
    if (!binding || binding.ok !== true) return "";
    if (String(binding.image_id || "").trim() !== imageId) return "";
    return imageId;
  };

  V2.namedTrackImageId = V2.namedTrackImageId || "";

  const baseRenderArgs = renderV2Args;
  const baseRenderPicks = renderV2MultiPicks;
  const baseToolChange = onV2ToolChange;
  const baseCollectArgs = _v2CollectArgs;
  const baseValidateBody = _v2ValidateBody;
  const baseSetImage = _v2SetImage;
  const baseApplyArtifacts = _applyV2ResponseArtifacts;
  const imageWrap = document.getElementById("v2-img-wrap");
  const baseImageClick = imageWrap ? imageWrap.onclick : null;
  let mediaLoadToken = 0;

  _applyV2ResponseArtifacts = function (payload) {
    const response = payload || {};
    const observation = response.observation || response;
    const imageId = String(
      observation.image_id || response.image_id || ""
    ).trim();
    const sessionId = String(
      observation.session_id || response.session_id || V2.session || ""
    ).trim();
    const binding = observation.track_object_distance_binding
      || response.track_object_distance_binding;
    if (imageId && sessionId && binding && typeof binding === "object") {
      captureBindings.set(
        captureBindingKey(sessionId, imageId),
        {...binding},
      );
      while (captureBindings.size > 32) {
        captureBindings.delete(captureBindings.keys().next().value);
      }
    }
    return baseApplyArtifacts(payload);
  };

  _v2SetImage = function (dataUrl, media = {}) {
    if (!dataUrl) return baseSetImage(dataUrl, media);
    const image = document.getElementById("v2-img");
    if (!image) return baseSetImage(dataUrl, media);
    const token = ++mediaLoadToken;
    const expectedSource = String(dataUrl);
    image.dataset.loadedImageId = "";
    image.dataset.loadedMediaKind = "";
    image.dataset.loadedMediaToken = String(token);
    image.addEventListener("load", () => {
      const publishLoadedImage = async () => {
        if (typeof image.decode === "function") {
          try {
            await image.decode();
          } catch (_) {
            return;
          }
        }
        if (token !== mediaLoadToken) return;
        const currentSource = String(image.currentSrc || image.src || "");
        const sourceAttribute = String(image.getAttribute("src") || "");
        if (currentSource !== expectedSource && sourceAttribute !== expectedSource) {
          return;
        }
        image.dataset.loadedImageId = String(media.imageId || "").trim();
        image.dataset.loadedMediaKind = String(media.kind || "").trim();
        renderV2MultiPicks();
      };
      void publishLoadedImage();
    }, {once: true});
    return baseSetImage(dataUrl, media);
  };

  function renderNamedRows() {
    if (!V2.multiPicks.length) V2.namedTrackImageId = "";
    const list = document.getElementById("v2-multi-list");
    const count = document.getElementById("v2-multi-count");
    const markers = document.getElementById("v2-multi-markers");
    if (count) count.textContent = `${V2.multiPicks.length} 个点`;
    if (list) {
      list.innerHTML = V2.multiPicks.length
        ? V2.multiPicks.map((point, index) => {
            const color = COLORS[index % COLORS.length];
            const fixedName = fixedNames()[index] || "";
            if (fixedName) point.name = fixedName;
            return `<div class="v2-named-row">
              <span class="v2-named-index" style="color:${color}">#${index + 1}</span>
              <input type="text" maxlength="128" value="${html(point.name)}"
                     data-v2-named-index="${index}" placeholder="名称"
                     ${fixedName ? 'readonly title="该工具的点名称固定"' : ""} />
              <span class="v2-named-coordinate">(${point.u}, ${point.v})</span>
              <button class="v2-named-remove" type="button"
                      data-v2-remove-index="${index}" title="删除这个点">×</button>
            </div>`;
          }).join("")
        : '<span style="font-size:11px;color:var(--muted)">点击下方图像添加一行</span>';
      list.querySelectorAll("[data-v2-named-index]").forEach(input => {
        input.addEventListener("input", event => {
          const index = Number(event.currentTarget.dataset.v2NamedIndex);
          if (!V2.multiPicks[index]) return;
          V2.multiPicks[index].name = event.currentTarget.value;
          renderNamedMarkers();
        });
      });
      list.querySelectorAll("[data-v2-remove-index]").forEach(button => {
        button.addEventListener("click", event => {
          const index = Number(event.currentTarget.dataset.v2RemoveIndex);
          V2.multiPicks.splice(index, 1);
          if (!V2.multiPicks.length) V2.namedTrackImageId = "";
          renderNamedRows();
        });
      });
    }
    renderNamedMarkers(markers);
    const pickText = document.getElementById("v2-pick");
    if (pickText) {
      pickText.textContent = V2.multiPicks.length
        ? `已选 ${V2.multiPicks.length} 个命名点`
        : "点击图像添加一行名称与坐标";
    }
    const imageInput = document.querySelector(
      '#v2-args input[data-arg="image_id"]'
    );
    if (imageInput) {
      imageInput.value = V2.namedTrackImageId || currentCaptureImageId();
    }
  }

  function renderNamedMarkers(markerBox) {
    const markers = markerBox || document.getElementById("v2-multi-markers");
    if (!markers) return;
    markers.innerHTML = V2.multiPicks.map((point, index) => {
      const color = COLORS[index % COLORS.length];
      const name = html(String(point.name || "").trim());
      return `<div class="v2-named-marker"
                   style="left:${point.u / 10}%;top:${point.v / 10}%;color:${color}">
        <span class="v2-named-marker-index" style="border-color:${color}">${index + 1}</span>
        <span class="v2-named-marker-label" style="${name ? "" : "display:none"}">${name}</span>
      </div>`;
    }).join("");
  }

  renderV2Args = function () {
    baseRenderArgs();
    const spec = namedSpec();
    if (!spec) return;
    const generated = document.querySelector(
      `#v2-args [data-arg="${spec.name}"]`
    );
    const cell = generated && generated.closest(".arg-cell");
    if (!cell) return;
    const required = spec.required ? "*" : "";
    cell.innerHTML = `<label style="font-size:10px;color:var(--muted)">${html(spec.name)}${required}</label>
      <div data-arg="${html(spec.name)}" data-widget="${WIDGET}" class="v2-named-picker">
        <div class="v2-named-toolbar">
          <button id="v2-named-clear" type="button" title="清空所有选点">清空</button>
          <span id="v2-multi-count" style="font-size:11px;color:var(--muted)"></span>
        </div>
        <div id="v2-multi-list" class="v2-named-list"></div>
      </div>`;
    document.getElementById("v2-named-clear")?.addEventListener("click", () => {
      V2.multiPicks = [];
      V2.namedTrackImageId = "";
      renderNamedRows();
    });
    const imageInput = document.querySelector(
      '#v2-args input[data-arg="image_id"]'
    );
    if (imageInput) {
      imageInput.value = V2.namedTrackImageId || currentCaptureImageId();
      imageInput.readOnly = true;
      imageInput.title = "自动绑定到当前显示的冻结 head capture";
    }
    renderNamedRows();
  };

  renderV2MultiPicks = function () {
    if (!isNamedTool()) return baseRenderPicks();
    renderNamedRows();
  };

  onV2ToolChange = function () {
    const wasNamed = isNamedTool();
    baseToolChange();
    if (wasNamed !== isNamedTool()) {
      V2.multiPicks = [];
      V2.namedTrackImageId = "";
      renderV2MultiPicks();
    }
  };

  _v2CollectArgs = function () {
    const body = baseCollectArgs();
    if (!isNamedTool()) return body;
    body.image_id = V2.namedTrackImageId || currentCaptureImageId();
    body.points = V2.multiPicks.map(point => ({
      name: String(point.name || "").trim(),
      u: Number(point.u),
      v: Number(point.v),
    }));
    return body;
  };

  _v2ValidateBody = function (tool, body) {
    const spec = (tool.args || []).find(arg => arg.widget === WIDGET);
    if (!spec) return baseValidateBody(tool, body);
    const currentImageId = currentCaptureImageId();
    const boundImageId = String(V2.namedTrackImageId || "").trim();
    if (!boundImageId || !body.image_id) {
      return {ok: false, msg: "请先 capture head 图，再在该图上重新选点"};
    }
    if (!currentImageId || currentImageId !== boundImageId
        || String(body.image_id) !== boundImageId) {
      return {ok: false, msg: "当前图像与选点 image_id 不一致，请在当前 capture 上重新选点"};
    }
    const minimum = Number(spec.min_points || 1);
    const maximum = Number(spec.max_points || 32);
    if (!Array.isArray(body.points) || body.points.length < minimum) {
      return {ok: false, msg: `请先在 head 图上选择至少 ${minimum} 个点`};
    }
    if (body.points.length > maximum) {
      return {ok: false, msg: `最多选择 ${maximum} 个点`};
    }
    const names = new Set();
    const requiredFixedNames = Array.isArray(spec.fixed_names)
      ? spec.fixed_names.map(name => String(name))
      : [];
    const maximumNameLength = Number(spec.name_max_length || 128);
    for (let index = 0; index < body.points.length; index += 1) {
      const point = body.points[index] || {};
      if (!Number.isFinite(point.u) || !Number.isFinite(point.v)
          || point.u < 0 || point.u > 1000 || point.v < 0 || point.v > 1000) {
        return {ok: false, msg: `第 ${index + 1} 行坐标必须在 0..1000`};
      }
      const name = String(point.name || "").trim();
      if (!name) return {ok: false, msg: `请填写第 ${index + 1} 行名称`};
      if (name.length > maximumNameLength) {
        return {ok: false, msg: `第 ${index + 1} 行名称不能超过 ${maximumNameLength} 个字符`};
      }
      if (names.has(name)) return {ok: false, msg: `名称不能重复: ${name}`};
      if (requiredFixedNames.length && name !== requiredFixedNames[index]) {
        return {ok: false, msg: `第 ${index + 1} 个点必须是 ${requiredFixedNames[index]}`};
      }
      names.add(name);
      point.name = name;
    }
    return {ok: true, body};
  };

  if (imageWrap) {
    imageWrap.onclick = event => {
      if (!isNamedTool()) {
        if (baseImageClick) baseImageClick.call(imageWrap, event);
        return;
      }
      const image = document.getElementById("v2-img");
      if (!image || !image.src) return;
      const imageId = currentCaptureImageId();
      if (!imageId) {
        document.getElementById("v2-pick").textContent =
          "当前不是可绑定的 head capture，请先运行 capture_head_camera";
        return;
      }
      if (V2.namedTrackImageId && V2.namedTrackImageId !== imageId) {
        V2.multiPicks = [];
      }
      V2.namedTrackImageId = imageId;
      const rect = image.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      if (x < 0 || y < 0 || x > rect.width || y > rect.height) return;
      const point = {
        u: Math.round(x / rect.width * 1000),
        v: Math.round(y / rect.height * 1000),
        name: fixedNames()[V2.multiPicks.length] || "",
      };
      const maximum = Number(namedSpec().max_points || 32);
      if (V2.multiPicks.length >= maximum) {
        document.getElementById("v2-pick").textContent = `最多选择 ${maximum} 个点`;
        return;
      }
      let index = V2.multiPicks.findIndex(
        existing => existing.u === point.u && existing.v === point.v
      );
      if (index < 0) {
        V2.multiPicks.push(point);
        index = V2.multiPicks.length - 1;
      }
      document.getElementById("v2-cross").style.display = "none";
      renderNamedRows();
      requestAnimationFrame(() => {
        document.querySelector(`[data-v2-named-index="${index}"]`)?.focus();
      });
    };
  }

  setTimeout(() => {
    if (isNamedTool()) renderV2Args();
  }, 0);
})();
