"use strict";

const els = {
  fileInput: document.getElementById("fileInput"),
  exportButton: document.getElementById("exportButton"),
  runtimeStatus: document.getElementById("runtimeStatus"),
  detectFacesButton: document.getElementById("detectFacesButton"),
  detectPlatesButton: document.getElementById("detectPlatesButton"),
  addFaceButton: document.getElementById("addFaceButton"),
  addPlateButton: document.getElementById("addPlateButton"),
  deleteButton: document.getElementById("deleteButton"),
  blurInput: document.getElementById("blurInput"),
  blurValue: document.getElementById("blurValue"),
  exportFormatInput: document.getElementById("exportFormatInput"),
  exportQualityInput: document.getElementById("exportQualityInput"),
  exportQualityValue: document.getElementById("exportQualityValue"),
  qualityRow: document.getElementById("qualityRow"),
  previewToggle: document.getElementById("previewToggle"),
  regionList: document.getElementById("regionList"),
  emptyState: document.getElementById("emptyState"),
  stage: document.getElementById("stage"),
  imageCanvas: document.getElementById("imageCanvas"),
  overlay: document.getElementById("overlay"),
};

const ctx = els.imageCanvas.getContext("2d", { willReadFrequently: true });

const state = {
  image: null,
  objectUrl: null,
  sourceName: "image",
  sourceType: "",
  regions: [],
  selectedId: null,
  blur: Number(els.blurInput.value),
  exportFormat: els.exportFormatInput.value,
  exportQuality: Number(els.exportQualityInput.value) / 100,
  preview: els.previewToggle.checked,
  drag: null,
};

const TYPE_LABELS = {
  face: "顔",
  plate: "ナンバー",
};

const TYPE_MIN_SIZE = {
  face: 36,
  plate: 30,
};

const MEDIAPIPE_WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const MEDIAPIPE_MODULE_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";
const MEDIAPIPE_FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite";
const FACE_TILE_SIZE = 1000;
const FACE_TILE_OVERLAP = 0.28;
const FACE_TILE_MAX_SIDE = 1536;
const FACE_TILE_LIMIT = 80;

let mediaPipeDetectorPromise = null;

function setStatus(message) {
  els.runtimeStatus.textContent = message;
}

function hasImage() {
  return Boolean(state.image);
}

function updateButtons() {
  const loaded = hasImage();
  els.exportButton.disabled = !loaded;
  els.detectFacesButton.disabled = !loaded;
  els.detectPlatesButton.disabled = !loaded;
  els.addFaceButton.disabled = !loaded;
  els.addPlateButton.disabled = !loaded;
  els.deleteButton.disabled = !state.selectedId;
}

function updateExportControls() {
  state.exportFormat = els.exportFormatInput.value;
  state.exportQuality = Number(els.exportQualityInput.value) / 100;
  els.exportQualityValue.textContent = els.exportQualityInput.value;
  els.qualityRow.hidden = state.exportFormat !== "image/jpeg";
}

function canUseLocalApi() {
  return ["127.0.0.1", "localhost", "::1"].includes(window.location.hostname);
}

function nextId(type) {
  return `${type}-${Date.now()}-${Math.round(Math.random() * 100000)}`;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function clampRegion(region) {
  const minSize = TYPE_MIN_SIZE[region.type] || 30;
  region.w = clamp(region.w, minSize, els.imageCanvas.width);
  region.h = clamp(region.h, minSize, els.imageCanvas.height);
  region.x = clamp(region.x, 0, els.imageCanvas.width - region.w);
  region.y = clamp(region.y, 0, els.imageCanvas.height - region.h);
  return region;
}

function createRegion(type, rect) {
  const region = clampRegion({
    id: nextId(type),
    type,
    x: Math.round(rect.x),
    y: Math.round(rect.y),
    w: Math.round(rect.w),
    h: Math.round(rect.h),
  });

  state.regions.push(region);
  state.selectedId = region.id;
  renderAll();
  return region;
}

function addManualRegion(type) {
  const width = els.imageCanvas.width;
  const height = els.imageCanvas.height;
  const regionWidth = type === "plate" ? width * 0.24 : width * 0.16;
  const regionHeight = type === "plate" ? regionWidth * 0.32 : regionWidth * 1.1;

  createRegion(type, {
    x: (width - regionWidth) / 2,
    y: (height - regionHeight) / 2,
    w: regionWidth,
    h: regionHeight,
  });
  setStatus(`${TYPE_LABELS[type]}を追加しました`);
}

function selectRegion(id) {
  state.selectedId = id;
  renderOverlay();
  renderRegionList();
  updateButtons();
}

function deleteSelectedRegion() {
  if (!state.selectedId) return;
  state.regions = state.regions.filter((region) => region.id !== state.selectedId);
  state.selectedId = state.regions[0]?.id || null;
  renderAll();
  setStatus("対象を削除しました");
}

async function loadImageFile(file) {
  if (!file) return;
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);

  const url = URL.createObjectURL(file);
  const image = new Image();
  image.decoding = "async";
  image.src = url;
  await image.decode();

  state.objectUrl = url;
  state.image = image;
  state.sourceName = file.name.replace(/\.[^.]+$/, "") || "image";
  state.sourceType = file.type || "";
  state.regions = [];
  state.selectedId = null;
  els.exportFormatInput.value = state.sourceType === "image/png" ? "image/png" : "image/jpeg";
  updateExportControls();

  els.imageCanvas.width = image.naturalWidth;
  els.imageCanvas.height = image.naturalHeight;
  els.emptyState.hidden = true;
  els.stage.hidden = false;

  renderAll();
  updateButtons();
  setStatus(`${image.naturalWidth} x ${image.naturalHeight}px`);
}

function drawImageWithBlur(targetContext, preview, backgroundColor = null) {
  if (!state.image) return;

  const width = els.imageCanvas.width;
  const height = els.imageCanvas.height;
  targetContext.clearRect(0, 0, width, height);
  targetContext.filter = "none";
  if (backgroundColor) {
    targetContext.fillStyle = backgroundColor;
    targetContext.fillRect(0, 0, width, height);
  }
  targetContext.drawImage(state.image, 0, 0, width, height);

  if (!preview) return;

  for (const region of state.regions) {
    targetContext.save();
    targetContext.beginPath();
    targetContext.rect(region.x, region.y, region.w, region.h);
    targetContext.clip();
    targetContext.filter = `blur(${state.blur}px)`;
    targetContext.drawImage(state.image, 0, 0, width, height);
    targetContext.restore();
  }

  targetContext.filter = "none";
}

function renderCanvas() {
  drawImageWithBlur(ctx, state.preview);
}

function getCanvasScale() {
  const rect = els.imageCanvas.getBoundingClientRect();
  return {
    x: rect.width / els.imageCanvas.width,
    y: rect.height / els.imageCanvas.height,
  };
}

function renderOverlay() {
  els.overlay.replaceChildren();
  if (!hasImage()) return;

  const scale = getCanvasScale();
  const fragment = document.createDocumentFragment();

  for (const region of state.regions) {
    const box = document.createElement("div");
    box.className = `mask-box ${region.type}`;
    if (region.id === state.selectedId) box.classList.add("selected");
    box.dataset.id = region.id;
    box.style.left = `${region.x * scale.x}px`;
    box.style.top = `${region.y * scale.y}px`;
    box.style.width = `${region.w * scale.x}px`;
    box.style.height = `${region.h * scale.y}px`;

    const label = document.createElement("span");
    label.className = "mask-label";
    label.textContent = TYPE_LABELS[region.type];
    box.appendChild(label);

    const handle = document.createElement("span");
    handle.className = "resize-handle";
    handle.dataset.resize = "1";
    box.appendChild(handle);

    box.addEventListener("pointerdown", startDrag);
    fragment.appendChild(box);
  }

  els.overlay.appendChild(fragment);
}

function renderRegionList() {
  els.regionList.replaceChildren();

  for (const region of state.regions) {
    const item = document.createElement("li");
    if (region.id === state.selectedId) item.classList.add("selected");
    item.dataset.id = region.id;

    const swatch = document.createElement("span");
    swatch.className = `region-color ${region.type}`;

    const name = document.createElement("strong");
    name.textContent = TYPE_LABELS[region.type];

    const size = document.createElement("span");
    size.textContent = `${Math.round(region.w)} x ${Math.round(region.h)}`;

    item.append(swatch, name, size);
    item.addEventListener("click", () => selectRegion(region.id));
    els.regionList.appendChild(item);
  }
}

function renderAll() {
  renderCanvas();
  renderOverlay();
  renderRegionList();
  updateButtons();
}

function getPointerImagePosition(event) {
  const rect = els.imageCanvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * els.imageCanvas.width,
    y: ((event.clientY - rect.top) / rect.height) * els.imageCanvas.height,
  };
}

function startDrag(event) {
  const box = event.currentTarget;
  const region = state.regions.find((item) => item.id === box.dataset.id);
  if (!region) return;

  event.preventDefault();
  selectRegion(region.id);

  const pointer = getPointerImagePosition(event);
  state.drag = {
    id: region.id,
    mode: event.target.dataset.resize ? "resize" : "move",
    startPointer: pointer,
    startRegion: { ...region },
  };

  document.addEventListener("pointermove", continueDrag);
  document.addEventListener("pointerup", stopDrag, { once: true });
}

function continueDrag(event) {
  if (!state.drag) return;
  const region = state.regions.find((item) => item.id === state.drag.id);
  if (!region) return;

  const pointer = getPointerImagePosition(event);
  const dx = pointer.x - state.drag.startPointer.x;
  const dy = pointer.y - state.drag.startPointer.y;

  if (state.drag.mode === "move") {
    region.x = state.drag.startRegion.x + dx;
    region.y = state.drag.startRegion.y + dy;
  } else {
    region.w = state.drag.startRegion.w + dx;
    region.h = state.drag.startRegion.h + dy;
  }

  clampRegion(region);
  renderCanvas();
  renderOverlay();
  renderRegionList();
}

function stopDrag() {
  document.removeEventListener("pointermove", continueDrag);
  state.drag = null;
}

function inflateRect(rect, ratio, width, height) {
  const growX = rect.w * ratio;
  const growY = rect.h * ratio;
  return {
    x: clamp(rect.x - growX / 2, 0, width),
    y: clamp(rect.y - growY / 2, 0, height),
    w: clamp(rect.w + growX, 1, width),
    h: clamp(rect.h + growY, 1, height),
  };
}

async function detectFaces() {
  setStatus("顔を検知中...");
  let faces = null;

  try {
    faces = await detectFacesWithMediaPipe();
    setStatus(`ブラウザAI顔検知: ${faces.length}件`);
  } catch (error) {
    console.warn("MediaPipe face detection failed. Falling back.", error);
  }

  if (!faces && canUseLocalApi()) {
    try {
      faces = await detectFacesWithLocalServer();
      setStatus(`ローカル顔検知: ${faces.length}件`);
    } catch (error) {
      console.warn("Local face detection failed. Falling back to browser API.", error);
    }
  }

  if (!faces && "FaceDetector" in window) {
    try {
      const detector = new window.FaceDetector({ fastMode: false, maxDetectedFaces: 80 });
      faces = await detector.detect(els.imageCanvas);
      setStatus(`ブラウザ顔検知: ${faces.length}件`);
    } catch (error) {
      console.warn("Browser face detection failed. Falling back to local server.", error);
    }
  }

  if (!faces) {
    setStatus(
      canUseLocalApi()
        ? "顔検知には server.py での起動が必要です"
        : "ブラウザ内AIモデルを読み込めませんでした。手動追加で補正してください",
    );
    return;
  }

  for (const face of faces) {
    const box = face.boundingBox || face;
    createRegion(
      "face",
      inflateRect(
        {
          x: box.x,
          y: box.y,
          w: box.width || box.w,
          h: box.height || box.h,
        },
        0.28,
        els.imageCanvas.width,
        els.imageCanvas.height,
      ),
    );
  }
}

async function loadMediaPipeFaceDetector() {
  if (!mediaPipeDetectorPromise) {
    setStatus("顔検知モデルを読み込み中...");
    mediaPipeDetectorPromise = import(MEDIAPIPE_MODULE_URL).then(
      async ({ FaceDetector, FilesetResolver }) => {
        const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
        return FaceDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MEDIAPIPE_FACE_MODEL_URL,
          },
          runningMode: "IMAGE",
          minDetectionConfidence: 0.35,
          minSuppressionThreshold: 0.3,
        });
      },
    );
  }
  return mediaPipeDetectorPromise;
}

async function detectFacesWithMediaPipe() {
  const detector = await loadMediaPipeFaceDetector();
  const tiles = buildFaceDetectionTiles(els.imageCanvas.width, els.imageCanvas.height);
  const candidates = [];

  for (const tile of tiles) {
    const canvas = document.createElement("canvas");
    const scale = Math.min(2, Math.max(1, FACE_TILE_MAX_SIDE / Math.max(tile.w, tile.h)));
    canvas.width = Math.round(tile.w * scale);
    canvas.height = Math.round(tile.h * scale);
    const tileCtx = canvas.getContext("2d");
    tileCtx.drawImage(
      state.image,
      tile.x,
      tile.y,
      tile.w,
      tile.h,
      0,
      0,
      canvas.width,
      canvas.height,
    );

    const result = detector.detect(canvas);
    for (const detection of result.detections || []) {
      const rect = normalizeMediaPipeDetection(detection);
      if (!rect) continue;
      candidates.push({
        x: tile.x + rect.x / scale,
        y: tile.y + rect.y / scale,
        w: rect.w / scale,
        h: rect.h / scale,
        score: rect.score,
        confidence: rect.score,
      });
    }
  }

  return nonMaxSuppress(candidates, 0.28)
    .sort((a, b) => b.score - a.score)
    .map(({ x, y, w, h, confidence }) => ({ x, y, w, h, confidence }));
}

function buildFaceDetectionTiles(width, height) {
  if (width <= FACE_TILE_SIZE && height <= FACE_TILE_SIZE) {
    return [{ x: 0, y: 0, w: width, h: height }];
  }

  const step = Math.max(240, Math.round(FACE_TILE_SIZE * (1 - FACE_TILE_OVERLAP)));
  const xs = axisStarts(width, FACE_TILE_SIZE, step);
  const ys = axisStarts(height, FACE_TILE_SIZE, step);
  const tiles = [];

  for (const y of ys) {
    for (const x of xs) {
      tiles.push({
        x,
        y,
        w: Math.min(FACE_TILE_SIZE, width - x),
        h: Math.min(FACE_TILE_SIZE, height - y),
      });
    }
  }

  if (tiles.length > FACE_TILE_LIMIT) {
    throw new Error(`画像が大きすぎます。顔検知タイル数: ${tiles.length}`);
  }

  return tiles;
}

function axisStarts(length, tileSize, step) {
  if (length <= tileSize) return [0];
  const starts = [];
  for (let value = 0; value <= length - tileSize; value += step) {
    starts.push(value);
  }
  const finalStart = length - tileSize;
  if (starts[starts.length - 1] !== finalStart) starts.push(finalStart);
  return starts;
}

function normalizeMediaPipeDetection(detection) {
  const box = detection.boundingBox;
  if (!box) return null;
  const x = box.originX ?? box.xMin ?? box.x ?? box.left;
  const y = box.originY ?? box.yMin ?? box.y ?? box.top;
  const w = box.width ?? (box.xMax != null && x != null ? box.xMax - x : null);
  const h = box.height ?? (box.yMax != null && y != null ? box.yMax - y : null);
  const score = detection.categories?.[0]?.score ?? detection.score?.[0] ?? 1;

  if ([x, y, w, h].some((value) => !Number.isFinite(value))) return null;
  return { x, y, w, h, score };
}

async function detectFacesWithLocalServer() {
  const blob = await new Promise((resolve) => {
    els.imageCanvas.toBlob(resolve, "image/png");
  });
  if (!blob) throw new Error("Could not encode image");

  const response = await fetch("/api/detect-faces", {
    method: "POST",
    headers: {
      "Content-Type": "image/png",
    },
    body: blob,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Local face detection failed: ${response.status}`);
  }

  const payload = await response.json();
  return payload.faces || [];
}

function detectPlateCandidates() {
  setStatus("ナンバー候補を検知中...");
  const candidates = findPlateCandidates();
  for (const candidate of candidates) {
    createRegion("plate", candidate);
  }
  setStatus(`ナンバー候補: ${candidates.length}件`);
}

function findPlateCandidates() {
  const maxWidth = 900;
  const sourceWidth = els.imageCanvas.width;
  const sourceHeight = els.imageCanvas.height;
  const scale = Math.min(1, maxWidth / sourceWidth);
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const localCtx = canvas.getContext("2d", { willReadFrequently: true });
  localCtx.drawImage(state.image, 0, 0, width, height);
  const { data } = localCtx.getImageData(0, 0, width, height);

  const gray = new Uint8ClampedArray(width * height);
  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    gray[p] = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
  }

  const edges = new Uint8Array(width * height);
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const p = y * width + x;
      const gx =
        -gray[p - width - 1] -
        gray[p - 1] * 2 -
        gray[p + width - 1] +
        gray[p - width + 1] +
        gray[p + 1] * 2 +
        gray[p + width + 1];
      const gy =
        -gray[p - width - 1] -
        gray[p - width] * 2 -
        gray[p - width + 1] +
        gray[p + width - 1] +
        gray[p + width] * 2 +
        gray[p + width + 1];
      if (Math.abs(gx) + Math.abs(gy) > 110) edges[p] = 1;
    }
  }

  const dilated = dilateEdges(edges, width, height);
  const components = connectedComponents(dilated, width, height);
  const results = [];

  for (const component of components) {
    const boxWidth = component.maxX - component.minX + 1;
    const boxHeight = component.maxY - component.minY + 1;
    const aspect = boxWidth / boxHeight;
    const areaRatio = (boxWidth * boxHeight) / (width * height);
    const density = component.count / (boxWidth * boxHeight);

    if (aspect < 1.55 || aspect > 5.6) continue;
    if (boxWidth < width * 0.055 || boxHeight < height * 0.018) continue;
    if (boxHeight > height * 0.18 || areaRatio > 0.08) continue;
    if (density < 0.12 || density > 0.92) continue;

    results.push({
      x: component.minX / scale,
      y: component.minY / scale,
      w: boxWidth / scale,
      h: boxHeight / scale,
      score: density * boxWidth,
    });
  }

  return nonMaxSuppress(results, 0.35)
    .sort((a, b) => b.score - a.score)
    .slice(0, 20)
    .map((rect) =>
      inflateRect(
        {
          x: rect.x,
          y: rect.y,
          w: rect.w,
          h: rect.h,
        },
        0.18,
        sourceWidth,
        sourceHeight,
      ),
    );
}

function dilateEdges(edges, width, height) {
  const dilated = new Uint8Array(width * height);
  const radiusX = 7;
  const radiusY = 3;

  for (let y = radiusY; y < height - radiusY; y += 1) {
    for (let x = radiusX; x < width - radiusX; x += 1) {
      const p = y * width + x;
      if (!edges[p]) continue;

      for (let yy = -radiusY; yy <= radiusY; yy += 1) {
        const row = (y + yy) * width;
        for (let xx = -radiusX; xx <= radiusX; xx += 1) {
          dilated[row + x + xx] = 1;
        }
      }
    }
  }

  return dilated;
}

function connectedComponents(binary, width, height) {
  const visited = new Uint8Array(binary.length);
  const components = [];
  const queue = [];

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const start = y * width + x;
      if (!binary[start] || visited[start]) continue;

      let minX = x;
      let maxX = x;
      let minY = y;
      let maxY = y;
      let count = 0;

      queue.length = 0;
      queue.push(start);
      visited[start] = 1;

      while (queue.length) {
        const current = queue.pop();
        const cx = current % width;
        const cy = Math.floor(current / width);
        count += 1;
        minX = Math.min(minX, cx);
        maxX = Math.max(maxX, cx);
        minY = Math.min(minY, cy);
        maxY = Math.max(maxY, cy);

        for (const next of [current - 1, current + 1, current - width, current + width]) {
          if (next < 0 || next >= binary.length) continue;
          const nx = next % width;
          if (Math.abs(nx - cx) > 1) continue;
          if (!binary[next] || visited[next]) continue;
          visited[next] = 1;
          queue.push(next);
        }
      }

      if (count > 40) components.push({ minX, maxX, minY, maxY, count });
    }
  }

  return components;
}

function nonMaxSuppress(rects, threshold) {
  const kept = [];
  const sorted = [...rects].sort((a, b) => b.score - a.score);

  for (const rect of sorted) {
    if (kept.every((item) => intersectionOverUnion(rect, item) < threshold)) {
      kept.push(rect);
    }
  }

  return kept;
}

function intersectionOverUnion(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.w * a.h + b.w * b.h - intersection;
  return union ? intersection / union : 0;
}

function exportImage() {
  if (!hasImage()) return;
  const canvas = document.createElement("canvas");
  canvas.width = els.imageCanvas.width;
  canvas.height = els.imageCanvas.height;
  const exportCtx = canvas.getContext("2d");

  drawImageWithBlur(exportCtx, true, state.exportFormat === "image/jpeg" ? "#ffffff" : null);
  const extension = state.exportFormat === "image/png" ? "png" : "jpg";
  const quality = state.exportFormat === "image/jpeg" ? state.exportQuality : undefined;

  canvas.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${state.sourceName}-masked.${extension}`;
    link.click();
    URL.revokeObjectURL(url);
    setStatus(`書き出し: ${(blob.size / 1024 / 1024).toFixed(2)} MB`);
  }, state.exportFormat, quality);
}

els.fileInput.addEventListener("change", (event) => {
  loadImageFile(event.target.files[0]).catch((error) => {
    console.error(error);
    setStatus("画像を読み込めませんでした");
  });
});

els.detectFacesButton.addEventListener("click", detectFaces);
els.detectPlatesButton.addEventListener("click", detectPlateCandidates);
els.addFaceButton.addEventListener("click", () => addManualRegion("face"));
els.addPlateButton.addEventListener("click", () => addManualRegion("plate"));
els.deleteButton.addEventListener("click", deleteSelectedRegion);
els.exportButton.addEventListener("click", exportImage);
els.exportFormatInput.addEventListener("change", updateExportControls);

els.blurInput.addEventListener("input", () => {
  state.blur = Number(els.blurInput.value);
  els.blurValue.textContent = state.blur;
  renderCanvas();
});

els.exportQualityInput.addEventListener("input", updateExportControls);

els.previewToggle.addEventListener("change", () => {
  state.preview = els.previewToggle.checked;
  renderCanvas();
});

window.addEventListener("resize", renderOverlay);
document.addEventListener("keydown", (event) => {
  if (event.key === "Delete" || event.key === "Backspace") {
    const tagName = document.activeElement?.tagName;
    if (tagName !== "INPUT" && tagName !== "TEXTAREA") deleteSelectedRegion();
  }
});

updateExportControls();
updateButtons();
