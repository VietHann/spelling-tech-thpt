const $ = (s) => document.querySelector(s);
const imageInput = $("#imageInput");
const runOcrBtn = $("#runOcrBtn");
const thres = $("#thres");
const thresVal = $("#thresVal");
const beam = $("#beam");
const maxNew = $("#maxNew");
const rerank = $("#rerank");

const preview = $("#preview");
const previewImg = $("#previewImg");
const ocrText = $("#ocrText");
const correctedText = $("#correctedText");
const diffBox = $("#diffBox");
const flagsBox = $("#flagsBox");
const spinner = $("#spinner");
const ocrError = $("#ocrError");

const copyBtn = $("#copyBtn");
const downloadBtn = $("#downloadBtn");

const manualInput = $("#manualInput");
const runManualBtn = $("#runManualBtn");
const manualOut = $("#manualText");
const manualError = $("#manualError");

let lastOcr = "";
let lastCorrect = "";
let lastFlags = [];

thres.addEventListener("input", () => (thresVal.textContent = thres.value));

imageInput.addEventListener("change", () => {
  if (imageInput.files && imageInput.files[0]) {
    runOcrBtn.disabled = false;
    const url = URL.createObjectURL(imageInput.files[0]);
    previewImg.src = url;
    preview.classList.remove("hidden");
  } else {
    runOcrBtn.disabled = true;
    preview.classList.add("hidden");
  }
});

runOcrBtn.addEventListener("click", async () => {
  resetOcrUI(true);
  try {
    const file = imageInput.files?.[0];
    if (!file) return;

    const fd = new FormData();
    fd.append("image", file);
    fd.append("det_thres", thres.value);
    fd.append("beam_size", beam.value);
    fd.append("max_new_tokens", maxNew.value);
    fd.append("rerank_lambda", rerank.value);

    const res = await fetch("/ocr-correct", {
      method: "POST",
      body: fd
    });

    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || `HTTP ${res.status}`);
    }

    const data = await res.json();
    lastOcr = data.ocr_text || "";
    lastCorrect = data.corrected?.final || "";
    lastFlags = data.corrected?.flagged_positions || [];

    ocrText.textContent = lastOcr;
    correctedText.textContent = lastCorrect;

    renderDiff(lastOcr, lastCorrect);
    renderFlags(lastOcr, lastFlags);
  } catch (e) {
    ocrError.textContent = "Lỗi: " + (e?.message || e);
    ocrError.classList.remove("hidden");
  } finally {
    spinner.classList.add("hidden");
  }
});

copyBtn.addEventListener("click", async () => {
  if (!lastCorrect) return;
  try {
    await navigator.clipboard.writeText(lastCorrect);
    copyBtn.textContent = "Đã copy";
    setTimeout(() => (copyBtn.textContent = "Copy"), 1500);
  } catch {}
});

downloadBtn.addEventListener("click", () => {
  if (!lastCorrect) return;
  const blob = new Blob([lastCorrect], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "corrected.txt";
  a.click();
  URL.revokeObjectURL(url);
});

runManualBtn.addEventListener("click", async () => {
  manualError.classList.add("hidden");
  manualOut.textContent = "";
  const text = manualInput.value.trim();
  if (!text) return;

  try {
    const res = await fetch("/correct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        det_thres: parseFloat(thres.value),
        beam_size: parseInt(beam.value, 10),
        max_new_tokens: parseInt(maxNew.value, 10),
        rerank_lambda: parseFloat(rerank.value)
      })
    });
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || `HTTP ${res.status}`);
    }
    const data = await res.json();
    manualOut.textContent = data.final || "";
  } catch (e) {
    manualError.textContent = "Lỗi: " + (e?.message || e);
    manualError.classList.remove("hidden");
  }
});

/* ---------- Helpers ---------- */
function resetOcrUI(spin = false) {
  ocrError.classList.add("hidden");
  ocrError.textContent = "";
  ocrText.textContent = "";
  correctedText.textContent = "";
  diffBox.innerHTML = "";
  flagsBox.innerHTML = "";
  if (spin) spinner.classList.remove("hidden");
}

// đơn giản: diff theo token (whitespace/punct), highlight thêm/bớt/giữ
function renderDiff(a, b) {
  const at = tokenize(a);
  const bt = tokenize(b);
  const lcs = longestCommonSubsequence(at, bt);

  let i = 0, j = 0, k = 0;
  const frag = document.createDocumentFragment();

  while (i < at.length || j < bt.length) {
    if (k < lcs.length && i < at.length && j < bt.length && at[i] === lcs[k] && bt[j] === lcs[k]) {
      frag.appendChild(spanKeep(at[i]));
      i++; j++; k++;
    } else if (j < bt.length && (k >= lcs.length || bt[j] !== lcs[k])) {
      frag.appendChild(spanAdd(bt[j]));
      j++;
    } else if (i < at.length && (k >= lcs.length || at[i] !== lcs[k])) {
      frag.appendChild(spanDel(at[i]));
      i++;
    } else {
      // fallback
      break;
    }
  }
  diffBox.innerHTML = "";
  diffBox.appendChild(frag);
}

function tokenize(s) {
  return (s || "")
    .replace(/([,.:;!?\"“”'‘’()\[\]{}…])/g, " $1 ")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ")
    .filter(Boolean);
}

function spanAdd(t) {
  const el = document.createElement("span");
  el.className = "add";
  el.textContent = t + " ";
  return el;
}
function spanDel(t) {
  const el = document.createElement("span");
  el.className = "del";
  el.textContent = t + " ";
  return el;
}
function spanKeep(t) {
  const el = document.createElement("span");
  el.textContent = t + " ";
  return el;
}

// LCS O(n*m) đơn giản
function longestCommonSubsequence(a, b) {
  const n = a.length, m = b.length;
  const dp = Array.from({ length: n + 1 }, () => Array(m + 1).fill(0));
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      dp[i][j] = a[i - 1] === b[j - 1]
        ? dp[i - 1][j - 1] + 1
        : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }
  const res = [];
  let i = n, j = m;
  while (i > 0 && j > 0) {
    if (a[i - 1] === b[j - 1]) {
      res.push(a[i - 1]); i--; j--;
    } else if (dp[i - 1][j] >= dp[i][j - 1]) {
      i--;
    } else {
      j--;
    }
  }
  return res.reverse();
}

// highlight vị trí nghi ngờ lỗi theo index token từ Detector
function renderFlags(text, indices) {
  const toks = tokenize(text);
  const frag = document.createDocumentFragment();
  for (let i = 0; i < toks.length; i++) {
    const span = document.createElement("span");
    span.textContent = toks[i] + " ";
    if (indices && indices.includes(i)) {
      span.className = "flag";
      span.title = "Nghi ngờ lỗi";
    }
    frag.appendChild(span);
  }
  flagsBox.innerHTML = "";
  flagsBox.appendChild(frag);
}
