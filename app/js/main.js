import { closenessLabel, deltaE2000, hexToLab, labToHex, labToRgb } from "./color.js";
import { computeUsage, DEFAULTS } from "./yarncalc.js";

// ---------- per-viewer settings (best effort; the app works without storage) ----------
const store = {
  get(key, fallback) {
    try {
      const v = localStorage.getItem(key);
      return v === null ? fallback : JSON.parse(v);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* storage unavailable: keep in memory only */
    }
  },
};

const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fmt = (n, d = 0) => n.toLocaleString("th-TH", { minimumFractionDigits: d, maximumFractionDigits: d });

const state = {
  mode: store.get("mode", "weaver"),
  warp: store.get("warp", "01"),
  weft: store.get("weft", "24"),
  target: store.get("target", "#8a4b5c"),
  useStock: store.get("useStock", false),
  stock: new Set(store.get("stock", [])),
  excludeFlagged: store.get("excludeFlagged", false),
  calc: store.get("calc", null),
};

let DATA;
const yarnByCode = new Map();
const fabricByKey = new Map();
const key = (w, f) => `${w}+${f}`;
const yarnHex = (code) => labToHex(yarnByCode.get(code).lab);
const yarnLabel = (code) => `${yarnByCode.get(code).name} (${code})`;

// ---------- weave drawing ----------
// warpHex(col) / weftHex(row) give the colour of each thread. Plain weave: warp is up when (row + col) is even.
function drawWeave(canvas, warpHex, weftHex) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  const cell = Math.max(6, Math.round(14 * dpr));
  const cols = Math.ceil(canvas.width / cell);
  const rows = Math.ceil(canvas.height / cell);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = c * cell;
      const y = r * cell;
      const warpUp = (r + c) % 2 === 0;
      ctx.fillStyle = warpUp ? warpHex(c) : weftHex(r);
      ctx.fillRect(x, y, cell, cell);
      // light shading along the thread direction so warp and weft read as threads
      const g = warpUp
        ? ctx.createLinearGradient(x, 0, x + cell, 0)
        : ctx.createLinearGradient(0, y, 0, y + cell);
      g.addColorStop(0, "rgba(0,0,0,0.28)");
      g.addColorStop(0.5, "rgba(255,255,255,0.12)");
      g.addColorStop(1, "rgba(0,0,0,0.28)");
      ctx.fillStyle = g;
      ctx.fillRect(x, y, cell, cell);
    }
  }
}

// ---------- yarn picker ----------
function pickYarn(title, current) {
  const dialog = document.getElementById("picker");
  document.getElementById("picker-title").textContent = title;
  const grid = document.getElementById("picker-grid");
  grid.innerHTML = DATA.yarns
    .map(
      (y) => `<button value="${y.code}" aria-pressed="${y.code === current}">
        <span class="chip" style="background:${labToHex(y.lab)}"></span>
        <span>${esc(y.name)}${y.flag ? ' <span class="flag-mark">*</span>' : ""}</span>
        <span class="code">รหัส ${y.code}</span>
      </button>`,
    )
    .join("");
  dialog.returnValue = "";
  dialog.showModal();
  return new Promise((resolve) => {
    dialog.addEventListener("close", () => resolve(dialog.returnValue || null), { once: true });
  });
}

function yarnButton(role, code, id) {
  const y = yarnByCode.get(code);
  return `<button type="button" class="yarn-btn" id="${id}">
    <span class="chip" style="background:${labToHex(y.lab)}"></span>
    <span class="grow"><span class="role">${role}</span><span class="name">${esc(y.name)}</span>
    <span class="role">รหัส ${code}${y.flag ? ' <span class="flag-mark">*</span>' : ""}</span></span>
  </button>`;
}

// ---------- view: look up a fabric ----------
function renderLook(root) {
  const f = fabricByKey.get(key(state.warp, state.weft));
  const rev = fabricByKey.get(key(state.weft, state.warp));
  const { inGamut } = labToRgb(f.lab);
  const lab = (v) => v.map((x) => fmt(x, 2)).join(", ");
  const loom = yarnByCode.get(state.warp).loom;

  root.innerHTML = `
    <h2>ดูสีผ้า</h2>
    <p class="muted only-weaver">เลือกสีเส้นยืนและเส้นพุ่ง แล้วดูสีผ้าที่ทอออกมา</p>
    <div class="card">
      <div class="pair-select">
        ${yarnButton("เส้นยืน", state.warp, "pick-warp")}
        <button type="button" class="swap" id="swap" aria-label="สลับเส้นยืนกับเส้นพุ่ง">⇄</button>
        ${yarnButton("เส้นพุ่ง", state.weft, "pick-weft")}
      </div>
    </div>

    <div class="card">
      <div class="result">
        <div>
          <div class="swatch-big" style="background:${labToHex(f.lab)}" role="img"
               aria-label="สีผ้าที่วัดได้"></div>
          <p class="caption">สีผ้าที่วัดได้จริง (มองจากระยะไกล)</p>
        </div>
        <div>
          <canvas class="weave" id="weave" role="img" aria-label="ภาพจำลองลายขัดเมื่อมองใกล้"></canvas>
          <p class="caption">ภาพจำลองเมื่อมองใกล้ (ลายขัด)</p>
        </div>
      </div>
      ${
        f.flag
          ? `<div class="notice"><strong>* ค่าสีนี้รอตรวจสอบ</strong> อาจไม่ตรงกับผ้าจริง
             <span class="only-research"><br>${esc(f.flagReason)}</span></div>`
          : ""
      }
      ${!inGamut ? `<p class="small muted only-research">สีนี้เกินขอบเขตสีของจอ สีบนจอจึงแสดงได้ไม่ตรง</p>` : ""}
      <p class="small muted">สีบนจอเป็นค่าประมาณ ควรเทียบกับแคตตาล็อกผ้าจริงเสมอ</p>
    </div>

    <div class="card only-research">
      <h3>ค่าสี</h3>
      <div class="table-wrap"><table class="data-table num">
        <tr><th></th><th>L*, a*, b*</th><th>sRGB</th></tr>
        <tr><td>เส้นยืน ${state.warp}</td><td>${lab(yarnByCode.get(state.warp).lab)}</td><td>${yarnHex(state.warp)}</td></tr>
        <tr><td>เส้นพุ่ง ${state.weft}</td><td>${lab(yarnByCode.get(state.weft).lab)}</td><td>${yarnHex(state.weft)}</td></tr>
        <tr><td><strong>ผ้า ${state.warp}+${state.weft}</strong></td><td>${lab(f.lab)}</td><td>${labToHex(f.lab)}</td></tr>
        <tr><td>ผ้าสลับ ${state.weft}+${state.warp}</td><td>${lab(rev.lab)}</td><td>ΔE00 = ${fmt(deltaE2000(f.lab, rev.lab), 2)}</td></tr>
      </table></div>
      <p class="small muted">ทอบนฟืมที่ ${loom} (ตามสีเส้นยืน)</p>
    </div>

    <div class="card only-research">
      <h3>ตาราง 25 × 25 (แถว = เส้นยืน, คอลัมน์ = เส้นพุ่ง)</h3>
      <div class="matrix-wrap">${matrixHtml()}</div>
      <p class="small muted"><span class="flag-mark">*</span> = ค่ารอตรวจสอบ</p>
    </div>`;

  drawWeave(root.querySelector("#weave"), () => yarnHex(state.warp), () => yarnHex(state.weft));

  root.querySelector("#pick-warp").onclick = async () => {
    const c = await pickYarn("เลือกสีเส้นยืน", state.warp);
    if (c) setPair(c, state.weft);
  };
  root.querySelector("#pick-weft").onclick = async () => {
    const c = await pickYarn("เลือกสีเส้นพุ่ง", state.weft);
    if (c) setPair(state.warp, c);
  };
  root.querySelector("#swap").onclick = () => setPair(state.weft, state.warp);
  root.querySelectorAll("table.matrix button").forEach((b) => {
    b.onclick = () => setPair(b.dataset.warp, b.dataset.weft);
  });
}

function matrixHtml() {
  const codes = DATA.yarns.map((y) => y.code);
  const head = `<tr><th></th>${codes.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  const body = codes
    .map(
      (w) =>
        `<tr><th>${w}</th>${codes
          .map((f) => {
            const fab = fabricByKey.get(key(w, f));
            const sel = w === state.warp && f === state.weft;
            return `<td><button data-warp="${w}" data-weft="${f}" aria-pressed="${sel}"
              style="background:${labToHex(fab.lab)}" title="${w}+${f}"
              aria-label="ยืน ${w} พุ่ง ${f}">${fab.flag ? "*" : ""}</button></td>`;
          })
          .join("")}</tr>`,
    )
    .join("");
  return `<table class="matrix num">${head}${body}</table>`;
}

function setPair(warp, weft) {
  state.warp = warp;
  state.weft = weft;
  store.set("warp", warp);
  store.set("weft", weft);
  if (location.hash !== "#look") location.hash = "#look";
  else render();
}

// ---------- view: find yarns for a target colour ----------
function renderFind(root) {
  root.innerHTML = `
    <h2>หาด้ายจากสีที่ต้องการ</h2>
    <p class="muted only-weaver">แตะกล่องสีด้านล่าง เลือกสีผ้าที่อยากได้ แล้วดูว่าควรใช้ด้ายคู่ไหน</p>
    <div class="card">
      <label for="target">สีที่ต้องการ</label>
      <input type="color" id="target" value="${esc(state.target)}">
      <div class="row only-research" style="margin-top:.6rem">
        <div class="grow"><label for="target-hex">รหัสสี (hex)</label>
          <input id="target-hex" value="${esc(state.target)}" inputmode="text" autocomplete="off"></div>
        <div class="grow"><label>L*, a*, b* ของเป้าหมาย</label>
          <div class="num" id="target-lab"></div></div>
      </div>
      <label style="margin-top:.8rem;display:flex;gap:.5rem;align-items:center;color:var(--text)">
        <input type="checkbox" id="use-stock" ${state.useStock ? "checked" : ""}> ใช้เฉพาะด้ายที่มีอยู่
      </label>
      <label class="only-research" style="display:flex;gap:.5rem;align-items:center;color:var(--text)">
        <input type="checkbox" id="exclude-flag" ${state.excludeFlagged ? "checked" : ""}> ไม่รวมค่าที่ติด *
      </label>
      <details id="stock-box" ${state.useStock ? "open" : ""}>
        <summary>ด้ายที่มีอยู่ (${state.stock.size} สี)</summary>
        <div class="stock-grid">${DATA.yarns
          .map(
            (y) => `<label><input type="checkbox" data-stock="${y.code}" ${state.stock.has(y.code) ? "checked" : ""}>
              <span class="chip sm" style="background:${labToHex(y.lab)}"></span>${esc(y.name)}</label>`,
          )
          .join("")}</div>
      </details>
    </div>
    <div id="find-results"></div>`;

  const update = () => {
    const targetLab = hexToLab(state.target);
    const labEl = root.querySelector("#target-lab");
    if (labEl) labEl.textContent = targetLab.map((v) => fmt(v, 1)).join(", ");
    root.querySelector("#find-results").innerHTML = findResultsHtml(targetLab);
    root.querySelectorAll("#find-results button[data-warp]").forEach((b) => {
      b.onclick = () => setPair(b.dataset.warp, b.dataset.weft);
    });
  };

  const setTarget = (hex) => {
    state.target = hex.toLowerCase();
    store.set("target", state.target);
    root.querySelector("#target").value = state.target;
    const hexInput = root.querySelector("#target-hex");
    if (hexInput && document.activeElement !== hexInput) hexInput.value = state.target;
    update();
  };
  root.querySelector("#target").oninput = (e) => setTarget(e.target.value);
  root.querySelector("#target-hex").oninput = (e) => {
    const v = e.target.value.trim();
    if (/^#?[0-9a-f]{6}$/i.test(v)) setTarget(v.startsWith("#") ? v : "#" + v);
  };
  root.querySelector("#use-stock").onchange = (e) => {
    state.useStock = e.target.checked;
    store.set("useStock", state.useStock);
    root.querySelector("#stock-box").open = state.useStock;
    update();
  };
  root.querySelector("#exclude-flag").onchange = (e) => {
    state.excludeFlagged = e.target.checked;
    store.set("excludeFlagged", state.excludeFlagged);
    update();
  };
  root.querySelectorAll("[data-stock]").forEach((cb) => {
    cb.onchange = () => {
      if (cb.checked) state.stock.add(cb.dataset.stock);
      else state.stock.delete(cb.dataset.stock);
      store.set("stock", [...state.stock]);
      root.querySelector("#stock-box summary").textContent = `ด้ายที่มีอยู่ (${state.stock.size} สี)`;
      update();
    };
  });
  update();
}

function findResultsHtml(targetLab) {
  const research = state.mode === "research";
  let candidates = DATA.fabrics;
  if (state.useStock) candidates = candidates.filter((f) => state.stock.has(f.warp) && state.stock.has(f.weft));
  if (research && state.excludeFlagged) candidates = candidates.filter((f) => !f.flag);
  if (candidates.length === 0) {
    return `<div class="notice">ยังไม่มีคู่ด้ายให้ค้น ${state.useStock ? "เลือกด้ายที่มีอยู่อย่างน้อย 1 สี" : ""}</div>`;
  }
  const ranked = candidates
    .map((f) => ({ f, de: deltaE2000(targetLab, f.lab) }))
    .sort((a, b) => a.de - b.de)
    .slice(0, research ? 20 : 6);

  const items = ranked
    .map(
      ({ f, de }) => `<li><button data-warp="${f.warp}" data-weft="${f.weft}">
        <span class="duo">
          <span class="chip" style="background:${esc(state.target)}" title="สีที่ต้องการ"></span>
          <span class="chip" style="background:${labToHex(f.lab)}" title="สีผ้า"></span>
        </span>
        <span>
          <strong>ยืน ${esc(yarnByCode.get(f.warp).name)}</strong> + <strong>พุ่ง ${esc(yarnByCode.get(f.weft).name)}</strong>
          ${f.flag ? '<span class="flag-mark">*</span>' : ""}
          <span class="small muted only-research"><br>${f.warp}+${f.weft}</span>
        </span>
        <span class="badge ${de <= 3 ? "good" : ""}">
          <span class="only-weaver">${closenessLabel(de)}</span>
          <span class="only-research num">ΔE00 ${fmt(de, 2)}</span>
        </span>
      </button></li>`,
    )
    .join("");
  const best = ranked[0].de;
  const warn =
    best > 6
      ? `<div class="notice">ไม่มีคู่ด้ายในแคตตาล็อกที่ให้สีใกล้เคียงสีนี้ ควรทอทดลองก่อน</div>`
      : "";
  return `<div class="card"><h3>คู่ด้ายที่ใกล้ที่สุด</h3>${warn}<ol class="results">${items}</ol>
    <p class="small muted">แตะคู่ด้ายเพื่อดูภาพผ้า</p></div>`;
}

// ---------- view: yarn calculator ----------
function defaultCalc() {
  const spec = DATA.yarnSpecs[0];
  return {
    ...DEFAULTS,
    warpStripes: [{ code: state.warp, count: 1 }],
    weftStripes: [{ code: state.weft, count: 1 }],
    spec: { tex: spec.tex, gramsPerSkein: spec.gramsPerSkein, pricePerSkein: spec.pricePerSkein },
  };
}

function renderCalc(root) {
  if (!state.calc) state.calc = defaultCalc();
  const c = state.calc;
  const num = (id, labelText, value, step = "any", extra = "") =>
    `<div><label for="${id}">${labelText}</label>
     <input id="${id}" type="number" inputmode="decimal" min="0" step="${step}" value="${value}" ${extra}></div>`;
  const yarnOptions = (sel) =>
    DATA.yarns.map((y) => `<option value="${y.code}" ${y.code === sel ? "selected" : ""}>${esc(y.name)} (${y.code})</option>`).join("");
  const stripeRows = (role, stripes) =>
    stripes
      .map(
        (s, i) => `<div class="stripe-row">
          <select data-role="${role}" data-i="${i}" data-k="code" aria-label="สี">${yarnOptions(s.code)}</select>
          <input type="number" min="1" step="1" inputmode="numeric" data-role="${role}" data-i="${i}" data-k="count"
                 value="${s.count}" aria-label="จำนวนเส้นต่อแถบ" ${stripes.length === 1 ? 'disabled title="ใช้สีเดียวทั้งผืน"' : ""}>
          <button type="button" class="ghost" data-remove="${role}" data-i="${i}" aria-label="ลบแถบนี้"
                  ${stripes.length === 1 ? "disabled" : ""}>✕</button>
        </div>`,
      )
      .join("");
  const specInfo = DATA.yarnSpecs[0];

  root.innerHTML = `
    <h2>คำนวณด้าย</h2>
    <p class="muted only-weaver">กรอกขนาดผ้าและเลือกสีด้าย แล้วดูว่าต้องซื้อด้ายกี่ไจ</p>
    <div class="card">
      <div class="field-grid">
        ${num("lengthM", "ความยาวผ้า (เมตร)", c.lengthM)}
        ${num("widthCm", "หน้ากว้าง (ซม.)", c.widthCm)}
      </div>
    </div>
    <div class="card">
      <h3>สีเส้นยืน</h3>
      <p class="small muted">ถ้าทำลายทาง ให้เพิ่มแถบ แล้วใส่จำนวนเส้นต่อแถบ (ลายจะวนซ้ำทั้งผืน)</p>
      ${stripeRows("warp", c.warpStripes)}
      <button type="button" data-add="warp">+ เพิ่มแถบสี</button>
      <h3 style="margin-top:1rem">สีเส้นพุ่ง</h3>
      ${stripeRows("weft", c.weftStripes)}
      <button type="button" data-add="weft">+ เพิ่มแถบสี</button>
    </div>
    <details class="card" ${state.mode === "research" ? "open" : ""}>
      <summary>ตั้งค่าเพิ่มเติม</summary>
      <div class="notice small">ความถี่ด้ายตั้งไว้เป็นค่าตัวอย่าง ควรนับจากผ้าจริง (จำนวนเส้นใน 1 ซม.)</div>
      <div class="field-grid">
        ${num("warpPerCm", "เส้นยืน / ซม.", c.warpPerCm)}
        ${num("weftPerCm", "เส้นพุ่ง / ซม.", c.weftPerCm)}
        ${num("warpCrimpPct", "การหดของเส้นยืน (%)", c.warpCrimpPct)}
        ${num("weftCrimpPct", "การหดของเส้นพุ่ง (%)", c.weftCrimpPct)}
        ${num("loomWasteM", "ด้ายเสียบนกี่ (เมตร/เส้น)", c.loomWasteM)}
        ${num("selvedgeEnds", "เส้นริมผ้า", c.selvedgeEnds, "1")}
        ${num("tex", "ความละเอียดด้าย (tex)", c.spec.tex)}
        ${num("gramsPerSkein", "น้ำหนักต่อไจ (กรัม)", c.spec.gramsPerSkein)}
        ${num("pricePerSkein", "ราคาต่อไจ (บาท)", c.spec.pricePerSkein)}
      </div>
      <p class="small muted">ค่าด้ายตั้งต้น: ${esc(specInfo.name)} ${esc(specInfo.composition)}
        ราคา ณ ${esc(specInfo.retrieved)} จาก <a href="${esc(specInfo.sourceUrl)}" target="_blank" rel="noopener">ร้านค้าออนไลน์</a>
        <span class="only-research">· ${esc(specInfo.notes)}</span></p>
      <button type="button" id="reset-calc">คืนค่าตั้งต้น</button>
    </details>
    <div id="calc-out"></div>`;

  const out = root.querySelector("#calc-out");
  const recompute = () => {
    store.set("calc", state.calc);
    out.innerHTML = calcOutputHtml(computeUsage(state.calc));
    const canvas = out.querySelector("#calc-weave");
    const seq = (stripes) => {
      const list = [];
      stripes.forEach((s) => {
        for (let k = 0; k < (stripes.length === 1 ? 1 : Math.min(s.count, 40)); k++) list.push(yarnHex(s.code));
      });
      return list;
    };
    const wa = seq(state.calc.warpStripes);
    const we = seq(state.calc.weftStripes);
    drawWeave(canvas, (col) => wa[col % wa.length], (row) => we[row % we.length]);
  };

  root.querySelectorAll("input[type=number]:not([data-role])").forEach((inp) => {
    inp.oninput = () => {
      const v = parseFloat(inp.value);
      if (!Number.isFinite(v) || v < 0) return;
      if (["tex", "gramsPerSkein", "pricePerSkein"].includes(inp.id)) state.calc.spec[inp.id] = v;
      else state.calc[inp.id] = v;
      recompute();
    };
  });
  root.querySelectorAll("[data-role]").forEach((el) => {
    el.oninput = el.onchange = () => {
      const s = state.calc[el.dataset.role + "Stripes"][+el.dataset.i];
      if (el.dataset.k === "code") s.code = el.value;
      else {
        const v = parseInt(el.value, 10);
        if (!(v >= 1)) return;
        s.count = v;
      }
      recompute();
    };
  });
  root.querySelectorAll("[data-add]").forEach((b) => {
    b.onclick = () => {
      const list = state.calc[b.dataset.add + "Stripes"];
      list.push({ code: list[list.length - 1].code, count: 10 });
      if (list.length === 2) list[0].count = Math.max(list[0].count, 10);
      renderCalc(root);
    };
  });
  root.querySelectorAll("[data-remove]").forEach((b) => {
    b.onclick = () => {
      state.calc[b.dataset.remove + "Stripes"].splice(+b.dataset.i, 1);
      renderCalc(root);
    };
  });
  root.querySelector("#reset-calc").onclick = () => {
    state.calc = defaultCalc();
    renderCalc(root);
  };
  recompute();
}

function calcOutputHtml(r) {
  const roleName = { warp: "ยืน", weft: "พุ่ง" };
  const buyRows = r.buy
    .map(
      (b) => `<tr><td><span class="chip sm" style="background:${yarnHex(b.code)};display:inline-block;vertical-align:middle"></span>
        ${esc(yarnLabel(b.code))}</td>
        <td class="r num">${fmt(b.grams, 0)}</td><td class="r num"><strong>${b.skeins}</strong></td>
        <td class="r num">${fmt(b.cost, 0)}</td></tr>`,
    )
    .join("");
  const detailRows = r.rows
    .map(
      (x) => `<tr><td>${esc(yarnLabel(x.code))}</td><td>${roleName[x.role]}</td>
        <td class="r num">${fmt(x.threads)}</td><td class="r num">${fmt(x.meters, 0)}</td>
        <td class="r num">${fmt(x.grams, 1)}</td></tr>`,
    )
    .join("");
  return `
    <div class="card">
      <h3>ต้องซื้อด้าย</h3>
      <p><span class="total-big num">${r.totals.skeins} ไจ</span>
         <span class="muted"> · ประมาณ ${fmt(r.totals.cost)} บาท</span></p>
      <div class="table-wrap"><table class="data-table">
        <tr><th>สี</th><th class="r">กรัม</th><th class="r">ไจ</th><th class="r">บาท</th></tr>${buyRows}
      </table></div>
      <p class="small muted">ปัดขึ้นเป็นไจเต็ม สีที่ใช้ทั้งยืนและพุ่งคิดรวมกัน</p>
    </div>
    <div class="card">
      <canvas class="weave" id="calc-weave" role="img" aria-label="ภาพจำลองลายผ้า"></canvas>
      <p class="caption">ภาพจำลองลายผ้า (ลายขัด)</p>
    </div>
    <div class="card only-research">
      <h3>รายละเอียด</h3>
      <p class="small num">เส้นยืน ${fmt(r.warpEnds)} เส้น × ${fmt(r.warpLenPerEnd, 2)} ม. ·
         เส้นพุ่ง ${fmt(r.weftPicks)} เส้น × ${fmt(r.weftLenPerPick, 3)} ม. · รวม ${fmt(r.totals.grams, 1)} กรัม</p>
      <div class="table-wrap"><table class="data-table">
        <tr><th>สี</th><th>ใช้เป็น</th><th class="r">จำนวนเส้น</th><th class="r">เมตร</th><th class="r">กรัม</th></tr>
        ${detailRows}
      </table></div>
    </div>`;
}

// ---------- view: about ----------
function renderAbout(root) {
  const flagged = DATA.fabrics.filter((f) => f.flag).length;
  root.innerHTML = `
    <h2>ข้อมูล</h2>
    <div class="card">
      <h3>ที่มาของข้อมูลสี</h3>
      <p>${esc(DATA.source)}</p>
      <p class="small muted">${esc(DATA.measurement)}</p>
      <p class="small">ด้าย ${DATA.yarns.length} สี · ผ้า ${DATA.fabrics.length} คู่ (ลายขัด)
        · ค่าที่รอตรวจสอบ <span class="flag-mark">*</span> ${flagged} คู่</p>
    </div>
    <div class="card">
      <h3>ข้อควรรู้</h3>
      <ul>
        <li>สีบนจอแต่ละเครื่องไม่เท่ากัน ใช้ดูเพื่อเปรียบเทียบเท่านั้น ควรเทียบกับแคตตาล็อกผ้าจริงก่อนทอ</li>
        <li>สีที่มี <span class="flag-mark">*</span> อาจบันทึกผิดหรือวัดคลาดเคลื่อน กำลังรอวัดซ้ำ</li>
        <li>ปริมาณด้ายเป็นค่าประมาณ ขึ้นกับความแน่นของผ้าและมือของช่างทอแต่ละคน</li>
        <li>แอปนี้ใช้งานได้โดยไม่ต้องต่อเน็ต หลังจากเปิดครั้งแรกแล้ว</li>
      </ul>
    </div>
    <div class="card only-research">
      <h3>ค่าที่ใช้ในการคำนวณ</h3>
      <ul class="small">
        <li>แปลง L*a*b* เป็นสีจอด้วย sRGB, white point D65 (2°)</li>
        <li>ความต่างสีใช้ CIEDE2000 (ตรวจกับข้อมูลทดสอบของ Sharma et al., 2005 ครบ 34 คู่)</li>
        <li>ชื่อคู่ X+Y = เส้นยืน X และเส้นพุ่ง Y</li>
      </ul>
    </div>
    <p class="small muted" id="offline-status"></p>`;
  const status = root.querySelector("#offline-status");
  if (NATIVE) {
    status.textContent = "แอปบนเครื่อง ใช้งานแบบออฟไลน์ได้เสมอ";
  } else if ("serviceWorker" in navigator) {
    navigator.serviceWorker.getRegistration().then((reg) => {
      status.textContent = reg?.active ? "พร้อมใช้งานแบบออฟไลน์" : "ยังไม่พร้อมใช้แบบออฟไลน์ (ต้องเปิดผ่าน https)";
    });
  }
}

// ---------- shell ----------
const NATIVE = location.protocol === "weave:";
const VIEWS = { look: renderLook, find: renderFind, calc: renderCalc, about: renderAbout };

function render() {
  const view = location.hash.slice(1) in VIEWS ? location.hash.slice(1) : "look";
  document.querySelectorAll(".tabs a").forEach((a) => {
    if (a.dataset.view === view) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  const root = document.getElementById("view");
  VIEWS[view](root);
}

function applyMode(mode) {
  state.mode = mode;
  store.set("mode", mode);
  document.documentElement.dataset.mode = mode;
  document.querySelectorAll(".mode-switch button").forEach((b) => {
    b.setAttribute("aria-checked", String(b.dataset.mode === mode));
  });
}

async function init() {
  // A shared link can choose the mode, e.g. index.html?mode=research
  const urlMode = new URLSearchParams(location.search).get("mode");
  if (urlMode === "weaver" || urlMode === "research") state.mode = urlMode;
  applyMode(state.mode);
  document.querySelectorAll(".mode-switch button").forEach((b) => {
    b.onclick = () => {
      applyMode(b.dataset.mode);
      render();
    };
  });

  try {
    DATA = await (await fetch("data/dataset.json")).json();
  } catch {
    document.getElementById("view").innerHTML =
      '<div class="notice">โหลดข้อมูลไม่สำเร็จ ลองเปิดแอปอีกครั้งเมื่อต่ออินเทอร์เน็ต</div>';
    return;
  }
  DATA.yarns.forEach((y) => yarnByCode.set(y.code, y));
  DATA.fabrics.forEach((f) => fabricByKey.set(key(f.warp, f.weft), f));
  if (!yarnByCode.has(state.warp)) state.warp = "01";
  if (!yarnByCode.has(state.weft)) state.weft = "24";

  window.addEventListener("hashchange", () => {
    render();
    window.scrollTo(0, 0);
  });
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(render, 200);
  });
  render();

  // The macOS app (macos/) bundles every file and serves it as weave://, so no service worker.
  if ("serviceWorker" in navigator && location.protocol !== "file:" && !NATIVE) {
    navigator.serviceWorker.register("sw.js").catch(() => {});
  }
}

init();
