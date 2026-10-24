/* ACCS v2 · 工作台前端（原生 JS，无构建） */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

  const state = {
    engines: [],
    models: [],
    engine: "sherpa_onnx",
    source: "system",
    running: false,
    phase: "idle",
    mode: "pipeline",
    segments: new Map(),   // id -> {el, data}
    rec: null,             // SpeechRecognition
    recStart: 0,
    lastFinalEnd: 0,
    autoscroll: true,
    lastLoud: 0,
    timer: null,
  };

  /* ------------------------------------------------------------ 小工具 */
  const toast = (() => {
    let t = null;
    return (msg, ms = 2200) => {
      const el = $("#toast");
      el.textContent = msg;
      el.classList.add("show");
      clearTimeout(t);
      t = setTimeout(() => el.classList.remove("show"), ms);
    };
  })();

  const fmtTime = (sec) => {
    sec = Math.max(0, Number(sec) || 0);
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);
    return (h ? String(h).padStart(2, "0") + ":" : "") +
      `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  async function api(path, opts = {}) {
    const r = await fetch(path, opts);
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
    return data;
  }

  /* ------------------------------------------------------------ 主题 */
  function applyTheme(mode) {
    document.documentElement.dataset.theme = mode;
    localStorage.setItem("accs-theme", mode);
  }
  $("#themeBtn").addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
  });

  /* ------------------------------------------------------------ 设置 */
  const FIELDS = ["source", "engine", "language", "threshold", "silence", "maxSeg", "overlap",
    "translate", "llmBase", "llmModel", "filePath", "device"];

  function saveSettings() {
    const cfg = { source: state.source };
    FIELDS.forEach((k) => {
      const el = $("#" + k);
      if (!el) return;
      cfg[k] = el.type === "checkbox" ? el.checked : el.value;
    });
    localStorage.setItem("accs-settings", JSON.stringify(cfg));
  }

  function loadSettings() {
    let cfg = {};
    try { cfg = JSON.parse(localStorage.getItem("accs-settings") || "{}"); } catch {}
    FIELDS.forEach((k) => {
      const el = $("#" + k);
      if (!el || cfg[k] === undefined) return;
      if (el.type === "checkbox") el.checked = cfg[k]; else el.value = cfg[k];
    });
    if (cfg.device) state.savedDevice = cfg.device;
    if (cfg.source) setSource(cfg.source);
    syncRanges();
  }

  function syncRanges() {
    $("#vThr").textContent = Number($("#threshold").value).toFixed(2);
    $("#vSil").textContent = $("#silence").value + " ms";
    $("#vMax").textContent = $("#maxSeg").value + " s";
    $("#vOvl").textContent = $("#overlap").value + " ms";
  }
  ["threshold", "silence", "maxSeg", "overlap"].forEach((id) => {
    $("#" + id).addEventListener("input", () => { syncRanges(); saveSettings(); });
  });
  FIELDS.forEach((k) => {
    const el = $("#" + k);
    if (el) el.addEventListener("change", saveSettings);
  });

  /* ------------------------------------------------------------ 来源 */
  function renderSource() {
    $$("#sourceSeg button").forEach((b) => b.classList.toggle("active", b.dataset.v === state.source));
    $("#fileField").hidden = state.source !== "file";
    const df = $("#deviceField");
    if (df) df.hidden = state.source === "file" || state.source === "browser";
    const hints = {
      system: "系统声音抓的是「电脑正在播放」的音频；听不到就换设备，或改用麦克风。",
      mic: "麦克风：直接说话即可；没反应就换一个输入设备。",
      file: "音频文件：填相对路径（如 speech_zh.wav）或绝对路径。",
      browser: "浏览器识别：用浏览器自带的 Web Speech，可能依赖厂商在线服务。",
    };
    const hint = $("#sourceHint");
    if (hint) hint.textContent = hints[state.source] || "";
  }

  async function setSource(v, opts = {}) {
    if (state.source === v) { renderSource(); return; }
    const wasRunning = state.running;
    state.source = v;
    renderSource();
    if (v === "browser" && state.engine !== "browser") {
      await selectEngine("browser", { noRestart: true });
    }
    await loadDevices();
    saveSettings();
    if (wasRunning && !opts.noRestart) await restart();
  }

  $$("#sourceSeg button").forEach((b) => b.addEventListener("click", () => setSource(b.dataset.v)));

  const deviceEl = $("#device");
  if (deviceEl) {
    deviceEl.addEventListener("change", async () => {
      state.savedDevice = deviceEl.value;
      saveSettings();
      if (state.running) await restart();
    });
  }

  async function loadDevices() {
    const sel = $("#device");
    if (!sel) return;
    const src = state.source;
    try {
      const d = await api("/api/devices");
      const list = src === "system" ? (d.loopback || []) : (d.inputs || []);
      const cur = sel.value;
      sel.innerHTML = '<option value="">默认设备</option>' +
        list.map((x) => `<option value="${esc(x.name)}">${esc(x.name)}</option>`).join("");
      const want = state.savedDevice || cur;
      if ([...sel.options].some((o) => o.value === want)) sel.value = want;
      $("#deviceHint").textContent = src === "system"
        ? (list.length ? `找到 ${list.length} 个环回设备` : "没有环回设备：确认系统正在播放声音")
        : (list.length ? `找到 ${list.length} 个输入设备` : "没有找到输入设备");
    } catch (e) {
      $("#deviceHint").textContent = "读取设备失败：" + e.message;
    }
  }

  /* ------------------------------------------------------------ 引擎 */
  const WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"];

  function renderEngineList() {
    const sel = $("#engine");
    sel.innerHTML = state.engines.map((e) => {
      const mark = e.installed ? "" : "（未安装）";
      return `<option value="${e.id}">${esc(e.label)}${mark}</option>`;
    }).join("");
    sel.value = state.engine;
    sel.addEventListener("change", () => selectEngine(sel.value));
  }

  async function selectEngine(id, opts = {}) {
    const wasRunning = state.running;
    state.engine = id;
    $("#engine").value = id;
    const meta = state.engines.find((e) => e.id === id);
    $("#enginePill").textContent = meta ? meta.label : id;
    $("#engineNote").textContent = meta
      ? (meta.installed ? "本地离线引擎，音频不出本机" : "未安装：" + (meta.note || "缺少依赖"))
      : "";
    renderEngineControls();
    if (id === "browser" && state.source !== "browser") {
      state.source = "browser";
      renderSource();
      await loadDevices();
    }
    saveSettings();
    if (wasRunning && !opts.noRestart) await restart();
  }

  function renderEngineControls() {
    const box = $("#engineControls");
    const id = state.engine;
    const sherpaModels = state.models.filter((m) => m.kind.startsWith("sherpa"));
    const voskModels = state.models.filter((m) => m.kind === "vosk");
    const ggmlModels = state.models.filter((m) => m.kind === "whisper.cpp");

    if (id === "faster_whisper" || id === "openai_whisper") {
      box.innerHTML = `
        <label class="field"><span>Whisper 模型</span>
          <select id="wmodel">${WHISPER_MODELS.map((m) => `<option ${m === "small" ? "selected" : ""}>${m}</option>`).join("")}</select>
          <small>越大越准、越慢；CPU 建议 tiny/base</small>
        </label>`;
    } else if (id === "sherpa_onnx") {
      box.innerHTML = `
        <label class="field"><span>模型目录</span>
          <select id="sherpaModel">
            <option value="">自动探测</option>
            ${sherpaModels.map((m) => `<option value="${esc(m.path)}">${esc(m.name)} · ${m.size_mb} MB</option>`).join("")}
          </select>
        </label>
        <label class="field"><span>模型类型</span>
          <select id="sherpaType">
            <option value="auto">自动</option>
            <option value="sense_voice">SenseVoice</option>
            <option value="paraformer">Paraformer</option>
            <option value="transducer">流式 Zipformer</option>
          </select>
        </label>`;
    } else if (id === "funasr") {
      box.innerHTML = `
        <label class="field"><span>FunASR 模型</span>
          <input id="funasrModel" value="paraformer-zh" />
          <small>需 pip install funasr modelscope；首次运行会下载模型</small>
        </label>`;
    } else if (id === "vosk") {
      box.innerHTML = `
        <label class="field"><span>Vosk 模型</span>
          <select id="voskModel">
            <option value="">自动探测</option>
            ${voskModels.map((m) => `<option value="${esc(m.path)}">${esc(m.name)} · ${m.size_mb} MB</option>`).join("")}
          </select>
          <small>没有的话把 vosk-model-* 放进 models/</small>
        </label>`;
    } else if (id === "whisper_cpp") {
      box.innerHTML = `
        <label class="field"><span>whisper-cli 路径</span>
          <input id="cppBin" placeholder="留空自动查找 PATH 里的 whisper-cli.exe" />
        </label>
        <label class="field"><span>GGML 模型</span>
          <select id="cppModel">
            <option value="">自动探测</option>
            ${ggmlModels.map((m) => `<option value="${esc(m.path)}">${esc(m.name)} · ${m.size_mb} MB</option>`).join("")}
          </select>
        </label>`;
    } else if (id === "browser") {
      box.innerHTML = `
        <p class="note">🔒 说明：浏览器识别用的是 Web Speech API。它<b>不一定</b>在本机运算——
        Chrome/Edge 通常会上传到厂商服务器；是否离线取决于浏览器与版本。
        如果要严格离线，请用 sherpa-onnx / faster-whisper。</p>`;
    } else {
      box.innerHTML = "";
    }
    saveSettings();
  }

  async function loadEngines() {
    try {
      const [eng, mods] = await Promise.all([api("/api/engines"), api("/api/models")]);
      state.engines = eng.engines || [];
      state.models = mods.models || [];
      renderEngineList();
      selectEngine(state.engine);
      $("#backendHint").textContent = "已连接";
    } catch (e) {
      $("#backendHint").textContent = "后端未连接";
      toast("无法连接后端：" + e.message, 4000);
    }
  }

  /* ------------------------------------------------------------ 分段渲染 */
  const segmentsEl = $("#segments");
  const emptyEl = $("#empty");

  function addSegment(d) {
    if (state.segments.has(d.id)) return;
    emptyEl.hidden = true;
    const el = document.createElement("div");
    el.className = "seg";
    el.dataset.id = d.id;
    const lang = d.language ? `<span class="lang">${esc(d.language)}</span>` : "";
    el.innerHTML = `
      <div class="meta">${lang}<span>${fmtTime(d.start)} – ${fmtTime(d.end)}</span>
        <span>${d.confidence ? "置信 " + (d.confidence * 100).toFixed(0) + "%" : ""}</span></div>
      <div class="text">${esc(d.text)}</div>`;
    segmentsEl.appendChild(el);
    state.segments.set(d.id, { el, data: d });
    applyFilter(el, d.text);
    if (state.autoscroll) segmentsEl.scrollTop = segmentsEl.scrollHeight;
    updateStats();
  }

  function setTranslation(id, text, failed) {
    const item = state.segments.get(id);
    if (!item) return;
    const el = item.el;
    el.querySelectorAll(".trans").forEach((n) => n.remove());
    const div = document.createElement("div");
    div.className = "trans" + (failed ? " error" : "");
    div.textContent = (failed ? "⚠ " : "↳ ") + text;
    el.appendChild(div);
    if (state.autoscroll) segmentsEl.scrollTop = segmentsEl.scrollHeight;
  }

  function updateStats() {
    const items = Array.from(state.segments.values());
    const dur = items.length ? items[items.length - 1].data.end : 0;
    $("#statSegs").textContent = `${items.length} 段`;
    $("#statDur").textContent = fmtTime(dur);
    $("#segPill").textContent = `${items.length} 段`;
  }

  function applyFilter(el, text) {
    const q = ($("#search").value || "").trim().toLowerCase();
    const match = !q || String(text).toLowerCase().includes(q);
    el.classList.toggle("seg-hidden", !match);
  }

  $("#search").addEventListener("input", () => {
    state.segments.forEach(({ el, data }) => applyFilter(el, data.text + " " + (data.translation || "")));
  });

  $("#autoscroll").addEventListener("change", (e) => { state.autoscroll = e.target.checked; });
  segmentsEl.addEventListener("scroll", () => {
    const nearBottom = segmentsEl.scrollHeight - segmentsEl.scrollTop - segmentsEl.clientHeight < 40;
    if (!nearBottom) state.autoscroll = false;
    $("#autoscroll").checked = state.autoscroll;
  });
  $("#showTrans").addEventListener("change", (e) => {
    segmentsEl.querySelectorAll(".trans").forEach((n) => { n.style.display = e.target.checked ? "" : "none"; });
  });
  $("#fontSize").addEventListener("change", (e) => {
    segmentsEl.style.fontSize = e.target.value + "px";
    segmentsEl.querySelectorAll(".seg .text").forEach((n) => { n.style.fontSize = e.target.value + "px"; });
  });
  $("#clearBtn").addEventListener("click", () => {
    state.segments.clear();
    segmentsEl.innerHTML = "";
    emptyEl.hidden = false;
    $("#chapters").innerHTML = '<p class="note">按停顿自动分章</p>';
    updateStats();
  });

  /* ------------------------------------------------------------ 状态 */
  function setPhase(phase, running) {
    state.phase = phase;
    state.running = running;
    if (running && !state.lastLoud) state.lastLoud = Date.now();
    $("#phaseDot").className = "dot " + (phase === "loading" ? "loading" : phase === "stopping" ? "stopping" : (running ? "running" : ""));
    $("#phaseText").textContent =
      phase === "loading" ? "加载模型…" :
      phase === "stopping" ? "停止中…" :
      running ? (state.mode === "browser" ? "浏览器识别中" : "识别中") : "未启动";
    $("#startBtn").disabled = running;
    $("#stopBtn").disabled = !running;
    if (!running) $("#alert").hidden = true;
    $("#footStatus").textContent = $("#phaseText").textContent;
  }

  function setStats(stats) {
    const on = stats && stats.active;
    $("#vadState").classList.toggle("on", !!on);
    $("#vadState em").textContent = on ? "检测到语音" : "等待语音";
    if (!state.running) $("#alert").hidden = true;
  }

  function updateLevel(stats) {
    const bar = $("#levelBar");
    if (!bar) return;
    const level = stats && typeof stats.level === "number" ? stats.level : 0;
    // RMS 0.08 左右已经是很响的语音，映射到 0–100%
    const pct = Math.max(0, Math.min(100, Math.round(level * 1200)));
    bar.style.width = pct + "%";
    const now = Date.now();
    if (level > 0.008) state.lastLoud = now;

    const alert = $("#alert");
    if (!state.running || state.mode === "browser") { alert.hidden = true; return; }
    const silentMs = now - (state.lastLoud || 0);
    if (silentMs > 5000) {
      alert.hidden = false;
      alert.textContent = state.source === "system"
        ? "已经 5 秒没拾到声音：确认电脑正在播放、音量不为 0，或在上面换一个「音频设备」。"
        : "已经 5 秒没拾到声音：确认麦克风没被静音，或在上面换一个「音频设备」。";
    } else {
      alert.hidden = true;
    }
  }

  /* ------------------------------------------------------------ 启动 / 停止 */
  function collectStartBody() {
    const body = {
      source: state.source,
      file: $("#filePath").value.trim(),
      engine: state.engine,
      language: $("#language").value,
      translate: $("#translate").checked,
      llm_base_url: $("#llmBase").value.trim(),
      llm_model: $("#llmModel").value.trim(),
      threshold: Number($("#threshold").value),
      silence_ms: Number($("#silence").value),
      max_segment_ms: Number($("#maxSeg").value) * 1000,
      overlap_ms: Number($("#overlap").value),
      audio_device: $("#device") ? $("#device").value : "",
    };
    const wmodel = $("#wmodel");
    if (wmodel) body.model = wmodel.value;
    const sherpa = $("#sherpaModel");
    if (sherpa) body.sherpa_onnx_model = sherpa.value;
    const stype = $("#sherpaType");
    if (stype) body.sherpa_onnx_type = stype.value;
    const fm = $("#funasrModel");
    if (fm) body.funasr_model = fm.value;
    const vm = $("#voskModel");
    if (vm) body.vosk_model = vm.value;
    const cb = $("#cppBin");
    if (cb) body.whisper_cpp_bin = cb.value;
    const cm = $("#cppModel");
    if (cm) body.whisper_cpp_model = cm.value;
    return body;
  }

  async function start() {
    if (DEMO) { toast("演示模式：这是示例数据，真实识别请去掉网址里的 ?demo=1"); return; }
    if (state.running) { toast("已经在运行，先点「停止」或直接切换配置"); return; }
    if (state.source === "file" && !$("#filePath").value.trim()) { toast("请填写音频文件路径"); return; }
    if (state.source === "browser" || state.engine === "browser") {
      try {
        await startBrowser();
      } catch (e) { toast(e.message, 4000); }
      return;
    }
    try {
      setPhase("loading", true);
      $("#alert").hidden = true;
      await api("/api/start", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectStartBody()),
      });
      toast("已开始");
    } catch (e) {
      setPhase("idle", false);
      toast("启动失败：" + e.message, 5000);
    }
  }

  /** 停止 → 等一会儿 → 用当前配置重新开始（切换音源/引擎时自动调用） */
  async function restart() {
    if (DEMO) return;
    try { await stop(true); } catch (e) { /* ignore */ }
    await new Promise((r) => setTimeout(r, 400));
    await start();
  }

  async function stop(silent = false) {
    if (state.rec) { try { state.rec.stop(); } catch {} state.rec = null; }
    setPhase("stopping", true);
    try {
      await Promise.race([
        api("/api/stop", { method: "POST" }),
        new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), 12000)),
      ]);
    } catch (e) {
      if (!silent) toast("停止超时，已重置界面状态", 4000);
    } finally {
      setPhase("idle", false);
      setStats(null);
      $("#alert").hidden = true;
      if (!silent) toast("已停止");
    }
  }

  $("#startBtn").addEventListener("click", start);
  $("#stopBtn").addEventListener("click", stop);

  /* ------------------------------------------------------------ 浏览器识别 */
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;

  async function startBrowser() {
    if (!SpeechRec) throw new Error("当前浏览器不支持 Web Speech API（建议 Chrome / Edge）");
    await api("/api/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ engine: "browser", translate: $("#translate").checked,
        llm_base_url: $("#llmBase").value.trim(), llm_model: $("#llmModel").value.trim() }),
    });
    const rec = new SpeechRec();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = ({ zh: "zh-CN", en: "en-US", ja: "ja-JP", ko: "ko-KR" })[$("#language").value] || "zh-CN";
    state.recStart = performance.now();
    state.lastFinalEnd = 0;

    rec.onresult = async (ev) => {
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const r = ev.results[i];
        if (!r.isFinal) continue;
        const text = (r[0] && r[0].transcript || "").trim();
        if (!text) continue;
        const now = (performance.now() - state.recStart) / 1000;
        const start = state.lastFinalEnd;
        state.lastFinalEnd = now;
        try {
          await api("/api/browser/segment", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, start, end: now, language: rec.lang }),
          });
        } catch (e) { toast("上传识别结果失败：" + e.message); }
      }
    };
    rec.onerror = (ev) => {
      if (ev.error === "not-allowed") toast("麦克风权限被拒绝", 5000);
      else if (ev.error !== "no-speech") toast("识别出错：" + ev.error, 4000);
    };
    rec.onend = () => {
      if (state.running) { try { rec.start(); } catch {} }
    };
    rec.start();
    state.rec = rec;
    state.mode = "browser";
    setPhase("running", true);
    toast("浏览器识别已启动（麦克风）");
  }

  /* ------------------------------------------------------------ SSE */
  function connectSSE() {
    const es = new EventSource("/api/stream");
    es.onmessage = (ev) => {
      let d;
      try { d = JSON.parse(ev.data); } catch { return; }
      if (d.type === "reset") {
        (d.events || []).forEach((e) => {
          if (e.type === "segment") addSegment(e);
          else if (e.type === "translation") setTranslation(e.id, e.translation, String(e.translation).startsWith("⚠"));
        });
      } else if (d.type === "segment") {
        addSegment(d);
      } else if (d.type === "translation") {
        setTranslation(d.id, d.translation, String(d.translation).startsWith("⚠"));
        const item = state.segments.get(d.id);
        if (item) item.data.translation = d.translation;
      } else if (d.type === "status") {
        if (d.mode) state.mode = d.mode;
        setPhase(d.phase, d.running);
      } else if (d.type === "stats") {
        setStats(d.stats);
        updateLevel(d.stats);
      } else if (d.type === "info") {
        $("#footStatus").textContent = d.message;
      } else if (d.type === "error") {
        toast(d.message, 6000);
        $("#footStatus").textContent = d.message;
        setPhase("idle", false);
      } else if (d.type === "analysis_status") {
        $("#analysis").innerHTML = '<p class="note">🔄 正在分析（本机 LLM）…</p>';
      } else if (d.type === "analysis") {
        renderAnalysis(d.data);
      }
    };
    es.onerror = () => { /* EventSource 会自动重连 */ };
  }

  /* ------------------------------------------------------------ 导出 */
  $$(".exp").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (!state.segments.size) { toast("还没有内容可导出"); return; }
      const fmt = btn.dataset.fmt;
      window.location.href = "/api/export?format=" + encodeURIComponent(fmt);
      toast("正在生成 " + fmt.toUpperCase() + " …");
    });
  });

  /* ------------------------------------------------------------ 章节 */
  $("#chapBtn").addEventListener("click", async () => {
    if (!state.segments.size) { toast("还没有内容"); return; }
    try {
      const { chapters } = await api("/api/chapters");
      const box = $("#chapters");
      if (!chapters || !chapters.length) { box.innerHTML = '<p class="note">没有可分的章节</p>'; return; }
      box.innerHTML = chapters.map((c) => `
        <button class="chap" data-start="${c.start}">
          <b>${esc(c.title)}</b>
          <small>${fmtTime(c.start)} · ${Math.round(c.duration)}s</small>
        </button>`).join("");
      box.querySelectorAll(".chap").forEach((b) => {
        b.addEventListener("click", () => {
          const start = Number(b.dataset.start);
          let best = null;
          state.segments.forEach(({ el, data }) => {
            if (!best || Math.abs(data.start - start) < Math.abs(best.data.start - start)) best = { el, data };
          });
          if (best) {
            best.el.scrollIntoView({ behavior: "smooth", block: "center" });
            best.el.classList.add("hit");
            setTimeout(() => best.el.classList.remove("hit"), 1600);
          }
        });
      });
    } catch (e) { toast("生成章节失败：" + e.message); }
  });

  /* ------------------------------------------------------------ 分析 */
  $("#analyzeBtn").addEventListener("click", async () => {
    try {
      await api("/api/analyze", { method: "POST" });
      $("#analysis").innerHTML = '<p class="note">🔄 正在分析（本机 LLM）…</p>';
    } catch (e) { toast("分析失败：" + e.message, 5000); }
  });

  function renderAnalysis(data) {
    if (!data) return;
    if (data.error) { $("#analysis").innerHTML = `<p class="note" style="color:var(--bad)">${esc(data.error)}</p>`; return; }
    const kp = (data.knowledge_points || []).map((k) => `<li>${esc(k)}</li>`).join("");
    $("#analysis").innerHTML = `
      <h4>✓ 纠正后的内容</h4>
      <div style="white-space:pre-wrap">${esc(data.corrected || "（无）")}</div>
      <h4>✦ 知识要点</h4>
      <ul>${kp || "<li style='color:var(--faint)'>未提取到</li>"}</ul>`;
  }

  /* ------------------------------------------------------------ 状态轮询 */
  async function poll() {
    try {
      const d = await api("/api/status");
      if (!state.running && d.running) setPhase(d.phase, true);
      if (state.running && !d.running) setPhase(d.phase, false);
      if (d.stats) { setStats(d.stats); updateLevel(d.stats); }
      $("#footInfo").textContent = `ACCS v2.0 · ${d.llm_online ? "LLM 在线" : "LLM 离线"} · ${d.mode}`;
    } catch { /* 后端可能未启动 */ }
  }

  /* ------------------------------------------------------------ 演示数据（截图/离线预览用） */
  const DEMO = new URLSearchParams(location.search).has("demo");

  function seedDemo() {
    const rows = [
      { id: 1, start: 0.0, end: 4.2, text: "上一节课我们讲了光栅衍射的基本条件，今天从光栅方程开始。", language: "zh", needs_translation: false },
      { id: 2, start: 4.6, end: 10.8, text: "The grating equation is d sine theta equals k lambda, where d is the grating constant.", language: "en", needs_translation: true },
      { id: 3, start: 11.2, end: 16.5, text: "注意这里的 k 是级次，垂直入射时才有这个简化形式。", language: "zh", needs_translation: false },
      { id: 4, start: 17.0, end: 24.3, text: "如果入射角不是零，就要用更一般的形式，斜入射会改变光程差。", language: "zh", needs_translation: false },
      { id: 5, start: 25.1, end: 31.6, text: "Let's compute the first order diffraction angle for a 600 line per millimeter grating.", language: "en", needs_translation: true },
      { id: 6, start: 32.0, end: 39.4, text: "算出角度之后，可以反过来验证光栅常数，这也是实验报告里要写的不确定度来源。", language: "zh", needs_translation: false },
    ];
    rows.forEach((r) => addSegment(r));
    setTimeout(() => {
      setTranslation(2, "光栅方程是 d·sinθ = kλ，其中 d 是光栅常数。", false);
      setTranslation(5, "我们来算一下每毫米 600 线的光栅，一级衍射角是多少。", false);
    }, 300);
    state.mode = "pipeline";
    setPhase("running", true);
    setStats({ active: true });
    $("#statEngine").textContent = "sherpa-onnx";
  }

  /* ------------------------------------------------------------ 初始化 */
  function init() {
    const forced = new URLSearchParams(location.search).get("theme");
    applyTheme(forced === "dark" || forced === "light" ? forced :
      (localStorage.getItem("accs-theme") ||
        (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark")));
    loadSettings();
    syncRanges();
    loadEngines();
    renderSource();
    loadDevices();
    if (DEMO) {
      seedDemo();
      $("#backendHint").textContent = "演示模式";
    } else {
      connectSSE();
      setPhase("idle", false);
      state.timer = setInterval(poll, 3000);
      poll();
    }
  }

  init();
})();
