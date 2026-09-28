window.FieldNotes = (() => {
  const PYODIDE_VER = "0.26.4";
  const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VER}/full/`;

  marked.setOptions({ breaks: true, gfm: true });

  const esc = (s) =>
    String(s ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );

  function md(src) {
    let text = String(src ?? "");
    const code = [],
      math = [];

    text = text.replace(/```[\s\S]*?```|`[^`\n]*`/g, (m) => {
      code.push(m);
      return `CODESTASH${code.length - 1}END`;
    });

    text = text.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => {
      math.push({ tex, display: true });
      return `MATHSTASH${math.length - 1}END`;
    });
    text = text.replace(/\$([^\n$]+?)\$/g, (_, tex) => {
      math.push({ tex, display: false });
      return `MATHSTASH${math.length - 1}END`;
    });

    text = text.replace(/CODESTASH(\d+)END/g, (_, i) => code[+i]);

    let html = DOMPurify.sanitize(marked.parse(text));
    return html.replace(/MATHSTASH(\d+)END/g, (_, i) => tex(math[+i]));
  }
  
  function tex({ tex: src, display }) {
    const raw = display ? `$$${src}$$` : `$${src}$`;
    if (typeof katex === "undefined") return esc(raw);
    try {
      return katex.renderToString(String(src).trim(), {
        displayMode: display,
        throwOnError: false,
        strict: false,
        output: "htmlAndMathml",
      });
    } catch {
      return `<code class="math-error">${esc(raw)}</code>`;
    }
  }

  const prettyDate = (iso) => {
    const d = new Date(iso + "T00:00:00");
    return isNaN(d)
      ? iso
      : d.toLocaleDateString("en-GB", {
          day: "numeric",
          month: "long",
          year: "numeric",
        });
  };

  function highlight(code, language) {
    if (typeof hljs !== "undefined" && language && hljs.getLanguage(language)) {
      try {
        return hljs.highlight(code, { language }).value;
      } catch {}
    }
    return esc(code);
  }

  function loadScript(src) {
    return new Promise((ok, no) => {
      const s = document.createElement("script");
      s.src = src;
      s.onload = ok;
      s.onerror = () => no(new Error("failed to load " + src));
      document.head.append(s);
    });
  }

  function autosize(ta) {
    ta.style.height = "auto";
    ta.style.height = ta.scrollHeight + "px";
  }

  /* ---------- registry ---------- */

  let uid = 0;

  const BLOCKS = {
    text: (b) => `<div class="prose">${md(b.md)}</div>`,

    image: (b) => `
    <figure class="b-image">
      <img src="${esc(b.src)}" alt="${esc(b.alt)}" loading="lazy">
      ${b.caption ? `<figcaption>${esc(b.caption)}</figcaption>` : ""}
    </figure>`,

    code: (b) => `
    <div class="b-code">
      ${b.language ? `<span class="lang">${esc(b.language)}</span>` : ""}
      <pre><code>${highlight(b.code, b.language)}</code></pre>
    </div>`,

    quote: (b) => `
    <blockquote class="b-quote">
      <p>${esc(b.text)}</p>
      ${b.cite ? `<cite>${esc(b.cite)}</cite>` : ""}
    </blockquote>`,

    gallery: (b) => `
    <figure class="b-gallery">
      <div class="grid">
        ${(b.images || [])
          .map(
            (im) => `
          <button class="shot" data-full="${esc(im.src)}">
            <img src="${esc(im.src)}" alt="${esc(im.alt)}" loading="lazy">
          </button>`,
          )
          .join("")}
      </div>
      ${b.caption ? `<figcaption>${esc(b.caption)}</figcaption>` : ""}
    </figure>`,

    slideshow: (b) => {
      const slides = b.slides || [];
      if (!slides.length)
        return `<figure class="b-slideshow empty">No slides yet.</figure>`;
      return `
    <figure class="b-slideshow" id="slides-${++uid}" tabindex="0">
      <div class="stage">
        ${slides
          .map(
            (s, i) => `
          <img class="slide${i ? "" : " on"}" src="${esc(s.src)}"
               alt="${esc(s.alt)}" loading="lazy" data-caption="${esc(s.caption)}">`,
          )
          .join("")}
      </div>
      <div class="controls">
        <button class="nav prev" aria-label="Previous">&larr;</button>
        <div class="dots">
          ${slides
            .map(
              (_, i) =>
                `<button class="dot${i ? "" : " on"}" data-i="${i}" aria-label="Slide ${i + 1}"></button>`,
            )
            .join("")}
        </div>
        <button class="nav next" aria-label="Next">&rarr;</button>
      </div>
      <figcaption>${esc(slides[0].caption || b.caption || "")}</figcaption>
    </figure>`;
    },

    math: (b) => `
    <figure class="b-math">
      ${tex({ tex: b.tex || "", display: true })}
      ${b.caption ? `<figcaption>${esc(b.caption)}</figcaption>` : ""}
    </figure>`,

    /* Runs in the reader's own browser via Pyodide. Nothing is sent
     anywhere; the code is editable so people can poke at it.       */
    python: (b) => `
    <div class="b-python" data-packages="${esc((b.packages || []).join(","))}"
         data-autorun="${b.autorun ? "true" : "false"}">
      <div class="py-head">
        <span class="lang">python</span>
        <div class="py-actions">
          <button class="py-reset">Reset</button>
          <button class="py-run">Run</button>
        </div>
      </div>
      <textarea class="py-code" spellcheck="false">${esc(b.code)}</textarea>
      <div class="py-out hidden"></div>
    </div>`,
  };

  const renderBlock = (b) => {
    const fn = BLOCKS[b.type];
    return fn
      ? fn(b)
      : `<div class="b-unknown">Unknown block type: ${esc(b.type)}</div>`;
  };

  const renderBlocks = (blocks) => (blocks || []).map(renderBlock).join("");

  /* ---------- python runtime ---------- */

  let pyodideReady = null;

  function bootPyodide(say) {
    if (!pyodideReady) {
      pyodideReady = (async () => {
        say("Downloading Python (~10 MB, once per visit)…");
        await loadScript(PYODIDE_URL + "pyodide.js");
        return loadPyodide({ indexURL: PYODIDE_URL });
      })();
    }
    return pyodideReady;
  }

  const FIGURE_SNIPPET = `
import sys, io, base64, json
_figs = []
_plt = sys.modules.get("matplotlib.pyplot")
if _plt is not None:
    for _n in _plt.get_fignums():
        _buf = io.BytesIO()
        _plt.figure(_n).savefig(_buf, format="png", dpi=110, bbox_inches="tight")
        _figs.append(base64.b64encode(_buf.getvalue()).decode())
    _plt.close("all")
json.dumps(_figs)
`;

  async function runPython(el) {
    const out = el.querySelector(".py-out");
    const btn = el.querySelector(".py-run");
    const code = el.querySelector(".py-code").value;
    const pkgs = (el.dataset.packages || "").split(",").filter(Boolean);

    btn.disabled = true;
    out.classList.remove("hidden");
    const say = (m) => {
      out.innerHTML = `<p class="py-status">${esc(m)}</p>`;
    };

    try {
      const py = await bootPyodide(say);

      if (pkgs.length) {
        say(`Loading ${pkgs.join(", ")}…`);
        await py.loadPackage(pkgs);
        if (pkgs.includes("matplotlib")) {
          await py.runPythonAsync('import matplotlib; matplotlib.use("AGG")');
        }
      }

      say("Running…");
      let buffer = "";
      py.setStdout({
        batched: (s) => {
          buffer += s + "\n";
        },
      });
      py.setStderr({
        batched: (s) => {
          buffer += s + "\n";
        },
      });

      let result;
      try {
        result = await py.runPythonAsync(code);
      } finally {
        py.setStdout({});
        py.setStderr({});
      }

      const figures = JSON.parse(await py.runPythonAsync(FIGURE_SNIPPET));

      let html = "";
      if (buffer.trim())
        html += `<pre class="py-stdout">${esc(buffer.trimEnd())}</pre>`;
      if (result !== undefined && result !== null) {
        html += `<pre class="py-value">${esc(String(result))}</pre>`;
      }
      html += figures
        .map(
          (f) =>
            `<img class="py-figure" src="data:image/png;base64,${f}" alt="Figure">`,
        )
        .join("");

      out.innerHTML = html || '<p class="py-status">Ran with no output.</p>';
    } catch (err) {
      out.innerHTML = `<pre class="py-err">${esc(err.message || err)}</pre>`;
    } finally {
      btn.disabled = false;
    }
  }

  /* ---------- hydration ----------
   `onImage` lets the host decide what clicking a gallery shot does.  */

  function hydrate(root, { onImage } = {}) {
    root
      .querySelectorAll(".b-gallery .shot")
      .forEach((btn) =>
        btn.addEventListener("click", () => onImage?.(btn.dataset.full)),
      );

    root.querySelectorAll(".b-slideshow").forEach((el) => {
      const slides = [...el.querySelectorAll(".slide")];
      if (!slides.length) return;
      const dots = [...el.querySelectorAll(".dot")];
      const cap = el.querySelector("figcaption");
      let i = 0;

      const go = (n) => {
        i = (n + slides.length) % slides.length;
        slides.forEach((s, k) => s.classList.toggle("on", k === i));
        dots.forEach((d, k) => d.classList.toggle("on", k === i));
        cap.textContent = slides[i].dataset.caption || "";
      };

      el.querySelector(".prev").addEventListener("click", () => go(i - 1));
      el.querySelector(".next").addEventListener("click", () => go(i + 1));
      dots.forEach((d) => d.addEventListener("click", () => go(+d.dataset.i)));
      el.addEventListener("keydown", (e) => {
        if (e.key === "ArrowLeft") {
          e.preventDefault();
          go(i - 1);
        }
        if (e.key === "ArrowRight") {
          e.preventDefault();
          go(i + 1);
        }
      });
    });

    root.querySelectorAll(".b-python").forEach((el) => {
      const ta = el.querySelector(".py-code");
      const original = ta.value;
      autosize(ta);
      ta.addEventListener("input", () => autosize(ta));
      ta.addEventListener("keydown", (e) => {
        if (e.key === "Tab") {
          e.preventDefault();
          const { selectionStart: s } = ta;
          ta.value =
            ta.value.slice(0, s) + "    " + ta.value.slice(ta.selectionEnd);
          ta.selectionStart = ta.selectionEnd = s + 4;
        }
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
          e.preventDefault();
          runPython(el);
        }
      });
      el.querySelector(".py-reset").addEventListener("click", () => {
        ta.value = original;
        autosize(ta);
        el.querySelector(".py-out").classList.add("hidden");
      });
      el.querySelector(".py-run").addEventListener("click", () =>
        runPython(el),
      );
      if (el.dataset.autorun === "true") runPython(el);
    });
  }

  return {
    BLOCKS,
    renderBlock,
    renderBlocks,
    hydrate,
    esc,
    md,
    tex,
    prettyDate,
    highlight,
  };
})();
