window.Crossbirds = (() => {
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

  const note = (cls, x, y) => `
  <g transform="translate(${x} ${y})"><g class="note ${cls}">
    <ellipse rx="4" ry="3" transform="rotate(-20)"/>
    <path d="M3.4 -1V-15q6 2 6.5 8" fill="none" stroke-width="2" stroke-linecap="round"/>
  </g></g>`;

  const birdSVG = (color) => `
<svg class="bird-svg" viewBox="0 0 120 112" aria-hidden="true" style="--c:${esc(color)}">
  ${note("n1", 100, 32)}${note("n2", 104, 26)}
  <path d="M26 90 L4 72 L12 96 Z" fill="var(--c)" stroke="var(--c)" stroke-width="4" stroke-linejoin="round"/>
  <path d="M16 94 L100 94 L60 18 Z" fill="var(--c)" stroke="var(--c)" stroke-width="8" stroke-linejoin="round"/>
  <path d="M28 90 L66 90 L45 60 Z" fill="#000" opacity=".13"/>
  <path class="beak-top" d="M75 38 L97 44 L76 45 Z" fill="#f4c24a" stroke="#f4c24a" stroke-width="2" stroke-linejoin="round"/>
  <path class="beak-bot" d="M76 45 L94 46 L77 51 Z" fill="#d9a02e" stroke="#d9a02e" stroke-width="2" stroke-linejoin="round"/>
  <circle cx="64" cy="42" r="4.4" fill="#172d30"/>
  <circle cx="65.5" cy="40.6" r="1.4" fill="#fff"/>
  <path d="M50 98v12M68 98v12M45 110h10M63 110h10" stroke="#172d30" stroke-width="3" stroke-linecap="round" fill="none"/>
</svg>`;

  return { birdSVG };
})();
