// MathJax config used ONLY for the PDF / book build (see mkdocs.pdf.yml).
//
// Difference vs. the web config (mathjax.js): SVG output instead of CHTML.
// SVG math paginates dramatically faster when Chromium prints the single,
// very large combined page, and embeds cleanly into the PDF without depending
// on web-font layout. fontCache "none" inlines glyph paths into each equation
// so nothing breaks across printed page boundaries.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  svg: {
    fontCache: "none"
  },
  options: {
    ignoreHtmlClass: ".*",
    processHtmlClass: "arithmatex"
  }
};

// Material loads pages instantly (SPA); re-typeset on each navigation.
// On the print page this simply typesets once on load.
if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    MathJax.typesetPromise();
  });
}
