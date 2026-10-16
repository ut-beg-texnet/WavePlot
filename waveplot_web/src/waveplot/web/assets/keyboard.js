// WavePlot keyboard shortcuts: forwards key presses (outside text inputs) to the "store-kbd" store.
// Page callbacks interpret them (frame stepping, display modes, ...). Mnemonics follow WavePlot v407.
(function () {
  const HANDLED = new Set([
    "ArrowLeft", "ArrowRight", "Home", "End", " ", "+", "=", "-", "_",
    "m", "M", "c", "C", "l", "L", "a", "A", "b", "B", ">", "<", "d", "D", "k", "K",
    "i", "I", "g", "G", "h", "H", "v", "V", "x", "X", "?", "z", "Z", "s", "S"
  ]);
  function isTyping(el) {
    if (!el) return false;
    const tag = (el.tagName || "").toLowerCase();
    return tag === "input" || tag === "textarea" || tag === "select" || el.isContentEditable;
  }
  document.addEventListener("keydown", function (ev) {
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (isTyping(document.activeElement)) return;
    if (!HANDLED.has(ev.key)) return;
    if (!window.dash_clientside || !window.dash_clientside.set_props) return;
    if (ev.key === " " || ev.key.startsWith("Arrow") || ev.key === "Home" || ev.key === "End") ev.preventDefault();
    window.dash_clientside.set_props("store-kbd", {
      data: { key: ev.key, shift: ev.shiftKey, ts: Date.now(), path: window.location.pathname }
    });
  });
})();
