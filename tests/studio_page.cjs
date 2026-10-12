// Loads studio/index.html's script into Node with a small stand-in for the browser's document, so
// tests can build parts of the page (a scene card, a panel) and read or type into what they hold.
const fs = require("fs");
const path = require("path");
const vm = require("vm");

class Node_ {
  constructor(tag) {
    this.tagName = tag.toUpperCase(); this.children = []; this.attrs = {}; this.listeners = {};
    this.className = ""; this.value = ""; this.style = {}; this.hidden = false; this._text = "";
  }
  append(...kids) { for (const k of kids) this.children.push(typeof k === "string" ? new Text_(k) : k); }
  replaceChildren(...kids) { this.children = []; this.append(...kids); }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  removeAttribute(k) { delete this.attrs[k]; }
  remove() {}
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  dispatch(type) { for (const fn of this.listeners[type] || []) fn({ currentTarget: this, target: this, preventDefault() {} }); }
  set textContent(t) { this.children = []; this._text = String(t); }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(""); }
  get placeholder() { return this.attrs.placeholder ?? ""; }
  all() { return [this, ...this.children.flatMap((c) => (c.all ? c.all() : []))]; }
}
class Text_ { constructor(t) { this.data = t; } get textContent() { return this.data; } }

function load() {
  const html = fs.readFileSync(path.join(__dirname, "..", "studio", "index.html"), "utf8");
  const code = html.match(/<script>([\s\S]*)<\/script>/)[1].replace(/\nroute\(\);\s*$/, "\n");
  const document = { createElement: (t) => new Node_(t), createElementNS: (_, t) => new Node_(t), createTextNode: (t) => new Text_(t), getElementById: () => null,
                     body: new Node_("body"), querySelector: () => null };
  const ctx = { document, Node: Node_, window: { addEventListener() {} }, location: { hash: "" }, console,
                fetch: async () => ({ ok: true, json: async () => ({}) }), setTimeout, clearTimeout, confirm: () => true };
  vm.createContext(ctx);
  vm.runInContext(code + "\n;globalThis.__S = S;", ctx);
  return ctx;
}

// The input or textarea of the field whose label starts with `label`.
function fieldOf(root, label) {
  const lab = root.all().find((n) => n.tagName === "LABEL" && n.children[0]?.textContent?.startsWith(label));
  return lab && lab.all().find((n) => n.tagName === "TEXTAREA" || n.tagName === "INPUT" || n.tagName === "SELECT");
}

module.exports = { load, fieldOf };
