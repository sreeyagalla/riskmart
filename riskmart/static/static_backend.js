/* Static (GitHub Pages) backend for the chat UI.
 * A JS port of the rule planner in riskmart/qa.py + riskmart/agent.py. It parses the question, then replays the
 * grounded answer that the Python engine precomputed (snapshot.json). It computes nothing itself.
 * Parity with the Python agent is enforced by tests/test_static_demo.py (runs this file under node). */
(function (root) {
  "use strict";
  let SNAP = null, R = null;

  const rx = (src, flags) => new RegExp(src, flags || "");
  function validate(o) {
    const known = R.intent_order.map(x => x[0]).concat(["portfolio_var", "var_by_asset_class"]);
    if (!o || !known.includes(o.intent)) return null;
    const conf = o.confidence === undefined ? 0.99 : o.confidence;
    if (!R.allowed_confidence.includes(conf)) return null;
    const method = o.method || "historical";
    if (!["historical", "parametric"].includes(method)) return null;
    const ac = o.asset_class === undefined ? null : o.asset_class;
    if (ac !== null && !["Equity", "FX", "Commodity", "Rates", "Credit"].includes(ac)) return null;
    return { intent: o.intent, confidence: conf, method: method, asset_class: ac };
  }
  function unknownEntity(q) {
    const allowed = new Set(R.allowed_caps);
    for (const w of q.match(/\b[A-Z][A-Za-z0-9&]+\b/g) || []) if (!allowed.has(w.toLowerCase())) return w;
    return null;
  }
  function parse(q) {
    const ql = q.toLowerCase(), P = R.patterns;
    if (rx(P.forecast, "i").test(q)) return { intent: null, reason: "forecast/advice" };
    const ent = unknownEntity(q);
    if (ent) return { intent: null, reason: "unknown entity " + ent };
    let conf = 0.99;
    const m = ql.match(rx(P.confidence));
    if (m) {
      conf = parseFloat(m[1]) / 100.0;
      if (!R.allowed_confidence.includes(conf)) return { intent: null, reason: "unsupported confidence level" };
    }
    const method = rx(P.parametric).test(ql) ? "parametric" : "historical";
    let ac = null;
    for (const [word, cls] of Object.entries(R.class_words)) if (rx("\\b" + word + "\\b").test(ql)) { ac = cls; break; }
    const byClass = rx(P.by_class).test(ql);
    let i = null;
    for (const [name, pat] of R.intent_order) if (rx(pat).test(ql)) { i = name; break; }
    if (i === null && rx(P.var).test(ql)) i = (byClass || ac) ? "var_by_asset_class" : "portfolio_var";
    if (i === null) return { intent: null, reason: "no supported intent" };
    return { intent: validate({ intent: i, confidence: conf, method: method, asset_class: ac }), reason: null };
  }
  function followup(q, ctx) {
    if (!ctx) return null;
    const base = validate({ intent: ctx.intent, confidence: ctx.confidence, method: ctx.method, asset_class: ctx.asset_class });
    if (!base || rx(R.patterns.forecast, "i").test(q) || unknownEntity(q)) return null;
    const ql = q.toLowerCase();
    let { confidence: conf, method, asset_class: ac } = base, changed = false;
    const m = ql.match(rx(R.patterns.confidence));
    if (m) { conf = parseFloat(m[1]) / 100.0; changed = true; }
    if (/parametric|normal/.test(ql)) { method = "parametric"; changed = true; }
    else if (/historical/.test(ql)) { method = "historical"; changed = true; }
    for (const [w, c] of Object.entries(R.mod_class)) if (rx("\\b" + w + "\\b").test(ql)) { ac = c; changed = true; break; }
    if (!changed) return null;
    return validate({ intent: base.intent, confidence: conf, method: method, asset_class: ac });
  }
  function keyOf(it) {
    const parts = [it.intent];
    for (const d of R.relevant[it.intent]) {
      const v = it[d];
      parts.push(v === null || v === undefined ? "-" : String(v));
    }
    return parts.join("|");
  }
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  async function* chat(question, ctx, delay) {
    const q = question.trim();
    if (rx(R.patterns.refresh, "i").test(q)) {
      yield { type: "plan", intent: "refresh_data", parser: "rules" };
      yield { type: "approval_required", id: "static-demo", action: "refresh_data", title: "Refresh market data?",
        detail: "In the live app this re-downloads the 10 public FRED series and rebuilds the SQLite mart, and only after you approve. " +
                "This hosted demo has no server, so the action is disabled here. Nothing has been changed." };
      yield { type: "done" };
      return;
    }
    let { intent, reason } = parse(q), parser = "rules";
    if (intent === null && reason === "no supported intent") {
      const fu = followup(q, ctx);
      if (fu) { intent = fu; parser = "rules+context"; }
    }
    if (intent === null) {
      yield { type: "refusal", reason: reason || "unsupported" };
      for (const t of tokens(R.refusal)) { if (delay) await sleep(delay); yield { type: "token", text: t }; }
      yield { type: "done" };
      return;
    }
    const e = SNAP.entries[keyOf(intent)];
    yield { type: "plan", intent: intent, parser: parser };
    if (!e) { yield { type: "refusal", reason: "not in snapshot" }; yield { type: "done" }; return; }
    for (let i = 0; i < e.tools.length; i++) yield { type: "tool_call", id: i, name: e.tools[i] };
    for (let i = 0; i < e.tools.length; i++) yield { type: "tool_result", id: i, ok: true, summary: "precomputed by the Python engine" };
    for (const c of e.cards) yield Object.assign({ type: "card" }, c);
    for (const t of tokens(e.answer)) { if (delay) await sleep(delay); yield { type: "token", text: t }; }
    yield { type: "sources", items: e.sources };
    yield { type: "final", answer: e.answer, intent: intent };
    yield { type: "done" };
  }
  const tokens = text => text.match(/\S+\s*/g) || [];

  const api = {
    init(snapshot) { SNAP = snapshot; R = snapshot.rules; return api; },
    meta() { return SNAP.meta; },
    parse, followup, chat,
    approve: async () => ({ status: "disabled", message: "Disabled in this static demo (no server). Clone the repo and run it locally to try the approval flow." }),
    async load(url) { const r = await fetch(url || "snapshot.json"); return api.init(await r.json()); },
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.RiskmartStatic = api;
})(typeof window !== "undefined" ? window : globalThis);
