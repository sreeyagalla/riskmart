// usage: node static_parity.js snapshot.json cases.json  -> prints the event stream per case as JSON
const fs = require("fs");
const RM = require("../riskmart/static/static_backend.js");
RM.init(JSON.parse(fs.readFileSync(process.argv[2], "utf8")));
const cases = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
(async () => {
  const out = [];
  for (const c of cases) {
    const ev = [];
    for await (const e of RM.chat(c.q, c.ctx || null, 0)) ev.push(e);
    out.push(ev);
  }
  process.stdout.write(JSON.stringify(out));
})();
