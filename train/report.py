#!/usr/bin/env python3
"""Builds base/report.html from base/history.json, base/eval.json and base/metrics.json."""
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
ld = lambda n, d: json.load(open(base / n)) if (base / n).exists() else d
hist, ev, met = ld("history.json", []), ld("eval.json", {}), ld("metrics.json", {})
it = [[m["iter"], m.get("winrate_last200"), m.get("delegate")] for m in met.get("iters", []) if m.get("winrate_last200") is not None]
it = it[::max(1, len(it) // 500)]
decks = sorted(([k, v[0], v[1]] for k, v in ev.get("by_deck", {}).items()), key=lambda r: r[1] / max(1, r[2]))
data = {"hist": hist, "it": it, "decks": decks, "ev": {k: ev.get(k) for k in ("games", "winrate", "wins", "losses", "draws")}}
HTML = """<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Обучение</title><style>
body{margin:0;background:#0f1420;color:#e8ecf4;font:14px/1.45 system-ui,sans-serif}main{max-width:760px;margin:0 auto;padding:14px}
h1{font-size:20px;margin:4px 0}h2{font-size:16px;margin:20px 0 8px}.m{color:#8a94a8}
.k{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}@media(min-width:560px){.k{grid-template-columns:repeat(4,1fr)}}
.k div{background:#171e2e;border-radius:12px;padding:10px}.k b{display:block;font-size:22px}.k span{color:#8a94a8;font-size:12px}
canvas{width:100%;background:#171e2e;border-radius:12px}table{width:100%;border-collapse:collapse;font-size:12px}
td{padding:5px 4px;border-bottom:1px solid #26304a}td:last-child{text-align:right}
.l{font-size:12px;color:#8a94a8}.d{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 4px 0 10px}
</style></head><body><main><h1>Обучение нейросети X-Bow 3.0</h1><div class="m" id="sub"></div>
<div class="k" id="k"></div>
<h2>Поколения (проверка на мета-колодах)</h2><canvas id="g" width="720" height="300"></canvas>
<div class="l"><span class="d" style="background:#3ccf8e"></span>лучший<span class="d" style="background:#4ea1ff"></span>средний по воркерам<span class="d" style="background:#8a94a8"></span>уровень правил 37.6%</div>
<h2>Кривая обучения текущей лучшей версии</h2><canvas id="c" width="720" height="260"></canvas>
<div class="l"><span class="d" style="background:#3ccf8e"></span>винрейт (200 боёв, все колоды)<span class="d" style="background:#ffb648"></span>свои решения нейросети</div>
<h2>Против каких колод (от худших)</h2><table id="t"></table></main><script>
const D=__DATA__,H=D.hist,L=H[H.length-1]||{},E=D.ev,pc=v=>v==null?'—':Math.round(v*1000)/10+'%';
document.getElementById('sub').textContent=H.length?`Поколений: ${H.length} · воркеров в последнем: ${(L.workers||[]).length}`:'Ещё нет завершённых поколений';
document.getElementById('k').innerHTML=[[pc(L.best),'лучший винрейт'],[pc(L.mean),'средний'],[E.games||0,'боёв в проверке'],[H.length,'поколений']].map(([a,b])=>`<div><b>${a}</b><span>${b}</span></div>`).join('');
function chart(id,n,series,ref){const c=document.getElementById(id),g=c.getContext('2d'),W=c.width,Hh=c.height,P=32,X=v=>P+(W-2*P)*v/Math.max(1,n),Y=v=>Hh-P-(Hh-2*P)*v;
g.strokeStyle='#26304a';g.fillStyle='#8a94a8';g.font='12px sans-serif';for(let k=0;k<=5;k++){const y=Y(k/5);g.beginPath();g.moveTo(P,y);g.lineTo(W-P,y);g.stroke();g.fillText(k*20+'%',2,y+4)}
if(ref){g.setLineDash([6,5]);g.strokeStyle='#8a94a8';g.beginPath();g.moveTo(P,Y(ref));g.lineTo(W-P,Y(ref));g.stroke();g.setLineDash([])}
for(const [pts,col] of series){g.strokeStyle=col;g.lineWidth=2;g.beginPath();pts.forEach(([x,y],q)=>q?g.lineTo(X(x),Y(y)):g.moveTo(X(x),Y(y)));g.stroke();if(pts.length<40)pts.forEach(([x,y])=>{g.fillStyle=col;g.beginPath();g.arc(X(x),Y(y),3,0,7);g.fill()})}}
chart('g',Math.max(1,H.length),[[H.map(h=>[h.gen,h.best]),'#3ccf8e'],[H.map(h=>[h.gen,h.mean]),'#4ea1ff']],.376);
const I=D.it;chart('c',I.length?I[I.length-1][0]:1,[[I.map(r=>[r[0],r[1]]),'#3ccf8e'],[I.map(r=>[r[0],r[2]==null?0:1-r[2]]),'#ffb648']]);
document.getElementById('t').innerHTML=D.decks.map(([n,w,g])=>`<tr><td>${n}</td><td>${w}/${g}</td><td>${pc(w/Math.max(1,g))}</td></tr>`).join('')||'<tr><td>нет данных</td></tr>';
</script></body></html>"""
open(base / "report.html", "w", encoding="utf-8").write(HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))))
print("report ok")
