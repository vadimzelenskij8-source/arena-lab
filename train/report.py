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
mt = ld("matchups.json", {})
def mrows(mode):
    rows = []
    for d, r in (mt.get(mode) or {}).items():
        b = r["hist"][-1] if r.get("hist") else None
        t = (r["teacher"][0] / max(1, r["teacher"][1])) if r.get("teacher") else None
        rows.append([d, b, t, r.get("gap"), r.get("status")])
    return sorted(rows, key=lambda x: -(x[3] if x[3] is not None else -9))
mtd = {"any": mrows("any"), "xbow": mrows("xbow"), "sa": mt.get("solved_any"), "sx": mt.get("solved_xbow"),
       "na": mt.get("decks_any"), "nx": mt.get("decks_xbow"), "pa": mt.get("paused_any"), "px": mt.get("paused_xbow")}
ex = {"ava": (ev.get("any_vs_any") or {}).get("winrate"), "xbow": (ev.get("xbow") or {}).get("winrate")}
grp = sorted(([k, v[0], v[1]] for k, v in ev.get("by_group", {}).items()), key=lambda r: r[1] / max(1, r[2]))
ph = [[k, v[0], v[1]] for k, v in ev.get("by_phase", {}).items()]
data = {"mt": mtd, "ex": ex, "grp": grp, "ph": ph, "tm": met.get("timing", {}), "hist": hist, "it": it, "decks": decks, "ev": {k: ev.get(k) for k in ("games", "winrate", "wins", "losses", "draws")}}
HTML = """<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Обучение</title><style>
body{margin:0;background:#0f1420;color:#e8ecf4;font:14px/1.45 system-ui,sans-serif}main{max-width:760px;margin:0 auto;padding:14px}
h1{font-size:20px;margin:4px 0}h2{font-size:16px;margin:20px 0 8px}.m{color:#8a94a8}
.k{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}@media(min-width:560px){.k{grid-template-columns:repeat(4,1fr)}}
.k div{background:#171e2e;border-radius:12px;padding:10px}.k b{display:block;font-size:22px}.k span{color:#8a94a8;font-size:12px}
canvas{width:100%;background:#171e2e;border-radius:12px}table{width:100%;border-collapse:collapse;font-size:12px}
td{padding:5px 4px;border-bottom:1px solid #26304a}td:last-child{text-align:right}
.l{font-size:12px;color:#8a94a8}.d{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 4px 0 10px}
</style></head><body><main><h1>Обучение нейросети X-Bow 3.0 · поколение 2+</h1><div class="m" id="sub"></div>
<div class="k" id="k"></div>
<h2>Поколения (проверка на мета-колодах)</h2><canvas id="g" width="720" height="300"></canvas>
<div class="l"><span class="d" style="background:#3ccf8e"></span>лучший<span class="d" style="background:#4ea1ff"></span>средний по воркерам<span class="d" style="background:#8a94a8"></span>уровень правил 37.6%</div>
<h2>Кривая обучения текущей лучшей версии</h2><canvas id="c" width="720" height="260"></canvas>
<div class="l"><span class="d" style="background:#3ccf8e"></span>винрейт (200 боёв, все колоды)<span class="d" style="background:#ffb648"></span>свои решения нейросети</div>
<h2>Три экзамена</h2><div class="k" id="k3"></div>
<h2>Матчапы: любая колода против меты</h2><div class="m" id="sa"></div><table id="ma"></table>
<h2>Матчапы: X-Bow против меты</h2><div class="m" id="sx"></div><table id="mx"></table>
<h2>По архетипам (от худших)</h2><table id="tg"></table>
<h2>По фазам боя</h2><table id="tp"></table>
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
const T=D.mt||{},st={solved:'решено',paused:'пауза',active:'учится'};
document.getElementById('k3').innerHTML=[[pc(E.winrate),'любая против меты'],[pc((D.ex||{}).ava),'любая против любой'],[pc((D.ex||{}).xbow),'X-Bow против меты']].map(([a,b])=>`<div><b>${a}</b><span>${b}</span></div>`).join('');
document.getElementById('sa').textContent=T.na?`Решено ${T.sa} из ${T.na}, на паузе ${T.pa}. Столбцы: бот / учитель (потолок) / разрыв / статус`:'Нет данных';
document.getElementById('sx').textContent=T.nx?`Решено ${T.sx} из ${T.nx}, на паузе ${T.px}`:'Нет данных';
const mr=r=>`<tr><td>${r[0]}</td><td>${pc(r[1])}</td><td>${pc(r[2])}</td><td>${r[3]==null?'—':(r[3]>0?'+':'')+Math.round(r[3]*100)}</td><td>${st[r[4]]||r[4]}</td></tr>`;
document.getElementById('ma').innerHTML=(T.any||[]).map(mr).join('');
document.getElementById('mx').innerHTML=(T.xbow||[]).map(mr).join('');
const row=([n,w,g])=>`<tr><td>${n==='overtime'?'дошло до овертайма':n==='regular'?'без овертайма':n}</td><td>${w}/${g}</td><td>${pc(w/Math.max(1,g))}</td></tr>`;
document.getElementById('tg').innerHTML=(D.grp||[]).map(row).join('')||'<tr><td>нет данных</td></tr>';
document.getElementById('tp').innerHTML=(D.ph||[]).map(row).join('')||'<tr><td>нет данных</td></tr>';
document.getElementById('t').innerHTML=D.decks.map(([n,w,g])=>`<tr><td>${n}</td><td>${w}/${g}</td><td>${pc(w/Math.max(1,g))}</td></tr>`).join('')||'<tr><td>нет данных</td></tr>';
</script></body></html>"""
open(base / "report.html", "w", encoding="utf-8").write(HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))))
print("report ok")
