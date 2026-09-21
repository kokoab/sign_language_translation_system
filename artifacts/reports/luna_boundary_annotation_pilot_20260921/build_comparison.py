#!/usr/bin/env python3
"""Build the source-versus-Luna boundary review page."""

from __future__ import annotations

from html import escape
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def seconds(value: float) -> str:
    return f"{value:.3f}s"


def delta(value: float) -> str:
    return f"{value * 1000:+.0f} ms"


def position(value: float, start: float, end: float) -> float:
    return max(0.0, min(100.0, 100 * (value - start) / (end - start)))


def span(left: float, right: float, start: float, end: float) -> tuple[float, float]:
    x = position(left, start, end)
    return x, max(0.8, position(right, start, end) - x)


def main() -> None:
    rows = json.loads((HERE / "luna_annotations.json").read_text())
    blind = {row["item"]: row for row in json.loads((HERE / "blind_manifest.json").read_text())}
    cards = []
    for row in rows:
        review = blind[row["item"]]
        times = review["frame_times_seconds"]
        review_start, review_end = float(times[0]), float(times[-1])
        source_x, source_w = span(row["source_start_seconds"], row["source_end_seconds"], review_start, review_end)
        luna_x, luna_w = span(row["luna_start_seconds"], row["luna_end_seconds"], review_start, review_end)
        start_abs, end_abs = abs(row["start_delta_seconds"]), abs(row["end_delta_seconds"])
        if start_abs <= .1 and end_abs <= .1:
            agreement, group = "both edges ≤100 ms", "within100"
        elif start_abs <= .2 and end_abs <= .2:
            agreement, group = "both edges ≤200 ms", "within200"
        else:
            agreement, group = "at least one edge >200 ms", "outside200"
        if row["edge_censored"]:
            group = "censored"
        video = os.path.relpath(ROOT / row["video_path"], HERE)
        sheet = review["sheet"]
        badges = [f'<span class="badge {group}">{escape(agreement)}</span>']
        if row["edge_censored"]:
            badges.append('<span class="badge censored">review range censored</span>')
        cards.append(f"""
<article class="card" data-group="{group}">
  <header><div><span class="item">{escape(row['item'])}</span><h2>{escape(row['gloss'])}</h2></div><div class="badges">{''.join(badges)}</div></header>
  <div class="media">
    <video controls preload="metadata" src="{escape(video)}"></video>
    <div class="details">
      <div class="times">
        <div><b>ASLLRP source</b><span>{seconds(row['source_start_seconds'])} → {seconds(row['source_end_seconds'])}</span></div>
        <div><b>Luna review</b><span>{seconds(row['luna_start_seconds'])} → {seconds(row['luna_end_seconds'])}</span></div>
        <div><b>Difference</b><span>start {delta(row['start_delta_seconds'])}; end {delta(row['end_delta_seconds'])}</span></div>
      </div>
      <div class="timeline" aria-label="Boundary comparison from {seconds(review_start)} to {seconds(review_end)}">
        <div class="track source"><span style="left:{source_x:.3f}%;width:{source_w:.3f}%"></span><b>source</b></div>
        <div class="track luna"><span style="left:{luna_x:.3f}%;width:{luna_w:.3f}%"></span><b>Luna</b></div>
        <div class="axis"><span>{seconds(review_start)}</span><span>{seconds(review_end)}</span></div>
      </div>
      <div class="buttons">
        <button data-time="{row['source_start_seconds']}">source start</button>
        <button data-time="{row['source_end_seconds']}">source end</button>
        <button data-time="{row['luna_start_seconds']}">Luna start</button>
        <button data-time="{row['luna_end_seconds']}">Luna end</button>
        <button data-start="{row['source_start_seconds']}" data-end="{row['source_end_seconds']}">play source</button>
        <button data-start="{row['luna_start_seconds']}" data-end="{row['luna_end_seconds']}">play Luna</button>
      </div>
      <p class="reason"><b>Luna rationale:</b> {escape(row['reason'])}</p>
      <p class="meta">{escape(review['source'])} · confidence {escape(row['confidence'])} · sampled frame {row['start_frame']}–{row['end_frame']}</p>
    </div>
  </div>
  <details><summary>Show the exact 24-frame sheet reviewed by Luna</summary><img loading="lazy" src="{escape(sheet)}" alt="Timestamped frame sheet for {escape(row['gloss'])}"></details>
</article>""")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ASLLRP source vs Luna boundary review</title>
<style>
:root{{--bg:#f5f7fb;--panel:#fff;--text:#172033;--muted:#637083;--line:#dbe1ea;--source:#2166d1;--luna:#e87817;--good:#16784a;--warn:#9b5b00;--bad:#b52e3a}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,sans-serif}} main{{max-width:1240px;margin:auto;padding:28px 18px 64px}} h1{{font-size:clamp(28px,5vw,48px);line-height:1.05;margin:.2em 0}} h2{{display:inline;margin:0 0 0 8px;font-size:22px}} a{{color:var(--source)}} .intro,.finding,.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:0 4px 18px #1720330b}} .intro,.finding{{padding:18px 20px;margin:14px 0}} .finding strong{{color:var(--bad)}} .controls{{display:flex;gap:10px;align-items:center;margin:22px 0 12px}} select,button{{font:inherit;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--text);padding:8px 10px;cursor:pointer}} .count{{color:var(--muted)}} .card{{padding:18px;margin:14px 0}} .card>header{{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:14px}} .item,.meta{{color:var(--muted);font-size:13px}} .badges{{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:6px}} .badge{{border-radius:999px;padding:4px 9px;font-size:12px;background:#edf1f6}} .badge.within100{{background:#d9f3e5;color:var(--good)}} .badge.within200{{background:#fff0cf;color:var(--warn)}} .badge.outside200,.badge.censored{{background:#ffe1e3;color:var(--bad)}} .media{{display:grid;grid-template-columns:minmax(280px,43%) 1fr;gap:18px}} video{{width:100%;max-height:430px;background:#0c1018;border-radius:10px}} .times{{display:grid;gap:5px}} .times div{{display:flex;justify-content:space-between;gap:16px;border-bottom:1px solid var(--line);padding:5px 0}} .timeline{{margin:18px 0 10px}} .track{{height:34px;position:relative;background:#eef1f5;border-radius:7px;margin:6px 0;overflow:hidden}} .track span{{position:absolute;top:0;bottom:0;border-radius:6px}} .track b{{position:relative;z-index:1;display:block;padding:6px 9px;color:#fff;text-shadow:0 1px 2px #0009}} .track.source span{{background:var(--source)}} .track.luna span{{background:var(--luna)}} .axis{{display:flex;justify-content:space-between;color:var(--muted);font-size:12px}} .buttons{{display:flex;flex-wrap:wrap;gap:7px;margin:14px 0}} .reason{{margin:14px 0 4px}} details{{margin-top:14px;border-top:1px solid var(--line);padding-top:10px}} summary{{cursor:pointer;font-weight:650}} details img{{display:block;width:100%;height:auto;margin-top:10px;border-radius:8px}} .legend{{display:flex;gap:18px;flex-wrap:wrap}} .dot{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px}} .dot.source{{background:var(--source)}} .dot.luna{{background:var(--luna)}}
@media(max-width:760px){{.media{{grid-template-columns:1fr}}.card>header{{display:block}}.badges{{justify-content:flex-start;margin-top:8px}}.times div{{display:block}}}}
@media(prefers-color-scheme:dark){{:root{{--bg:#10141d;--panel:#181f2b;--text:#edf2fa;--muted:#aab4c4;--line:#344052}}.track{{background:#293241}}.badge{{background:#2b3544}}}}
</style></head><body><main>
<p class="item">24 train-only ASLLRP examples · one blind Luna-low review per clip</p>
<h1>Source annotation vs Luna boundary review</h1>
<section class="intro">
  <p><b>The two intervals do not use perfectly identical semantics.</b> ASLLRP marks the linguistic sign interval: preparatory movement is excluded, the sign ends before movement into the next sign, and a final hold may be annotated separately. The Luna prompt asked for the first through last frame belonging to the named sign, which reviewers often interpreted as the full visible articulation or hold.</p>
  <p class="legend"><span><i class="dot source"></i>ASLLRP source interval</span><span><i class="dot luna"></i>Luna visual review</span></p>
  <p>Source convention: <a href="https://www.bu.edu/asllrp/asllrpr11.pdf">ASLLRP annotation conventions</a>. Dataset context: <a href="https://www.bu.edu/asllrp/2.html">ASLLRP SignStream corpora</a>.</p>
</section>
<section class="finding">
  <p><strong>Correction:</strong> the pilot does not show that ASLLRP annotated these signs incorrectly. Calling the source boundaries “questionable” was too strong. The disagreement mixes a definition mismatch, coarse sampled frames, and a single reviewer.</p>
  <p>The ±100 ms score was strict for exact-edge evaluation and should not be a data-quality verdict. It was not the whole model failure: the coherent decoder reached only 40.46% boundary F1 even at ±200 ms and still failed held-sign/repetition checks.</p>
  <p>The confirmed source-pipeline problems were narrower: 21 of 1,160 derived ASLLRP crops clipped at least one annotation (22 clipped occurrences, all mapped to OTHER), and the earlier context-window objective stopped every known training window before its annotated sign end. Those are preprocessing and supervision defects, not evidence that the original study annotations are broadly wrong.</p>
</section>
<div class="controls"><label for="filter">Show</label><select id="filter"><option value="all">all 24 clips</option><option value="within100">both edges ≤100 ms</option><option value="within200">both edges ≤200 ms</option><option value="outside200">an edge &gt;200 ms</option><option value="censored">review range censored</option></select><span class="count" id="count"></span></div>
{''.join(cards)}
</main><script>
const cards=[...document.querySelectorAll('.card')], filter=document.querySelector('#filter'), count=document.querySelector('#count');
function apply(){{let n=0;for(const card of cards){{const show=filter.value==='all'||card.dataset.group===filter.value;card.hidden=!show;n+=show}}count.textContent=`${{n}} shown`}} filter.addEventListener('change',apply);apply();
document.addEventListener('click',event=>{{const button=event.target.closest('button');if(!button)return;const video=button.closest('.card').querySelector('video');if(button.dataset.time){{video.currentTime=Number(button.dataset.time);video.pause();return}}const end=Number(button.dataset.end);video.currentTime=Number(button.dataset.start);video.play();const stop=()=>{{if(video.currentTime>=end){{video.pause();video.removeEventListener('timeupdate',stop)}}}};video.addEventListener('timeupdate',stop)}});
</script></body></html>"""
    (HERE / "comparison.html").write_text(html)
    assert len(cards) == 24 and html.count('<article class="card"') == 24
    print(HERE / "comparison.html")


if __name__ == "__main__":
    main()
