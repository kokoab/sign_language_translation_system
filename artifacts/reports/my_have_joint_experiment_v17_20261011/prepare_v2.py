from pathlib import Path
P=Path(__file__).resolve().parent;d=P/'v2';d.mkdir(exist_ok=True)
s=(P/'validation/LiveReelDecoder.swift').read_text().replace('                    last = w','                    if last == nil || w.startFrame >= last!.startFrame { last = w }')
(d/'LiveReelDecoder.swift').write_text(s)
s=(P/'runtime_hybrid.py').read_text()
s=s.replace("                    self.last = self._word(g['gloss'], st, g['end'] + origin, T, g['log_q'], early=True)\n                    self.shown_early.append((st, self.last))\n                    committed.append(self.last)","                    w = self._word(g['gloss'], st, g['end'] + origin, T, g['log_q'], early=True)\n                    if self.last is None or st >= self.last['start_frame']: self.last = w\n                    self.shown_early.append((st, w))\n                    committed.append(w)")
s=s.replace("                    self.last = self._word(g['gloss'], start, end, T, g['log_q'])\n                    committed.append(self.last)","                    w = self._word(g['gloss'], start, end, T, g['log_q'])\n                    if self.last is None or start >= self.last['start_frame']: self.last = w\n                    committed.append(w)")
(P/'runtime_hybrid_v2.py').write_text(s)
# Source-equivalent no-ensemble runtime for focused state checks.
a=s.index('        my, like = self.recognizer.labels.index');b=s.index('        fist = self._fist_geometry()',a)
(P/'runtime_ordered_v2.py').write_text(s[:a]+s[b:])
