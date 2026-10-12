// Streaming segmental decoder, words only (port of LiveReelDecoder.swift / segmental_runtime_v17.py).
// Per frame: observation -> hand evidence (or none) -> boundary BIO (6-frame lookahead) -> candidate
// spans -> span recognizer -> semi-Markov sign/rest DP -> fixed-lag commit with early identity commit.
package com.example.slt_mobile_app.live

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

private const val UNK = 0
private const val O = 1
private const val B = 2
private const val I = 3

object LiveSegmental {
    fun logSoftmax(x: FloatArray): DoubleArray {
        val m = x.max().toDouble()
        var sum = 0.0
        for (v in x) sum += exp(v.toDouble() - m)
        val lse = m + ln(sum)
        return DoubleArray(x.size) { x[it].toDouble() - lse }
    }

    private fun argmax(row: FloatArray): Int {
        var best = 0
        for (k in 1 until row.size) if (row[k] > row[best]) best = k
        return best
    }

    fun candidateSpans(bio: List<FloatArray>, bThreshold: Float = 0.2f, minLen: Int = 3, maxLen: Int = 30,
                       maxInnerO: Int = 3): List<SpanKey> {
        val p = bio.map { r -> FloatArray(r.size) { exp(r[it]) } }
        val arg = p.map { argmax(it) }
        val n = p.size
        val starts = HashSet<Int>()
        val ends = HashSet<Int>()
        var t = 0
        while (t < n) {
            if (arg[t] == B || arg[t] == I) {
                var u = t
                while (u + 1 < n && (arg[u + 1] == B || arg[u + 1] == I)) u += 1
                starts.add(t); ends.add(u)
                t = u + 1
            } else t += 1
        }
        for (k in 0 until n) {
            val pb = p[k][B]
            val left = if (k > 0) p[k - 1][B] else 0f
            val right = if (k + 1 < n) p[k + 1][B] else 0f
            if (pb >= bThreshold && pb >= left && pb >= right) {
                starts.add(k)
                if (k > 0) ends.add(k - 1)
            }
        }
        val oRun = IntArray(n)
        var running = 0
        for (k in 0 until n) { running += if (arg[k] == O) 1 else 0; oRun[k] = running }
        val spans = ArrayList<SpanKey>()
        val sortedEnds = ends.sorted()
        for (s in starts.sorted()) for (e in sortedEnds) {
            val length = e - s + 1
            if (length < minLen || length > maxLen) continue
            if (oRun[e] - (if (s > 0) oRun[s - 1] else 0) > maxInnerO) continue
            spans.add(SpanKey(s, e))
        }
        return spans
    }

    fun denseSpans(bio: List<FloatArray>, step: Int = 2, minLen: Int = 3, maxLen: Int = 20, active: Float = 0.3f,
                   maxInnerO: Int = 3): List<SpanKey> {
        val n = bio.size
        val on = BooleanArray(n) { exp(bio[it][B]) + exp(bio[it][I]) >= active }
        val oRun = IntArray(n)
        var running = 0
        for (t in 0 until n) { running += if (on[t]) 0 else 1; oRun[t] = running }
        val points = (0 until n).filter { on[it] }
        if (points.isEmpty()) return emptyList()
        val grid = HashSet<Int>()
        var i = 0
        while (i < points.size) { grid.add(points[i]); i += step }
        for (t in points) if (t == 0 || !on[t - 1]) grid.add(t)
        for (t in points) if (t == n - 1 || !on[t + 1]) grid.add(t)
        val sorted = grid.sorted()
        val spans = ArrayList<SpanKey>()
        for (s in sorted) for (e in sorted) {
            val length = e - s + 1
            if (length < minLen || length > maxLen) continue
            if (oRun[e] - (if (s > 0) oRun[s - 1] else 0) > maxInnerO) continue
            spans.add(SpanKey(s, e))
        }
        return spans
    }

    fun bothSpans(bio: List<FloatArray>, minLen: Int = 3, maxInnerO: Int = 3): List<SpanKey> =
        (candidateSpans(bio, minLen = minLen, maxInnerO = maxInnerO).toSet() +
            denseSpans(bio, minLen = minLen, maxInnerO = maxInnerO)).sorted()

    class Segment(val start: Int, val end: Int, val gloss: String?, val logQ: Double)

    private class Candidate(val s: Int, val seg: Double, val gloss: String?, val lq: Double)

    /** Semi-Markov DP over rest frames and labelled segments (no duplicate merge: streaming). */
    fun decode(bio: List<FloatArray>, spans: List<SpanKey>, records: Map<SpanKey, FloatArray>, labels: List<String>,
               config: LiveDecoderConfig): List<Segment> {
        val n = bio.size
        val act = FloatArray(n)
        val rest = FloatArray(n)
        val logb = FloatArray(n)
        for (t in 0 until n) {
            val pU = exp(bio[t][UNK]); val pO = exp(bio[t][O]); val pB = exp(bio[t][B]); val pI = exp(bio[t][I])
            act[t] = ln(min(max(pB + pI, config.innerFloor ?: 1e-6f), 1f))
            rest[t] = ln(min(max(pO + pU, 1e-6f), 1f))
            logb[t] = ln(min(max(pB, 1e-6f), 1f))
        }
        val cact = DoubleArray(n + 1)
        var running = 0f   // numpy cumsum of float32 accumulates in float32
        for (t in 0 until n) { running += act[t]; cact[t + 1] = running.toDouble() }
        val byEnd = HashMap<Int, ArrayList<Candidate>>()
        for (span in spans) {
            val record = records[span] ?: continue
            val q = logSoftmax(record)
            var g = 0
            for (k in 1 until q.size) if (q[k] > q[g]) g = k
            val lq = q[g]
            val emit = lq >= config.logTheta
            val seg = config.wR * max(lq, config.logTheta) + config.wA * (cact[span.e + 1] - cact[span.s]) +
                config.wB * logb[span.s].toDouble() - config.c
            byEnd.getOrPut(span.e) { ArrayList() }.add(Candidate(span.s, seg, if (emit) labels[g] else null, lq))
        }
        val best = DoubleArray(n + 1) { Double.NEGATIVE_INFINITY }
        val backSign = BooleanArray(n + 1)
        val backStart = IntArray(n + 1)
        val backGloss = arrayOfNulls<String>(n + 1)
        val backLq = DoubleArray(n + 1)
        best[0] = 0.0
        for (t in 1..n) {
            best[t] = best[t - 1] + config.wA * rest[t - 1].toDouble()
            backSign[t] = false; backStart[t] = t - 1
            for (c in byEnd[t - 1] ?: emptyList()) {
                val v = best[c.s] + c.seg
                if (v > best[t]) { best[t] = v; backSign[t] = true; backStart[t] = c.s; backGloss[t] = c.gloss; backLq[t] = c.lq }
            }
        }
        val out = ArrayList<Segment>()
        var t = n
        while (t > 0) {
            if (!backSign[t]) t -= 1
            else { out.add(Segment(backStart[t], t - 1, backGloss[t], backLq[t])); t = backStart[t] }
        }
        out.reverse()
        return out
    }
}

/** Streaming words-only runtime (LiveSegmentalRuntime, mode .words). */
class LiveSegmentalRuntime(
    val boundary: LiveBoundaryStream,
    val recognizer: LiveSpanRecognizer,
    val encoder: LiveHandEncoder?,          // null = landmark-only mode (no hand images)
    val config: LiveDecoderConfig,
) {
    val labels: List<String> = recognizer.labels
    val timing = HashMap<String, Double>()

    private var off = 0
    private var count = 0
    private val obs = ArrayList<LiveObservation>()
    private val hands = ArrayList<LiveHandFrame>()
    private val bio = ArrayList<FloatArray>()
    private var bioOff = 0
    private val scores = HashMap<SpanKey, FloatArray?>()
    var origin = 0
        private set
    private var last: LiveWord? = null
    private var early = HashMap<Int, Pair<String, Int>>()
    private val shownEarly = ArrayList<Pair<Int, LiveWord>>()
    private var lastSeconds = Double.NEGATIVE_INFINITY
    var preview: LivePreview? = null
        private set

    init { reset() }

    fun reset() {
        boundary.reset()
        off = 0; count = 0
        obs.clear(); hands.clear(); bio.clear(); bioOff = 0
        scores.clear(); origin = 0; last = null
        early = HashMap(); shownEarly.clear()
        lastSeconds = Double.NEGATIVE_INFINITY
        preview = null
    }

    private fun tick(name: String, started: Long): Long {
        val now = System.nanoTime()
        timing[name] = (timing[name] ?: 0.0) + (now - started) / 1e9
        return now
    }

    private fun spanObs(s: Int, e: Int): List<Int> {
        val lo = obs[s - off].seconds - 0.1
        val hi = obs[e - off].seconds + 0.1
        val from = max(0, s - off - 3)
        val to = min(obs.size, e - off + 4)
        if (from >= to) return emptyList()
        return (from until to).filter { lo - 1e-9 <= obs[it].seconds && obs[it].seconds <= hi + 1e-9 }
    }

    private fun score(spans: List<SpanKey>) {
        var t = System.nanoTime()
        val batch = ArrayList<LiveSpanInput>()
        val keys = ArrayList<SpanKey>()
        for (key in spans) {
            val ids = spanObs(key.s, key.e)
            val x = if (ids.size >= 4) LiveSpanRecognizer.inputs(ids.map { obs[it] }, ids.map { hands[it] }) else null
            if (x == null) { scores[key] = null; continue }
            batch.add(x); keys.add(key)
        }
        t = tick("span_inputs", t)
        val logits = if (batch.isEmpty()) emptyList() else recognizer.logits(batch)
        tick("recognizer", t)
        for ((key, row) in keys.zip(logits)) scores[key] = row
    }

    /** Add one observation; `image` is the frame for hand crops (ignored in landmark-only mode). */
    fun observe(observation: LiveObservation, image: LiveRGBAImage?): List<LiveWord> {
        val words = ArrayList<LiveWord>()
        if (observation.seconds - lastSeconds > 0.26 && count > 0) {
            words.addAll(finish())
            reset()
        }
        lastSeconds = observation.seconds
        obs.add(observation)
        var t = System.nanoTime()
        hands.add(if (encoder != null && image != null) encoder.frameHand(observation, image) else LiveHandFrame())
        t = tick("hand_crops_encode", t)
        count += 1
        boundary.update(LiveFeatures.raw(observation), observation.seconds)?.let { bio.add(it) }
        tick("boundary", t)
        words.addAll(step(false))
        return words
    }

    /** Flush at stop/finish: remaining frames get boundary estimates with the future masked. */
    fun finish(): List<LiveWord> {
        if (count <= 0) return emptyList()
        val pending = boundary.flush()
        val missing = count - (bioOff + bio.size)
        if (missing > 0) bio.addAll(pending.takeLast(missing))
        return step(true)
    }

    private fun trim() {
        val keep = max(origin - 40, 0)
        var drop = keep - off
        if (drop > 0) { repeat(drop) { obs.removeAt(0); hands.removeAt(0) }; off = keep }
        drop = keep - bioOff
        if (drop > 0) { repeat(drop) { bio.removeAt(0) }; bioOff = keep }
        scores.keys.removeAll { it.s < origin }
    }

    private fun seconds(k: Int): Double = obs[min(max(k - off, 0), obs.size - 1)].seconds

    private fun word(gloss: String, start: Int, end: Int, T: Int, logQ: Double, early: Boolean = false) =
        LiveWord(gloss, start, end, T, seconds(start), seconds(end), seconds(T), exp(logQ), early)

    private fun step(final: Boolean): List<LiveWord> {
        val T = count - 1
        val known = bioOff + bio.size - 1
        if (known < origin || bio.isEmpty()) return emptyList()
        val origin = this.origin
        val sub = bio.subList(origin - bioOff, known - bioOff + 1).toList()
        val spans = LiveSegmental.bothSpans(sub, config.minSpanFrames, config.maxInnerO).map { SpanKey(it.s + origin, it.e + origin) }
            .filter { final || it.e + config.contextFrames <= T }
        val missing = spans.filter { !scores.containsKey(it) }
        if (missing.isNotEmpty()) score(missing)
        val local = HashMap<SpanKey, FloatArray>()
        for (span in spans) scores[span]?.let { local[SpanKey(span.s - origin, span.e - origin)] = it }
        val localSpans = spans.map { SpanKey(it.s - origin, it.e - origin) }
        val segs = LiveSegmental.decode(sub, localSpans, local, labels, config)
        val horizon = if (final) known else known - config.lag
        val committed = ArrayList<LiveWord>()
        preview = null
        for (g in segs) {
            if (g.end + origin <= horizon) continue
            val st = g.start + origin
            val en = g.end + origin
            if (shownEarly.any { abs(st - it.first) <= 2 }) break
            scores[SpanKey(st, en)]?.let { raw ->
                val lp = LiveSegmental.logSoftmax(raw)
                val p = DoubleArray(lp.size) { exp(lp[it]) }
                val order = p.indices.sortedWith { a, b -> if (p[a] != p[b]) p[b].compareTo(p[a]) else a.compareTo(b) }.take(3)
                preview = LivePreview(labels[order.first()], p[order.first()], g.gloss != null, order.map { labels[it] to p[it] })
            }
            break
        }
        val gap = config.duplicateGap
        if (config.earlyK > 0 && !final) {
            val next = HashMap<Int, Pair<String, Int>>()
            for (g in segs) {
                val gloss = g.gloss
                if (g.end + origin <= horizon || gloss == null) continue
                val st = g.start + origin
                val previous = early[st]
                val n = if (previous != null && previous.first == gloss) previous.second + 1 else 1
                next[st] = gloss to n
                if (n >= config.earlyK && exp(g.logQ) >= config.earlyQ && gloss !in config.earlyUnsafe &&
                    g.end - g.start + 1 >= config.earlyMin && shownEarly.none { abs(st - it.first) <= 2 }) {
                    val l = last
                    if (l != null && l.gloss == gloss && st - l.endFrame <= gap) {
                        shownEarly.add(st to l)
                        continue
                    }
                    val w = word(gloss, st, g.end + origin, T, g.logQ, early = true)
                    last = w
                    shownEarly.add(st to w)
                    committed.add(w)
                }
            }
            early = next
        }
        var newOrigin = origin
        var pending = horizon + 1
        for (g in segs) {
            val end = g.end + origin
            if (end > horizon) { pending = g.start + origin; break }
            if (!final && config.softExtra > 0 && end + 1 <= known) {
                val r = bio[end + 1 - bioOff]
                if (exp(r[O]) + exp(r[UNK]) + exp(r[B]) < config.hard && end > horizon - config.softExtra) {
                    pending = g.start + origin
                    break
                }
            }
            val start = g.start + origin
            val shown = shownEarly.firstOrNull { abs(start - it.first) <= 2 }?.second
            if (shown != null) {
                shown.endFrame = max(shown.endFrame, end)
                shown.endSeconds = seconds(end)
                shown.closed = true
                newOrigin = end + 1
                continue
            }
            val gloss = g.gloss
            if (gloss != null) {
                val l = last
                if (l != null && l.gloss == gloss && start - l.endFrame <= gap) {
                    l.endFrame = end
                } else {
                    val w = word(gloss, start, end, T, g.logQ)
                    last = w
                    committed.add(w)
                }
            }
            newOrigin = end + 1
        }
        this.origin = max(newOrigin, min(pending, horizon + 1 - 30))
        shownEarly.removeAll { it.first < this.origin - 2 }
        trim()
        return committed
    }
}

/** Split an utterance at long pauses (older Stage 3 models). */
fun liveSplitClauses(words: List<LiveWord>, pauseSeconds: Double = 1.5): List<List<LiveWord>> {
    val clauses = ArrayList<ArrayList<LiveWord>>()
    for (w in words) {
        val lastClause = clauses.lastOrNull()
        if (lastClause != null && w.startSeconds - lastClause.last().endSeconds >= pauseSeconds) clauses.add(arrayListOf(w))
        else if (lastClause == null) clauses.add(arrayListOf(w))
        else lastClause.add(w)
    }
    return clauses
}
