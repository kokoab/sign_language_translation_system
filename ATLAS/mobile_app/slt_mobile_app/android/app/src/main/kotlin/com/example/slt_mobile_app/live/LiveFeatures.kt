// Per-frame features, span landmarks and cv2-compatible hand crops (port of LiveReelCore.swift).
// Float32 arithmetic and round-half-even are kept so values match the Python/Swift references.
package com.example.slt_mobile_app.live

import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

private fun dist(ax: Float, ay: Float, bx: Float, by: Float): Float {
    val dx = ax - bx
    val dy = ay - by
    return sqrt(dx * dx + dy * dy)
}

private fun clamp(v: Float, lo: Float, hi: Float) = min(max(v, lo), hi)

object LiveFeatures {
    const val BOUNDARY_DIMENSION = 450

    /** raw_observation_features for one frame: [61*5] (isotropic x, y, 0, present, confidence). */
    fun raw(o: LiveObservation): FloatArray {
        val raw = FloatArray(61 * 5)
        fun put(node: Int, p: LivePoint) {
            raw[node * 5] = p.x; raw[node * 5 + 1] = p.y; raw[node * 5 + 4] = p.c
        }
        o.left?.let { hand -> for (j in 0 until 21) put(j, hand.points[j]) }
        o.right?.let { hand -> for (j in 0 until 21) put(21 + j, hand.points[j]) }
        for (j in 0 until 4) put(57 + j, o.body[j])
        if (o.faceForFeatures) for (j in 0 until 15) put(42 + j, o.face[j])
        val w = o.width.toFloat()
        val h = o.height.toFloat()
        val longest = max(o.width, o.height).toFloat()
        for (node in 0 until 61) {
            val present = if (raw[node * 5 + 4] > 0f) 1f else 0f
            raw[node * 5 + 3] = present
            raw[node * 5] = (raw[node * 5] * w - w / 2f) / longest * present
            raw[node * 5 + 1] = (raw[node * 5 + 1] * h - h / 2f) / longest * present
        }
        return raw
    }

    /** boundary_features(raw, times, hand_geometry=True) for a short history; returns the last row. */
    fun boundaryRow(raws: List<FloatArray>, times: List<Double>): FloatArray {
        val n = raws.size
        val present = Array(n) { BooleanArray(61) }
        val nx = Array(n) { FloatArray(61) }
        val ny = Array(n) { FloatArray(61) }
        var cx = 0f
        var cy = 0f
        var scale = 0.3f
        var lastBody = Double.NEGATIVE_INFINITY
        for (i in 0 until n) {
            val raw = raws[i]
            for (node in 0 until 61) present[i][node] = raw[node * 5 + 3] > 0.5f && raw[node * 5 + 4] > 0f
            if (present[i][57] && present[i][58]) {
                val ax = raw[57 * 5]; val ay = raw[57 * 5 + 1]
                val bx = raw[58 * 5]; val by = raw[58 * 5 + 1]
                val width = dist(bx, by, ax, ay)
                if (width > 0.02f) { cx = (ax + bx) / 2f; cy = (ay + by) / 2f; scale = width; lastBody = times[i] }
            }
            if (times[i] - lastBody > 0.8) { cx = 0f; cy = 0f; scale = 0.3f }
            for (node in 0 until 61) {
                if (!present[i][node]) continue
                nx[i][node] = clamp((raw[node * 5] - cx) / scale, -4f, 4f)
                ny[i][node] = clamp((raw[node * 5 + 1] - cy) / scale, -4f, 4f)
            }
        }
        val last = n - 1
        val row = FloatArray(BOUNDARY_DIMENSION)
        for (node in 0 until 61) { row[node * 2] = nx[last][node]; row[node * 2 + 1] = ny[last][node] }
        if (n > 1) {
            val dt = (times[last] - times[last - 1]).toFloat()
            if (dt.toDouble() <= 0.26) {
                for (node in 0 until 61) {
                    if (!(present[last][node] && present[last - 1][node])) continue
                    row[122 + node * 2] = clamp((nx[last][node] - nx[last - 1][node]) / dt, -10f, 10f) / 10f
                    row[122 + node * 2 + 1] = clamp((ny[last][node] - ny[last - 1][node]) / dt, -10f, 10f) / 10f
                }
            }
        }
        val raw = raws[last]
        for (node in 0 until 61) {
            val p = if (present[last][node]) 1f else 0f
            row[244 + node] = p
            row[305 + node] = clamp(raw[node * 5 + 4], 0f, 1f) * p
        }
        for ((h, start) in listOf(0, 21).withIndex()) {
            val wx = raw[start * 5]; val wy = raw[start * 5 + 1]
            val palm = dist(raw[(start + 9) * 5], raw[(start + 9) * 5 + 1], wx, wy)
            if (!(present[last][start] && present[last][start + 9] && palm > 0.005f)) continue
            for (j in 0 until 21) {
                if (!present[last][start + j]) continue
                val d = max(palm, 0.005f)
                row[366 + h * 42 + j * 2] = clamp((raw[(start + j) * 5] - wx) / d, -4f, 4f)
                row[366 + h * 42 + j * 2 + 1] = clamp((raw[(start + j) * 5 + 1] - wy) / d, -4f, 4f)
            }
        }
        return row
    }
}

/** landmarks_from_observations: [32*61*5] span features plus the trim range. */
object LiveSpanLandmarks {
    const val FRAMES = 32

    private class Track { val x = FloatArray(61); val y = FloatArray(61); val c = FloatArray(61) }

    class Built(val features: FloatArray, val trimStart: Int, val trimEnd: Int)

    fun build(observations: List<LiveObservation>): Built? {
        if (observations.size < 4) return null
        val width = observations[0].width
        val height = observations[0].height
        if (!observations.all { it.width == width && it.height == height }) return null
        var tracks = observations.map { o ->
            val t = Track()
            o.left?.let { hand -> for (j in 0 until 21) { t.x[j] = hand.points[j].x; t.y[j] = hand.points[j].y; t.c[j] = hand.points[j].c } }
            o.right?.let { hand -> for (j in 0 until 21) { t.x[21 + j] = hand.points[j].x; t.y[21 + j] = hand.points[j].y; t.c[21 + j] = hand.points[j].c } }
            if (o.body.any { it.c > 0f }) for (j in 0 until 4) { t.x[57 + j] = o.body[j].x; t.y[57 + j] = o.body[j].y; t.c[57 + j] = o.body[j].c }
            if (o.faceForFeatures && o.face.any { it.c > 0f }) {
                for (j in 0 until 15) { t.x[42 + j] = o.face[j].x; t.y[42 + j] = o.face[j].y; t.c[42 + j] = o.face[j].c }
            }
            t
        }
        val active = tracks.indices.filter { i -> (0 until 42).count { tracks[i].c[it] > 0f } >= 5 }
        if (active.size < 2) return null
        val trimStart = max(0, active.first() - 2)
        val trimEnd = min(tracks.size, active.last() + 3)
        tracks = tracks.subList(trimStart, trimEnd)
        val w = width.toFloat()
        val h = height.toFloat()
        val longest = max(width, height).toFloat()
        for (t in tracks) for (node in 0 until 61) {
            if (t.c[node] > 0f) {
                t.x[node] = (t.x[node] * w - w / 2f) / longest
                t.y[node] = (t.y[node] * h - h / 2f) / longest
            } else { t.x[node] = 0f; t.y[node] = 0f }
        }
        interpolate(tracks, 0 until 42, 3)
        interpolate(tracks, 42 until 61, 16)
        return Built(resample(normalize(tracks), tracks.size), trimStart, trimEnd)
    }

    private fun interpolate(tracks: List<Track>, nodes: IntRange, maximumGap: Int) {
        for (node in nodes) {
            val valid = tracks.indices.filter { tracks[it].c[node] > 0f }
            for (k in 0 until valid.size - 1) {
                val left = valid[k]
                val right = valid[k + 1]
                val gap = right - left - 1
                if (gap <= 0 || gap > maximumGap) continue
                val sx = tracks[left].x[node]; val sy = tracks[left].y[node]
                val ex = tracks[right].x[node]; val ey = tracks[right].y[node]
                val confidence = min(tracks[left].c[node], tracks[right].c[node]) * 0.5f
                for (offset in 1..gap) {
                    val f = offset.toFloat() / (gap + 1).toFloat()
                    tracks[left + offset].x[node] = sx + f * (ex - sx)
                    tracks[left + offset].y[node] = sy + f * (ey - sy)
                    tracks[left + offset].c[node] = confidence
                }
            }
        }
    }

    private fun median(values: List<Float>): Float {
        if (values.isEmpty()) return 0f
        val s = values.sorted()
        val m = s.size / 2
        return if (s.size % 2 == 0) (s[m - 1] + s[m]) / 2f else s[m]
    }

    /** body_relative_normalize, returning per frame [61*5] (x, y, depth, present, confidence). */
    private fun normalize(tracks: List<Track>): List<FloatArray> {
        val n = tracks.size
        val widths = ArrayList<Float>()
        val known = ArrayList<Triple<Int, Float, Float>>()
        val shoulderValid = BooleanArray(n)
        val shoulderWidth = FloatArray(n)
        for (i in 0 until n) {
            val t = tracks[i]
            if (!(t.c[57] > 0f && t.c[58] > 0f)) continue
            val width = dist(t.x[58], t.y[58], t.x[57], t.y[57])
            if (width <= 1e-5f) continue
            shoulderValid[i] = true; shoulderWidth[i] = width
            widths.add(width)
            known.add(Triple(i, (t.x[57] + t.x[58]) / 2f, (t.y[57] + t.y[58]) / 2f))
        }
        val centerX = FloatArray(n)
        val centerY = FloatArray(n)
        var scale: Float
        if (known.isNotEmpty()) {
            for (i in 0 until n) {
                if (i <= known[0].first) { centerX[i] = known[0].second; centerY[i] = known[0].third; continue }
                if (i >= known.last().first) { centerX[i] = known.last().second; centerY[i] = known.last().third; continue }
                val r = known.indexOfFirst { it.first >= i }
                val a = known[r - 1]
                val b = known[r]
                val f = (i - a.first).toFloat() / (b.first - a.first).toFloat()
                centerX[i] = a.second + f * (b.second - a.second)
                centerY[i] = a.third + f * (b.third - a.third)
            }
            scale = median(widths)
        } else {
            val xs = ArrayList<Float>()
            val ys = ArrayList<Float>()
            for (start in listOf(0, 21)) for (t in tracks) if (t.c[start] > 0f) { xs.add(t.x[start]); ys.add(t.y[start]) }
            val fx = if (xs.isEmpty()) 0f else median(xs)
            val fy = if (xs.isEmpty()) 0f else median(ys)
            centerX.fill(fx); centerY.fill(fy)
            val palms = ArrayList<Float>()
            for (start in listOf(0, 21)) for (t in tracks) {
                if (t.c[start] > 0f && t.c[start + 9] > 0f) {
                    val length = dist(t.x[start + 9], t.y[start + 9], t.x[start], t.y[start])
                    if (length > 1e-5f) palms.add(length)
                }
            }
            scale = if (palms.isEmpty()) 1f else median(palms)
        }
        if (!scale.isFinite() || scale <= 1e-5f) scale = 1f
        val out = List(n) { FloatArray(61 * 5) }
        for (i in 0 until n) for (node in 0 until 61) {
            val t = tracks[i]
            if (t.c[node] <= 0f) continue
            out[i][node * 5] = (t.x[node] - centerX[i]) / scale
            out[i][node * 5 + 1] = (t.y[node] - centerY[i]) / scale
            out[i][node * 5 + 3] = 1f
            out[i][node * 5 + 4] = clamp(t.c[node], 0f, 1f)
        }
        for (start in listOf(0, 21)) {
            val lengths = FloatArray(n)
            val valid = BooleanArray(n)
            for (i in 0 until n) {
                val t = tracks[i]
                if (t.c[start] > 0f && t.c[start + 9] > 0f) {
                    lengths[i] = dist(t.x[start + 9], t.y[start + 9], t.x[start], t.y[start])
                    valid[i] = lengths[i] > 1e-5f
                }
            }
            val reference = median((0 until n).filter { valid[it] }.map { lengths[it] })
            if (!valid.any { it }) continue
            for (i in 0 until n) {
                if (!valid[i]) continue
                val depth = ln(reference / lengths[i])
                for (node in start until start + 21) if (out[i][node * 5 + 3] > 0f) out[i][node * 5 + 2] = depth
            }
        }
        if (widths.isNotEmpty()) {
            val reference = median(widths)
            for (i in 0 until n) {
                if (!shoulderValid[i]) continue
                val depth = ln(reference / shoulderWidth[i])
                for (node in 42 until 61) if (out[i][node * 5 + 3] > 0f) out[i][node * 5 + 2] = depth
            }
        }
        return out
    }

    /** resample_features to 32 frames: linear for x/y/depth/confidence, nearest for presence. */
    private fun resample(input: List<FloatArray>, count: Int): FloatArray {
        val out = FloatArray(FRAMES * 61 * 5)
        for (target in 0 until FRAMES) {
            val position = if (count == 1) 0.0 else target.toDouble() / (FRAMES - 1).toDouble() * (count - 1).toDouble()
            val left = min(floor(position).toInt(), count - 1)
            val right = min(left + 1, count - 1)
            val fraction = (position - left).toFloat()
            val nearest = Math.rint(position).toInt()
            for (node in 0 until 61) {
                val presence = if (input[nearest][node * 5 + 3] >= 0.5f) 1f else 0f
                val base = (target * 61 + node) * 5
                out[base + 3] = presence
                if (presence <= 0f) continue
                for (channel in intArrayOf(0, 1, 2, 4)) {
                    val a = input[left][node * 5 + channel]
                    val b = input[right][node * 5 + channel]
                    out[base + channel] = a + fraction * (b - a)
                }
            }
        }
        return out
    }
}

/** An upright, unmirrored RGBA8888 frame in memory. */
class LiveRGBAImage(val bytes: ByteArray, val width: Int, val height: Int, val rowStride: Int)

/** hand_box / union_box / crop_square (reflect-101; INTER_AREA shrink, INTER_CUBIC enlarge). */
object LiveHandCrops {
    const val CROP_SIZE = 256
    private const val HAND_BOX_SCALE = 1.70
    private const val UNION_BOX_SCALE = 1.20f
    private const val MINIMUM_BOX_FRACTION = 0.14

    fun handBox(hand: LiveHand, width: Int, height: Int): FloatArray? {
        val valid = hand.points.filter { it.c > 0f }
        if (valid.size < 5) return null
        val x0 = valid.minOf { it.x }.toDouble(); val x1 = valid.maxOf { it.x }.toDouble()
        val y0 = valid.minOf { it.y }.toDouble(); val y1 = valid.maxOf { it.y }.toDouble()
        val cx = 0.5 * (x0 + x1) * width
        val cy = 0.5 * (y0 + y1) * height
        val detected = max((x1 - x0) * width, (y1 - y0) * height)
        val side = max(detected * HAND_BOX_SCALE, MINIMUM_BOX_FRACTION * max(width, height))
        return floatArrayOf((cx - side / 2).toFloat(), (cy - side / 2).toFloat(), (cx + side / 2).toFloat(), (cy + side / 2).toFloat())
    }

    fun unionBox(boxes: List<FloatArray>): FloatArray? {
        if (boxes.isEmpty()) return null
        val x0 = boxes.minOf { it[0] }; val y0 = boxes.minOf { it[1] }
        val x1 = boxes.maxOf { it[2] }; val y1 = boxes.maxOf { it[3] }
        val cx = 0.5f * (x0 + x1); val cy = 0.5f * (y0 + y1)
        val side = max(max(x1 - x0, y1 - y0) * UNION_BOX_SCALE, 1f)
        return floatArrayOf(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    }

    class Crops(val valid: FloatArray, val boxes: FloatArray, val crops: Array<ByteArray?>)

    /** Boxes and 256x256 RGB crops for the left, right and union views of one frame. */
    fun crops(o: LiveObservation, image: LiveRGBAImage): Crops {
        val valid = FloatArray(3)
        val boxes = FloatArray(12)
        val crops = arrayOfNulls<ByteArray>(3)
        val seen = ArrayList<FloatArray>()
        val w = o.width.toFloat(); val h = o.height.toFloat()
        for ((view, hand) in listOf(o.left, o.right).withIndex()) {
            if (hand == null) continue
            val box = handBox(hand, o.width, o.height) ?: continue
            valid[view] = 1f
            boxes[view * 4] = box[0] / w; boxes[view * 4 + 1] = box[1] / h; boxes[view * 4 + 2] = box[2] / w; boxes[view * 4 + 3] = box[3] / h
            seen.add(box)
            crops[view] = cropSquare(image, box)
        }
        unionBox(seen)?.let { box ->
            valid[2] = 1f
            boxes[8] = box[0] / w; boxes[9] = box[1] / h; boxes[10] = box[2] / w; boxes[11] = box[3] / h
            crops[2] = cropSquare(image, box)
        }
        return Crops(valid, boxes, crops)
    }

    private fun reflect101(p0: Int, n: Int): Int {
        if (n <= 1) return 0
        var p = p0
        while (p < 0 || p >= n) p = if (p < 0) -p else 2 * n - 2 - p
        return p
    }

    fun cropSquare(image: LiveRGBAImage, box: FloatArray): ByteArray {
        val r = IntArray(4) { Math.rint(box[it].toDouble()).toInt() }
        val cw = max(r[2] - r[0], 1); val ch = max(r[3] - r[1], 1)
        val xs = IntArray(cw) { reflect101(r[0] + it, image.width) }
        val ys = IntArray(ch) { reflect101(r[1] + it, image.height) }
        return if (max(cw, ch) > CROP_SIZE) resizeArea(image, xs, ys) else resizeCubic(image, xs, ys)
    }

    private fun pixel(image: LiveRGBAImage, x: Int, y: Int, rgb: Int): Int =
        image.bytes[y * image.rowStride + x * 4 + rgb].toInt() and 0xFF

    /** cv2 INTER_CUBIC (A = -0.75) with 11-bit fixed-point coefficients, replicate at the crop edge. */
    private fun resizeCubic(image: LiveRGBAImage, xs: IntArray, ys: IntArray): ByteArray {
        val size = CROP_SIZE
        fun table(source: Int): Pair<IntArray, IntArray> {
            val scale = 1.0 / (size.toDouble() / source.toDouble())
            val index = IntArray(size * 4)
            val coef = IntArray(size * 4)
            for (d in 0 until size) {
                var f = ((d + 0.5) * scale - 0.5).toFloat()
                val s = floor(f).toInt()
                f -= s.toFloat()
                val a = -0.75f
                val c = FloatArray(4)
                c[0] = ((a * (f + 1) - 5 * a) * (f + 1) + 8 * a) * (f + 1) - 4 * a
                c[1] = ((a + 2) * f - (a + 3)) * f * f + 1
                c[2] = ((a + 2) * (1 - f) - (a + 3)) * (1 - f) * (1 - f) + 1
                c[3] = 1 - c[0] - c[1] - c[2]
                for (k in 0 until 4) {
                    index[d * 4 + k] = min(max(s + k - 1, 0), source - 1)
                    coef[d * 4 + k] = Math.rint((c[k] * 2048f).toDouble()).toInt()
                }
            }
            return index to coef
        }
        val (txI, txC) = table(xs.size)
        val (tyI, tyC) = table(ys.size)
        val rows = IntArray(ys.size * size * 3)
        for (sy in tyI.toSet()) {
            val y = ys[sy]
            for (d in 0 until size) for (rgb in 0 until 3) {
                var sum = 0
                for (k in 0 until 4) sum += pixel(image, xs[txI[d * 4 + k]], y, rgb) * txC[d * 4 + k]
                rows[(sy * size + d) * 3 + rgb] = sum
            }
        }
        val out = ByteArray(size * size * 3)
        for (d in 0 until size) for (x in 0 until size) for (rgb in 0 until 3) {
            var sum = 0L
            for (k in 0 until 4) sum += rows[(tyI[d * 4 + k] * size + x) * 3 + rgb].toLong() * tyC[d * 4 + k].toLong()
            val value = (sum + (1L shl 21)) shr 22
            out[(d * size + x) * 3 + rgb] = value.coerceIn(0, 255).toInt().toByte()
        }
        return out
    }

    /** cv2 INTER_AREA for a shrinking resize (computeResizeAreaTab weights, float accumulation). */
    private fun resizeArea(image: LiveRGBAImage, xs: IntArray, ys: IntArray): ByteArray {
        val size = CROP_SIZE
        fun table(source: Int): List<List<Pair<Int, Float>>> {
            val scale = 1.0 / (size.toDouble() / source.toDouble())
            return (0 until size).map { d ->
                val taps = ArrayList<Pair<Int, Float>>()
                val fsx1 = d * scale
                val fsx2 = fsx1 + scale
                val cell = min(scale, source - fsx1)
                var sx1 = ceil(fsx1).toInt()
                var sx2 = floor(fsx2).toInt()
                sx2 = min(sx2, source - 1)
                sx1 = min(sx1, sx2)
                if (sx1 - fsx1 > 1e-3) taps.add((sx1 - 1) to ((sx1 - fsx1) / cell).toFloat())
                for (sx in sx1 until sx2) taps.add(sx to (1 / cell).toFloat())
                if (fsx2 - sx2 > 1e-3) taps.add(sx2 to (min(min(fsx2 - sx2, 1.0), cell) / cell).toFloat())
                taps
            }
        }
        val tx = table(xs.size)
        val ty = table(ys.size)
        val rows = FloatArray(ys.size * size * 3)
        val needed = ty.flatMap { taps -> taps.map { it.first } }.toSet()
        for (sy in needed) {
            val y = ys[sy]
            for (d in 0 until size) {
                var r = 0f; var g = 0f; var b = 0f
                for ((sx, weight) in tx[d]) {
                    val x = xs[sx]
                    r += weight * pixel(image, x, y, 0).toFloat()
                    g += weight * pixel(image, x, y, 1).toFloat()
                    b += weight * pixel(image, x, y, 2).toFloat()
                }
                rows[(sy * size + d) * 3] = r; rows[(sy * size + d) * 3 + 1] = g; rows[(sy * size + d) * 3 + 2] = b
            }
        }
        val out = ByteArray(size * size * 3)
        for (d in 0 until size) for (x in 0 until size) for (rgb in 0 until 3) {
            var sum = 0f
            for ((sy, weight) in ty[d]) sum += weight * rows[(sy * size + x) * 3 + rgb]
            out[(d * size + x) * 3 + rgb] = Math.rint(sum.toDouble()).toInt().coerceIn(0, 255).toByte()
        }
        return out
    }
}
