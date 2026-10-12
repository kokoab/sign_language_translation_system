// MediaPipe Tasks detector for the Android family; the same contract as active/v17/mediapipe_full_v17.py:
// hands every frame (VIDEO mode, GPU, 0.5 thresholds, handedness label as chirality, joints outside the
// image missing, >= 5 joints, handedness score as confidence); pose 11-14 and 15 face-mesh points every
// 8th processed frame (IMAGE mode; visibility >= 0.5 and in-image for pose; face confidence 1.0).
package com.example.slt_mobile_app.live

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import kotlin.math.sqrt

class LiveVision(context: Context, useGpu: Boolean = true) {
    companion object {
        const val AUXILIARY_INTERVAL = 8
        val POSE_INDICES = intArrayOf(11, 12, 13, 14)
        val FACE_MESH_INDICES = intArrayOf(468, 473, 63, 46, 293, 276, 344, 40, 291, 0, 17, 264, 152, 34, 168)
        private const val HAND_THRESHOLD = 0.5f
        private const val POSE_VISIBILITY = 0.5f
    }

    private val delegate = if (useGpu) Delegate.GPU else Delegate.CPU
    private fun base(asset: String) = BaseOptions.builder().setModelAssetPath(asset).setDelegate(delegate).build()

    private val hands = HandLandmarker.createFromOptions(context, HandLandmarker.HandLandmarkerOptions.builder()
        .setBaseOptions(base("hand_landmarker.task")).setRunningMode(RunningMode.VIDEO).setNumHands(2)
        .setMinHandDetectionConfidence(HAND_THRESHOLD).setMinHandPresenceConfidence(HAND_THRESHOLD)
        .setMinTrackingConfidence(HAND_THRESHOLD).build())
    private val pose = PoseLandmarker.createFromOptions(context, PoseLandmarker.PoseLandmarkerOptions.builder()
        .setBaseOptions(base("pose_landmarker_lite.task")).setRunningMode(RunningMode.IMAGE).setNumPoses(1).build())
    private val face = FaceLandmarker.createFromOptions(context, FaceLandmarker.FaceLandmarkerOptions.builder()
        .setBaseOptions(base("face_landmarker.task")).setRunningMode(RunningMode.IMAGE).setNumFaces(1).build())

    private var previousWrists = HashMap<String, Pair<Float, Float>>()
    private var processed = 0
    private var lastTimestampMs = -1L
    /** VIDEO mode needs increasing timestamps across streams; each stream's clock restarts at 0. */
    private var streamBaseMs = 0L

    private var lastSize: Pair<Int, Int>? = null

    /**
     * New stream (Start / reset): slot history and the auxiliary cadence restart. Like Python
     * reset_sequence, a blank frame one second later drops hand tracking, and the clock moves on 1 s.
     */
    fun resetTracking() {
        previousWrists = HashMap()
        processed = 0
        val size = lastSize ?: return
        val blank = Bitmap.createBitmap(size.first, size.second, Bitmap.Config.ARGB_8888)
        lastTimestampMs += 1000
        hands.detectForVideo(BitmapImageBuilder(blank).build(), lastTimestampMs)
        lastTimestampMs += 1000
        streamBaseMs = lastTimestampMs
        lastSize = null
    }

    fun close() { hands.close(); pose.close(); face.close() }

    private fun inside(x: Float, y: Float) = x in 0f..1f && y in 0f..1f

    /** One upright, unmirrored frame. `seconds` is the camera clock (monotonic within a stream). */
    fun observe(image: MPImage, width: Int, height: Int, seconds: Double): LiveObservation {
        val auxiliary = processed % AUXILIARY_INTERVAL == 0
        processed += 1
        var timestamp = streamBaseMs + (seconds * 1000).toLong()
        if (timestamp <= lastTimestampMs) timestamp = lastTimestampMs + 1
        lastTimestampMs = timestamp
        lastSize = image.width to image.height
        val result = hands.detectForVideo(image, timestamp)
        val detected = ArrayList<LiveHand>()
        for (k in result.landmarks().indices) {
            val landmarks = result.landmarks()[k]
            val category = result.handedness().getOrNull(k)?.firstOrNull() ?: continue
            if (landmarks.size != 21) continue
            val score = category.score()
            val name = category.categoryName().lowercase()
            val hand = LiveHand(chirality = if (name == "left" || name == "right") name else "unknown", score = score)
            var valid = 0
            for (j in 0 until 21) {
                val p = landmarks[j]
                if (inside(p.x(), p.y())) { hand.points[j] = LivePoint(p.x(), p.y(), score); valid += 1 }
            }
            if (valid >= 5) detected.add(hand)
        }
        val body = Array(4) { LivePoint() }
        val faceRow = Array(15) { LivePoint() }
        if (auxiliary) {
            pose.detect(image).landmarks().firstOrNull()?.let { landmarks ->
                for ((slot, index) in POSE_INDICES.withIndex()) {
                    val p = landmarks[index]
                    val visibility = p.visibility().orElse(0f)
                    if (visibility >= POSE_VISIBILITY && inside(p.x(), p.y())) body[slot] = LivePoint(p.x(), p.y(), visibility)
                }
            }
            face.detect(image).faceLandmarks().firstOrNull()?.let { mesh ->
                for ((slot, index) in FACE_MESH_INDICES.withIndex()) {
                    val p = mesh[index]
                    if (inside(p.x(), p.y())) faceRow[slot] = LivePoint(p.x(), p.y(), 1f)
                }
            }
        }
        val assigned = assignHands(detected, previousWrists)
        for (slot in listOf("left", "right")) {
            val hand = assigned[slot] ?: continue
            if (hand.points[0].c > 0f) previousWrists[slot] = hand.points[0].x to hand.points[0].y
        }
        return LiveObservation(seconds, width, height, assigned["left"], assigned["right"], body, faceRow, auxiliary)
    }
}

private fun cost(hand: LiveHand, slot: String, previous: Pair<Float, Float>?): Float {
    var cost = -0.1f * hand.score
    if (hand.chirality != "unknown") {
        cost += if (hand.chirality == slot) 0f else 2f
    } else if (hand.points[0].c > 0f) {
        val expectedLeft = hand.points[0].x >= 0.5f
        if ((slot == "left") != expectedLeft) cost += 0.2f
    }
    if (previous != null && hand.points[0].c > 0f) {
        val dx = hand.points[0].x - previous.first
        val dy = hand.points[0].y - previous.second
        cost += sqrt(dx * dx + dy * dy)
    }
    return cost
}

/** assign_hands: up to two detections to left/right slots without relying on result order. */
fun assignHands(hands: List<LiveHand>, previous: Map<String, Pair<Float, Float>>): Map<String, LiveHand> {
    // Stable sort by score, descending (Python's sorted(..., reverse=True) keeps ties in order).
    val usable = hands.withIndex().sortedWith { a, b ->
        if (a.value.score != b.value.score) b.value.score.compareTo(a.value.score) else a.index.compareTo(b.index)
    }.map { it.value }.take(2)
    val first = usable.firstOrNull() ?: return emptyMap()
    if (usable.size == 1) {
        val left = cost(first, "left", previous["left"])
        val right = cost(first, "right", previous["right"])
        return mapOf((if (left <= right) "left" else "right") to first)
    }
    val second = usable[1]
    val direct = cost(first, "left", previous["left"]) + cost(second, "right", previous["right"])
    val swapped = cost(first, "right", previous["right"]) + cost(second, "left", previous["left"])
    return if (direct <= swapped) mapOf("left" to first, "right" to second) else mapOf("left" to second, "right" to first)
}

/** Two upright open palms held continuously for one second, once per release (desktop FinishGesture). */
class LiveFinishGesture {
    private val holdSeconds = 1.0
    private var startedAt: Double? = null
    var active = false
        private set
    private var latched = false
    var progress = 0.0
        private set

    fun reset() { startedAt = null; active = false; latched = false; progress = 0.0 }

    fun update(left: LiveHand?, right: LiveHand?, seconds: Double): Boolean {
        if (left == null || right == null || !isOpenPalm(left) || !isOpenPalm(right) ||
            length(left.points[0].x - right.points[0].x, left.points[0].y - right.points[0].y) < 0.12f) {
            reset(); return false
        }
        active = true
        if (latched) return false
        val start = startedAt ?: seconds.also { startedAt = it }
        progress = ((seconds - start) / holdSeconds).coerceIn(0.0, 1.0)
        if (progress < 1.0) return false
        latched = true
        return true
    }

    private fun length(x: Float, y: Float) = sqrt(x * x + y * y)

    private fun isOpenPalm(hand: LiveHand): Boolean {
        val required = intArrayOf(0, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20)
        if (!required.all { hand.points[it].c > 0f }) return false
        val p = hand.points
        val wx = p[0].x; val wy = p[0].y
        val palmX = p[9].x - wx; val palmY = p[9].y - wy
        val palmLength = length(palmX, palmY)
        if (palmLength < 0.025f) return false
        val ax = palmX / palmLength; val ay = palmY / palmLength
        if (!(ay <= -0.25f && wy <= 0.80f)) return false
        for ((mcp, pip, tip) in listOf(Triple(5, 6, 8), Triple(9, 10, 12), Triple(13, 14, 16), Triple(17, 18, 20))) {
            val px = p[pip].x - p[mcp].x; val py = p[pip].y - p[mcp].y
            val dx = p[tip].x - p[pip].x; val dy = p[tip].y - p[pip].y
            val scale = length(px, py) * length(dx, dy)
            if (!(scale > 1e-6f && (px * dx + py * dy) / scale > 0.45f &&
                    length(p[tip].x - wx, p[tip].y - wy) > 1.10f * length(p[pip].x - wx, p[pip].y - wy) &&
                    dx * ax + dy * ay > 0.005f)) return false
        }
        val px = p[3].x - p[2].x; val py = p[3].y - p[2].y
        val dx = p[4].x - p[3].x; val dy = p[4].y - p[3].y
        val scale = length(px, py) * length(dx, dy)
        return scale > 1e-6f && (px * dx + py * dy) / scale > 0.15f &&
            length(p[4].x - wx, p[4].y - wy) > 1.05f * length(p[3].x - wx, p[3].y - wy)
    }
}
