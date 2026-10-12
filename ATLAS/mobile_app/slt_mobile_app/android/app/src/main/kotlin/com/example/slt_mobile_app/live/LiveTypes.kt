// Live segmental Reel for Android (MediaPipe family): shared value types.
// Port of ios/Runner/LiveReel/LiveReelCore.swift + LiveReelDecoder.swift (words-only path), which are
// themselves ports of active/v17/segmental_runtime_v17.py and the live observation contract.
package com.example.slt_mobile_app.live

class LiveException(message: String) : Exception(message)

/** Image-normalized point (origin top-left) with confidence; c == 0 means missing. */
data class LivePoint(val x: Float = 0f, val y: Float = 0f, val c: Float = 0f)

class LiveHand(
    val points: Array<LivePoint> = Array(21) { LivePoint() },
    var chirality: String = "unknown",
    var score: Float = 0f,
)

/** One processed camera frame (the Python ObservedFrame without pixels). */
class LiveObservation(
    val seconds: Double,
    val width: Int,
    val height: Int,
    val left: LiveHand?,
    val right: LiveHand?,
    val body: Array<LivePoint>,          // 4: L/R shoulder, L/R elbow; zero unless detected this frame
    val face: Array<LivePoint>,          // 15; zero unless detected this frame
    val faceForFeatures: Boolean,
)

/** Per-frame hand-crop evidence: embeddings [3*512], valid [3], boxes [3*4] (left, right, union). */
class LiveHandFrame(
    val embeddings: FloatArray = FloatArray(3 * 512),
    val valid: FloatArray = FloatArray(3),
    val boxes: FloatArray = FloatArray(12),
)

/** A committed word; early-shown words are later updated in place with the sign's real end. */
class LiveWord(
    var gloss: String,
    var startFrame: Int, var endFrame: Int, var commitFrame: Int,
    var startSeconds: Double, var endSeconds: Double, var commitSeconds: Double,
    var score: Double, var early: Boolean,
) {
    var closed = false
    var computeSeconds = 0.0

    fun toMap(): Map<String, Any> = mapOf(
        "gloss" to gloss, "start_frame" to startFrame, "end_frame" to endFrame, "commit_frame" to commitFrame,
        "start_seconds" to startSeconds, "end_seconds" to endSeconds, "commit_seconds" to commitSeconds,
        "score" to score, "early" to early, "compute_seconds" to computeSeconds,
    )
}

data class LivePreview(val gloss: String, val score: Double, val emit: Boolean, val top3: List<Pair<String, Double>>)

data class SpanKey(val s: Int, val e: Int) : Comparable<SpanKey> {
    override fun compareTo(other: SpanKey): Int = if (s != other.s) s.compareTo(other.s) else e.compareTo(other.e)
}

/** Decoder + stream settings (stream_config_mediapipe_v2*.json; values of stream_config_v2). */
class LiveDecoderConfig(
    val wR: Double = 2.0, val wA: Double = 0.25, val wB: Double = 0.5, val c: Double = -1.5,
    val logTheta: Double = Math.log(0.5), val duplicateGap: Int = 20,
    val lag: Int = 1, val contextFrames: Int = 2, val earlyK: Int = 3, val earlyQ: Double = 0.9,
    val softExtra: Int = 2, val hard: Float = 0.5f, val earlyMin: Int = 4,
    val earlyUnsafe: Set<String> = emptySet(),
)

object LiveNodes {
    const val COUNT = 61
    const val FACE_START = 42
    const val BODY_START = 57
}
