// One live segmental Reel on Android (words only): MediaPipe -> segmental decoder; plus the shared
// model holder, the hand-images mode choice and the session store used by the Flutter channel.
// Port of ios/Runner/LiveReel/LiveReelEngine.swift (word mode) and LiveReelApp.swift.
package com.example.slt_mobile_app.live

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.google.mediapipe.framework.image.MPImage
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.concurrent.Executors
import kotlin.math.ln

class LiveEngine(context: Context, val handImages: Boolean, useGpu: Boolean = true) {
    companion object {
        const val FPS = 20.0
        const val DETECTION_SIDE = 640

        fun configAsset(handImages: Boolean) = if (handImages) "live_config_hand_images.json" else "live_config_landmark_only.json"
        fun recognizerAsset(handImages: Boolean) =
            if (handImages) "span_recognizer_mediapipe_b8_fp32.tflite" else "span_recognizer_landmark_only_b8_fp32.tflite"
        const val ENCODER_ASSET = "mobileclip2_s0_image_fp32.tflite"

        fun loadConfig(context: Context, asset: String): LiveDecoderConfig {
            val json = JSONObject(context.assets.open(asset).bufferedReader().use { it.readText() })
            val d = json.getJSONObject("decoder")
            val s = json.getJSONObject("stream")
            val unsafe = s.optJSONArray("early_unsafe") ?: JSONArray()
            return LiveDecoderConfig(
                wR = d.getDouble("w_r"), wA = d.getDouble("w_a"), wB = d.getDouble("w_b"), c = d.getDouble("c"),
                logTheta = ln(d.getDouble("theta")), duplicateGap = d.getInt("duplicate_gap"),
                lag = s.getInt("lag"), contextFrames = s.getInt("context_frames"), earlyK = s.getInt("early_k"),
                earlyQ = s.getDouble("early_q"), softExtra = s.getInt("soft_extra"),
                earlyUnsafe = (0 until unsafe.length()).map { unsafe.getString(it) }.toSet(),
            )
        }

        fun loadLabels(context: Context): List<String> {
            val array = JSONArray(context.assets.open("live_reel_labels.json").bufferedReader().use { it.readText() })
            return List(array.length()) { array.getString(it) }.also {
                if (it.size != 100) throw LiveException("Expected 100 labels, found ${it.size}")
            }
        }
    }

    val labels = loadLabels(context)
    val config = loadConfig(context, configAsset(handImages))
    val vision = LiveVision(context, useGpu)
    private val boundaryModel = LiveBoundaryModel(context, "boundary_mediapipe_fp32.tflite", lookahead = 6)
    private val recognizer = LiveSpanRecognizer(context, recognizerAsset(handImages), labels)
    private val encoder = if (handImages) LiveHandEncoder(context, ENCODER_ASSET) else null
    val runtime = LiveSegmentalRuntime(LiveBoundaryStream(boundaryModel), recognizer, encoder, config)
    private val finishGesture = LiveFinishGesture()

    init {
        if (recognizer.usesHandImages != handImages) throw LiveException("Recognizer graph does not match the hand-images mode")
    }

    class Step(
        val observation: LiveObservation,
        val committed: List<LiveWord>,
        val computeSeconds: Double,
        val finishRequested: Boolean,
        val finishProgress: Double?,
    )

    val preview: LivePreview? get() = runtime.preview

    /** First interpreter calls allocate and plan; do them before the camera starts. */
    fun warm() {
        val zero = LiveSpanInput(FloatArray(32 * 61 * 5), FloatArray(16 * 3 * 512), FloatArray(48), FloatArray(192))
        recognizer.logits(listOf(zero))
        encoder?.embed(ByteArray(256 * 256 * 3))
        boundaryModel.logits(List(64) { FloatArray(LiveFeatures.BOUNDARY_DIMENSION) }, BooleanArray(64) { true })
    }

    fun resetFinishGesture() = finishGesture.reset()

    fun reset() {
        resetFinishGesture()
        runtime.reset()
        vision.resetTracking()
    }

    /**
     * One upright, unmirrored frame. `image` is the MediaPipe view of the same pixels as `rgba`
     * (longest side <= DETECTION_SIDE); `rgba` feeds the hand crops in hand-images mode.
     */
    fun process(image: MPImage, rgba: LiveRGBAImage, seconds: Double, allowFinishGesture: Boolean): Step {
        val started = SystemClock.elapsedRealtimeNanos()
        val observation = vision.observe(image, rgba.width, rgba.height, seconds)
        if (!allowFinishGesture) resetFinishGesture()
        val finishRequested = allowFinishGesture && finishGesture.update(observation.left, observation.right, seconds)
        if (finishGesture.active) {
            // Preserve the pending sign for Finish; never classify the control pose.
            return Step(observation, emptyList(), (SystemClock.elapsedRealtimeNanos() - started) / 1e9,
                finishRequested, finishGesture.progress)
        }
        val words = runtime.observe(observation, if (handImages) rgba else null)
        val spent = (SystemClock.elapsedRealtimeNanos() - started) / 1e9
        for (w in words) w.computeSeconds = spent
        return Step(observation, words, spent, false, null)
    }

    /** Flush at Finish/Stop: the open segment is committed. */
    fun finish(): List<LiveWord> = runtime.finish().also { runtime.reset() }

    fun close() = vision.close()
}

/** Hand-images mode: AUTO follows the first-launch speed test; ON/OFF are the manual override. */
enum class LiveHandImageMode { AUTO, ON, OFF }

/**
 * Process-wide models (LiveReelShared on iOS): the engine and Stage 3 are built once in the
 * background behind the home page; the live screen waits for `status`.
 */
object LiveShared {
    private const val TAG = "SLTLive"
    private const val PREFS = "slt_live"
    private const val KEY_MODE = "hand_image_mode"
    private const val KEY_AUTO_MS = "hand_crop_encode_ms"
    private const val KEY_AUTO_DEVICE = "hand_crop_encode_device"
    /**
     * Hand-images mode adds up to three MobileCLIP crops per frame. At 20 Hz a frame has 50 ms; hand
     * landmarks (~20 ms, Huawei nova Y70 GPU) and the boundary student (~9 ms) overlap with them only
     * partly, so three crops must fit in ~36 ms: 12 ms each. (The nova Y70 measured 193 ms per crop.)
     */
    const val AUTO_CROP_BUDGET_MS = 12.0

    /**
     * Inference threads run at display priority (XNNPACK's worker threads inherit it), so the scheduler
     * keeps the per-frame work on fast cores instead of treating it as background load.
     */
    const val INFERENCE_PRIORITY = android.os.Process.THREAD_PRIORITY_DISPLAY

    private fun inferenceThread(name: String) = { r: Runnable ->
        Thread({ android.os.Process.setThreadPriority(INFERENCE_PRIORITY); r.run() }, name)
    }

    /** One model job at a time (frames, Finish, reloads). */
    val engineExecutor = Executors.newSingleThreadExecutor(inferenceThread("slt-live-engine"))
    /** Stage 3 renders, off the frame path. */
    val languageExecutor = Executors.newSingleThreadExecutor(inferenceThread("slt-live-language"))

    @Volatile var engine: LiveEngine? = null; private set
    @Volatile var stage3: LiveStage3? = null; private set
    @Volatile private var state = "idle"
    @Volatile private var error: String? = null
    @Volatile var cropEncodeMs: Double? = null; private set

    val status: Map<String, Any?>
        get() = mapOf("ready" to (state == "ready"), "state" to state, "error" to error,
            "hand_images" to engine?.handImages, "crop_encode_ms" to cropEncodeMs)

    private fun prefs(context: Context) = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    fun mode(context: Context): LiveHandImageMode =
        runCatching { LiveHandImageMode.valueOf(prefs(context).getString(KEY_MODE, "AUTO")!!) }.getOrDefault(LiveHandImageMode.AUTO)

    /** Median ms of one MobileCLIP crop on this phone, measured once (on first launch) and stored. */
    private fun measuredCropMs(context: Context): Double {
        val p = prefs(context)
        val device = "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL} ${android.os.Build.VERSION.SDK_INT}"
        if (p.contains(KEY_AUTO_MS) && p.getString(KEY_AUTO_DEVICE, "") == device) return p.getFloat(KEY_AUTO_MS, 1e9f).toDouble()
        val encoder = LiveHandEncoder(context, LiveEngine.ENCODER_ASSET)
        val crop = ByteArray(256 * 256 * 3) { (it % 251).toByte() }
        repeat(2) { encoder.embed(crop) }
        val times = List(5) {
            val t = SystemClock.elapsedRealtimeNanos()
            encoder.embed(crop)
            (SystemClock.elapsedRealtimeNanos() - t) / 1e6
        }.sorted()
        encoder.close()
        val ms = times[times.size / 2]
        p.edit().putFloat(KEY_AUTO_MS, ms.toFloat()).putString(KEY_AUTO_DEVICE, device).apply()
        Log.i(TAG, "crop encode ${"%.1f".format(ms)} ms on $device")
        return ms
    }

    fun handImagesFor(context: Context, mode: LiveHandImageMode): Boolean = when (mode) {
        LiveHandImageMode.ON -> true
        LiveHandImageMode.OFF -> false
        LiveHandImageMode.AUTO -> measuredCropMs(context).also { cropEncodeMs = it } <= AUTO_CROP_BUDGET_MS
    }

    /** Build the models in the background (idempotent while loading or ready). */
    fun warm(context: Context) {
        val app = context.applicationContext
        synchronized(this) {
            if (state == "loading" || state == "ready") return
            state = "loading"; error = null
        }
        engineExecutor.execute {
            try {
                val started = SystemClock.elapsedRealtime()
                val built = LiveEngine(app, handImagesFor(app, mode(app)))
                built.warm()
                engine = built
                Log.i(TAG, "engine ready (hand images ${built.handImages}) in ${SystemClock.elapsedRealtime() - started} ms")
                languageExecutor.execute {
                    try { stage3 = LiveStage3(app).also { it.warm() } } catch (t: Throwable) { Log.e(TAG, "Stage 3 failed", t) }
                }
                state = "ready"
            } catch (t: Throwable) {
                Log.e(TAG, "engine failed", t)
                error = t.message ?: t.javaClass.simpleName
                state = "failed"
            }
        }
    }

    /** Change the hand-images mode; the engine is rebuilt on the engine thread when the choice changes. */
    fun setMode(context: Context, mode: LiveHandImageMode, done: (Boolean) -> Unit) {
        val app = context.applicationContext
        prefs(app).edit().putString(KEY_MODE, mode.name).apply()
        engineExecutor.execute {
            val wanted = runCatching { handImagesFor(app, mode) }.getOrDefault(false)
            val current = engine
            if (current == null || current.handImages != wanted) {
                state = "loading"
                try {
                    current?.close()
                    engine = null
                    engine = LiveEngine(app, wanted).also { it.warm() }
                    state = "ready"
                } catch (t: Throwable) {
                    Log.e(TAG, "engine reload failed", t)
                    error = t.message; state = "failed"
                }
            }
            done(engine?.handImages ?: false)
        }
    }
}

/** Saved live sessions (Documents/live_reel_sessions on iOS), rows in the History page format. */
object LiveSessions {
    fun directory(context: Context) = File(context.filesDir, "live_reel_sessions").also { it.mkdirs() }

    fun utcNow(): String = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US)
        .apply { timeZone = TimeZone.getTimeZone("UTC") }.format(Date())

    fun save(context: Context, stamp: String, value: JSONObject) {
        val file = File(directory(context), "$stamp.json")
        val tmp = File(file.path + ".tmp")
        tmp.writeText(value.toString())
        tmp.renameTo(file)
    }

    fun list(context: Context): List<Map<String, Any?>> {
        val rows = ArrayList<Map<String, Any?>>()
        for (file in directory(context).listFiles { f -> f.extension == "json" } ?: emptyArray()) {
            val value = runCatching { JSONObject(file.readText()) }.getOrNull() ?: continue
            val events = value.optJSONArray("events") ?: JSONArray()
            val glosses = ArrayList<String>()
            val sentences = ArrayList<String>()
            val timeline = ArrayList<Map<String, Any?>>()
            for (i in 0 until events.length()) {
                val e = events.optJSONObject(i) ?: continue
                when (e.optString("event")) {
                    "word" -> e.optJSONObject("word")?.optString("gloss")?.let {
                        glosses.add(it); timeline.add(mapOf("seconds" to e.optDouble("seconds", 0.0), "gloss" to it))
                    }
                    "sentence" -> sentences.add(e.optString("sentence"))
                }
            }
            rows.add(mapOf(
                "stamp" to file.nameWithoutExtension,
                "started_utc" to value.optString("started_utc", utcNow()),
                "duration_seconds" to value.optDouble("duration_seconds", 0.0),
                "glosses" to glosses, "sentences" to sentences,
                "complete" to value.optBoolean("complete", true),
                "timeline" to timeline,
            ))
        }
        return rows.sortedByDescending { it["started_utc"] as String }
    }
}
