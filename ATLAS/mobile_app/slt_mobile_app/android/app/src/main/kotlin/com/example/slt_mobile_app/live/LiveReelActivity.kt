// The Android Live Reel / practice camera screen (port of ios/Runner/LiveReel/LiveReelViewController.swift,
// words only). Front camera, 640x480 analysis frames made upright and unmirrored for the models, paced at
// 20 Hz; the preview and the skeleton overlay are mirrored for the signer. LIVE: Start -> Finish (or two
// open palms held for one second) renders the utterance with Stage 3, speaks it and saves the session.
// PRACTICE: the first word committed after each target is judged; Skip moves on.
package com.example.slt_mobile_app.live

import android.Manifest
import android.app.AlertDialog
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Size
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import org.json.JSONArray
import org.json.JSONObject
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.max

class LiveReelActivity : ComponentActivity(), TextToSpeech.OnInitListener {
    companion object {
        const val EXTRA_PRACTICE = "practice_glosses"
        const val RESULT_JSON = "result_json"
        private const val CAMERA_REQUEST = 41
        private val BLUE = Color.rgb(0x1F, 0x5B, 0xFF)

        fun display(gloss: String): String = when {
            gloss.startsWith("fs-") -> LiveStage3.spelledText(gloss).uppercase()
            gloss.startsWith("FS_") -> gloss.substring(3)
            else -> gloss
        }
    }

    private val main = Handler(Looper.getMainLooper())
    private lateinit var previewView: PreviewView
    private lateinit var overlay: SkeletonView
    private lateinit var railLabel: TextView
    private lateinit var previewLabel: TextView
    private lateinit var bannerLabel: TextView
    private lateinit var statsLabel: TextView
    private lateinit var modeChip: TextView
    private lateinit var targetLabel: TextView
    private lateinit var primaryButton: Button
    private lateinit var secondaryButton: Button
    private var tts: TextToSpeech? = null
    private var ttsReady = false

    private var practiceTargets: List<String> = emptyList()
    private val isPractice get() = practiceTargets.isNotEmpty()

    // Engine-thread state (LiveShared.engineExecutor).
    @Volatile private var running = false
    private var firstStamp: Double? = null
    private var deadline = 0.0
    private val utterance = ArrayList<LiveWord>()
    private val computeHistory = ArrayDeque<Double>()
    private val frameTimes = ArrayDeque<Double>()
    private var processedFrames = 0

    // Language-thread state (LiveShared.languageExecutor).
    private val translator = LiveIncrementalTranslator()
    private val pendingTranslation = ArrayList<LiveWord>()

    // Session log (any thread; guarded by itself).
    private val events = JSONArray()
    private var sessionStamp: String? = null
    private var sessionStartedUtc = ""
    private var sessionStartedAt = 0L
    private val sessionGlosses = ArrayList<String>()
    private val sessionSentences = ArrayList<String>()
    private var lastSave = 0L

    // UI-thread state.
    private val rail = ArrayList<String>()
    private var finishing = false
    private var loaded = false
    private var practiceIndex = 0
    private var practiceScore = 0
    private var practiceAttempts = 0
    private val practiceMissed = ArrayList<String>()
    private var awaitingNext = false
    private var roundDone = false
    private var closeResult = JSONObject().put("navigate", "HOME")

    // ---------------------------------------------------------------- lifecycle

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        practiceTargets = intent.getStringArrayListExtra(EXTRA_PRACTICE) ?: emptyList()
        if (isPractice) closeResult = JSONObject().put("navigate", "PRACTICE")
        buildViews()
        tts = TextToSpeech(this, this)
        LiveShared.warm(this)
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        else requestPermissions(arrayOf(Manifest.permission.CAMERA), CAMERA_REQUEST)
        waitForModels()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != CAMERA_REQUEST) return
        if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) startCamera()
        else showBanner("Camera permission is needed for live translation.")
    }

    override fun onInit(status: Int) {
        ttsReady = status == TextToSpeech.SUCCESS && tts?.setLanguage(Locale.US) != TextToSpeech.LANG_NOT_SUPPORTED
    }

    override fun onPause() {
        super.onPause()
        if (running && !isPractice) stopRun(save = true)
    }

    override fun onDestroy() {
        running = false
        tts?.shutdown()
        super.onDestroy()
    }

    @Deprecated("Back closes the screen with its result")
    override fun onBackPressed() = close(if (isPractice) "PRACTICE" else "HOME")

    private fun close(navigate: String) {
        running = false
        if (!isPractice) saveSession(complete = true)
        closeResult.put("navigate", navigate)
        if (isPractice) {
            closeResult.put("completed", roundDone).put("score", practiceScore).put("attempts", practiceAttempts)
                .put("total", practiceTargets.size).put("missed", JSONArray(practiceMissed))
        }
        setResult(RESULT_OK, Intent().putExtra(RESULT_JSON, closeResult.toString()))
        LiveShared.engineExecutor.execute { LiveShared.engine?.reset() }
        finish()
    }

    private fun waitForModels() {
        val status = LiveShared.status
        when (status["state"]) {
            "ready" -> { loaded = true; LiveShared.languageExecutor.execute { translator.reset() } }
            "failed" -> { showBanner("Models failed to load: ${status["error"] ?: ""}"); return }
            else -> { showBanner("Loading models…"); main.postDelayed({ waitForModels() }, 300); return }
        }
        if (bannerLabel.text == "Loading models…") showBanner(null)
        refreshModeChip()
        render()
        if (isPractice) startRun()
    }

    // ---------------------------------------------------------------- camera

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val selector = ResolutionSelector.Builder().setResolutionStrategy(
                ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build()
            val preview = Preview.Builder().setResolutionSelector(selector).build()
            preview.setSurfaceProvider(previewView.surfaceProvider)
            val analysis = ImageAnalysis.Builder().setResolutionSelector(selector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888).build()
            analysis.setAnalyzer(LiveShared.engineExecutor) { analyze(it) }
            val camera = if (provider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA)) CameraSelector.DEFAULT_FRONT_CAMERA
                else CameraSelector.DEFAULT_BACK_CAMERA
            provider.unbindAll()
            provider.bindToLifecycle(this, camera, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    /** Upright, unmirrored ARGB frame with its longest side limited to LiveEngine.DETECTION_SIDE. */
    private fun upright(proxy: ImageProxy): Bitmap {
        val raw = proxy.toBitmap()
        val rotation = proxy.imageInfo.rotationDegrees
        val longest = max(raw.width, raw.height)
        val scale = if (longest > LiveEngine.DETECTION_SIDE) LiveEngine.DETECTION_SIDE.toFloat() / longest else 1f
        if (rotation == 0 && scale == 1f) return raw
        val matrix = Matrix().apply { postRotate(rotation.toFloat()); postScale(scale, scale) }
        return Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, matrix, true)
    }

    private fun analyze(proxy: ImageProxy) {
        try {
            if (!running) return
            val engine = LiveShared.engine ?: return
            val stamp = proxy.imageInfo.timestamp / 1e9
            val first = firstStamp ?: stamp.also { firstStamp = it }
            val seconds = stamp - first
            if (seconds + 1e-6 < deadline) return
            deadline = max(deadline + 1.0 / LiveEngine.FPS, seconds)
            val frame = upright(proxy)
            val bytes = if (engine.handImages) ByteArray(frame.width * frame.height * 4).also {
                frame.copyPixelsToBuffer(ByteBuffer.wrap(it))
            } else ByteArray(0)
            val step = engine.process(BitmapImageBuilder(frame).build(), LiveRGBAImage(bytes, frame.width, frame.height,
                frame.width * 4), seconds, allowFinishGesture = !isPractice)
            onStep(engine, step, seconds)
        } catch (t: Throwable) {
            main.post { showBanner("Frame error: ${t.message ?: t.javaClass.simpleName}") }
        } finally {
            proxy.close()
        }
    }

    /** Engine thread: bookkeeping for one processed frame, then a UI update. */
    private fun onStep(engine: LiveEngine, step: LiveEngine.Step, seconds: Double) {
        processedFrames += 1
        computeHistory.addLast(step.computeSeconds); if (computeHistory.size > 40) computeHistory.removeFirst()
        frameTimes.addLast(seconds); if (frameTimes.size > 30) frameTimes.removeFirst()
        if (!isPractice) {
            utterance.addAll(step.committed)
            translateIncrementally(step.committed)
            for (w in step.committed) {
                event(JSONObject().put("event", "word").put("seconds", seconds).put("word", JSONObject(w.toMap())))
                synchronized(events) { sessionGlosses.add(w.gloss) }
            }
            if (SystemClock.elapsedRealtime() - lastSave >= 5000) saveSession(complete = false)
        }
        val o = step.observation
        val preview = if (step.finishProgress == null) engine.preview else null
        val median = 1000 * computeHistory.sorted()[computeHistory.size / 2]
        val fps = if (frameTimes.size > 1) (frameTimes.size - 1) / max(frameTimes.last() - frameTimes.first(), 1e-6) else 0.0
        val hands = listOfNotNull(o.left, o.right).size
        val frames = processedFrames
        val committed = step.committed.map { it.gloss }
        main.post {
            if (isFinishing) return@post
            overlay.show(o, step.finishProgress)
            for (g in committed) { rail.add(display(g)); if (rail.size > 8) rail.removeAt(0) }
            previewLabel.text = preview?.let { "${display(it.gloss)}  ${(it.score * 100).toInt()}%" } ?: ""
            statsLabel.text = "FPS %.1f   FRAME %.0f ms   HANDS %d   FRAMES %d".format(fps, median, hands, frames)
            if (isPractice) judge(committed)
            render()
        }
        if (step.finishRequested) finishUtterance("two_hand_gesture")
    }

    // ---------------------------------------------------------------- live flow

    private fun startRun() {
        if (!loaded || running) return
        showBanner(null)
        rail.clear()
        LiveShared.engineExecutor.execute {
            LiveShared.engine?.reset()
            firstStamp = null; deadline = 0.0
            frameTimes.clear(); computeHistory.clear()
            utterance.clear()
            if (!isPractice && sessionStamp == null) {
                sessionStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                sessionStartedUtc = LiveSessions.utcNow()
                sessionStartedAt = SystemClock.elapsedRealtime()
            }
            event(JSONObject().put("event", "start").put("time", System.currentTimeMillis() / 1000.0))
            running = true
            main.post { render() }
        }
    }

    /** Stop without rendering (screen left mid-run). */
    private fun stopRun(save: Boolean) {
        running = false
        LiveShared.engineExecutor.execute {
            LiveShared.engine?.reset()
            if (save) saveSession(complete = false)
        }
    }

    private fun takePendingTranslation(): List<LiveWord> = synchronized(pendingTranslation) {
        val words = pendingTranslation.toList(); pendingTranslation.clear(); words
    }

    /** Hand committed words to the incremental translator (counted-output models only). */
    private fun translateIncrementally(committed: List<LiveWord>) {
        if (committed.isEmpty()) return
        synchronized(pendingTranslation) { pendingTranslation.addAll(committed) }
        LiveShared.languageExecutor.execute {
            val stage3 = LiveShared.stage3 ?: return@execute     // still loading: words stay pending
            if (!stage3.countedOutput) { takePendingTranslation(); return@execute }
            val lock = translator.append(takePendingTranslation(), stage3) ?: return@execute
            event(JSONObject().put("event", "stage3_lock").put("sentence", lock.text)
                .put("glosses", JSONArray(lock.words.map { it.gloss })))
        }
    }

    /** Engine thread: flush the open segment, then render and speak the utterance. */
    private fun finishUtterance(reason: String) {
        val engine = LiveShared.engine ?: return
        val tail = runCatching { engine.finish() }.getOrDefault(emptyList())
        running = false
        utterance.addAll(tail)
        synchronized(events) { sessionGlosses.addAll(tail.map { it.gloss }) }
        synchronized(pendingTranslation) { pendingTranslation.addAll(tail) }
        val words = utterance.toList()
        utterance.clear()
        event(JSONObject().put("event", reason).put("words", JSONArray(words.map { JSONObject(it.toMap()) })))
        main.post {
            for (w in tail) { rail.add(display(w.gloss)); if (rail.size > 8) rail.removeAt(0) }
            previewLabel.text = ""
            if (words.isEmpty()) showBanner("No recognized signs to finish.") else { finishing = true; showBanner(null) }
            render()
        }
        if (words.isEmpty()) {
            takePendingTranslation()
            LiveShared.languageExecutor.execute { translator.reset() }
            saveSession(complete = true)
            return
        }
        LiveShared.languageExecutor.execute {
            val started = SystemClock.elapsedRealtimeNanos()
            val pending = takePendingTranslation()
            val stage3 = LiveShared.stage3
            val extra = JSONObject()
            val sentence: String
            val clauses: List<List<String>>
            if (stage3 != null && stage3.countedOutput &&
                translator.locked.sumOf { it.words.size } + translator.buffer.size + pending.size == words.size) {
                // Only the open tail is rendered here; earlier sentences were locked while signing.
                val result = translator.finish(pending, stage3)
                sentence = result.sentence; clauses = result.clauses
                extra.put("locked", result.locked).put("tail_words", result.tailWords)
            } else {
                translator.reset()
                val rendered = stage3?.renderUtterance(words)
                sentence = rendered?.first ?: (words.joinToString(" ") {
                    if (it.gloss.startsWith("fs-")) LiveStage3.spelledText(it.gloss) else it.gloss.lowercase()
                } + ".")
                clauses = rendered?.second ?: emptyList()
            }
            val ms = (SystemClock.elapsedRealtimeNanos() - started) / 1e6
            val row = JSONObject().put("event", "sentence").put("sentence", sentence)
                .put("clauses", JSONArray(clauses.map { JSONArray(it) })).put("ms", ms)
            for (key in extra.keys()) row.put(key, extra.get(key))
            event(row)
            synchronized(events) { sessionSentences.add(sentence) }
            saveSession(complete = true)
            main.post {
                finishing = false
                rail.clear()
                showBanner(sentence)
                say(sentence)
                render()
            }
        }
    }

    private fun say(text: String) {
        if (ttsReady && text.isNotBlank()) tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "slt-live")
    }

    // ---------------------------------------------------------------- practice

    private val target: String? get() = if (!roundDone && practiceIndex < practiceTargets.size) practiceTargets[practiceIndex] else null

    private fun judge(committed: List<String>) {
        val target = target ?: return
        if (awaitingNext) return
        val first = committed.firstOrNull() ?: return
        val seen = display(first)
        practiceAttempts += 1
        if (seen == target) {
            practiceScore += 1
            showBanner("✓ $seen")
            awaitingNext = true
            val index = practiceIndex
            main.postDelayed({ if (awaitingNext && practiceIndex == index) nextPressed() }, 1500)
        } else {
            showBanner("Saw $seen — try again")
        }
        clearBuffer()
    }

    private fun nextPressed() {
        if (!awaitingNext) return
        awaitingNext = false
        showBanner(null)
        advance()
    }

    private fun skipPressed() {
        val target = target ?: return
        if (awaitingNext) return
        practiceAttempts += 1
        practiceMissed.add(target)
        showBanner(null)
        advance()
    }

    private fun advance() {
        practiceIndex += 1
        clearBuffer()
        if (practiceIndex >= practiceTargets.size) {
            roundDone = true
            running = false
            render()
            main.postDelayed({ close("PRACTICE") }, 1200)
            return
        }
        render()
    }

    private fun clearBuffer() {
        rail.clear(); previewLabel.text = ""
        LiveShared.engineExecutor.execute { LiveShared.engine?.reset() }
    }

    // ---------------------------------------------------------------- session log

    private fun event(value: JSONObject) { synchronized(events) { events.put(value) } }

    private fun saveSession(complete: Boolean) {
        val stamp = sessionStamp ?: return
        if (isPractice) return
        lastSave = SystemClock.elapsedRealtime()
        val value = synchronized(events) {
            JSONObject().put("format", "slt_live_reel_session_android_v1").put("started_utc", sessionStartedUtc)
                .put("duration_seconds", (SystemClock.elapsedRealtime() - sessionStartedAt) / 1000.0)
                .put("complete", complete).put("glosses", JSONArray(sessionGlosses))
                .put("sentences", JSONArray(sessionSentences)).put("events", JSONArray(events.toString()))
                .put("hand_images", LiveShared.engine?.handImages).put("stage3", LiveShared.stage3?.checkpointIdentifier)
                .put("device", "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}")
        }
        runCatching { LiveSessions.save(this, stamp, value) }
    }

    // ---------------------------------------------------------------- hand-images mode

    private fun modeText(): String {
        val mode = LiveShared.mode(this)
        val active = LiveShared.engine?.handImages
        val state = when (active) { true -> "hand images"; false -> "landmarks"; null -> "loading" }
        return when (mode) {
            LiveHandImageMode.AUTO -> "AUTO · $state"
            LiveHandImageMode.ON -> "HAND IMAGES"
            LiveHandImageMode.OFF -> "LANDMARKS"
        }
    }

    private fun refreshModeChip() { modeChip.text = modeText() }

    private fun chooseMode() {
        if (finishing) return
        val modes = LiveHandImageMode.values()
        val ms = LiveShared.cropEncodeMs?.let { " — %.0f ms per crop here".format(it) } ?: ""
        val names = arrayOf("Automatic$ms", "Hand images on (faster phones)", "Landmarks only")
        AlertDialog.Builder(this).setTitle("Recognition mode")
            .setSingleChoiceItems(names, modes.indexOf(LiveShared.mode(this))) { dialog, which ->
                dialog.dismiss()
                if (running) { if (isPractice) running = false else stopRun(save = true) }
                loaded = false
                modeChip.text = "LOADING…"
                render()
                LiveShared.setMode(this, modes[which]) {
                    main.post { loaded = true; refreshModeChip(); render(); if (isPractice) startRun() }
                }
            }.show()
    }

    // ---------------------------------------------------------------- views

    private fun dp(v: Float) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, resources.displayMetrics)

    private fun label(size: Float, color: Int, bold: Boolean = false) = TextView(this).apply {
        setTextColor(color); setTextSize(TypedValue.COMPLEX_UNIT_SP, size)
        if (bold) typeface = Typeface.DEFAULT_BOLD
        setShadowLayer(dp(3f), 0f, 0f, Color.argb(220, 0, 0, 0))
    }

    private fun pill(color: Int) = GradientDrawable().apply { cornerRadius = dp(28f); setColor(color) }

    private fun button(text: String, color: Int, onClick: () -> Unit) = Button(this).apply {
        this.text = text; isAllCaps = false; setTextColor(Color.WHITE)
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 17f); typeface = Typeface.DEFAULT_BOLD
        background = pill(color); setPadding(dp(24f).toInt(), 0, dp(24f).toInt(), 0)
        setOnClickListener { onClick() }
    }

    private fun buildViews() {
        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        previewView = PreviewView(this).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
        overlay = SkeletonView(this)
        root.addView(previewView, FrameLayout.LayoutParams(-1, -1))
        root.addView(overlay, FrameLayout.LayoutParams(-1, -1))

        val pad = dp(16f).toInt()
        val top = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL; setPadding(pad, pad, pad, 0) }
        val back = label(17f, Color.WHITE, bold = true).apply {
            text = "‹ Back"; background = pill(Color.argb(140, 0, 0, 0)); setPadding(pad, pad / 2, pad, pad / 2)
            setOnClickListener { close(if (isPractice) "PRACTICE" else "HOME") }
        }
        modeChip = label(13f, Color.WHITE, bold = true).apply {
            text = "LOADING…"; background = pill(Color.argb(140, 0, 0, 0)); setPadding(pad, pad / 2, pad, pad / 2)
            setOnClickListener { chooseMode() }
        }
        statsLabel = label(12f, Color.argb(230, 255, 255, 255)).apply { typeface = Typeface.MONOSPACE }
        top.addView(back)
        top.addView(View(this), LinearLayout.LayoutParams(pad, 1))
        top.addView(modeChip)
        top.addView(View(this), LinearLayout.LayoutParams(0, 1, 1f))
        top.addView(statsLabel)
        root.addView(top, FrameLayout.LayoutParams(-1, -2, Gravity.TOP))

        targetLabel = label(40f, Color.WHITE, bold = true).apply { gravity = Gravity.CENTER }
        root.addView(targetLabel, FrameLayout.LayoutParams(-2, -2, Gravity.TOP or Gravity.CENTER_HORIZONTAL).apply { topMargin = dp(64f).toInt() })

        val bottom = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.BOTTOM; setPadding(pad, 0, pad, pad) }
        val texts = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        bannerLabel = label(22f, Color.WHITE, bold = true)
        railLabel = label(28f, Color.WHITE, bold = true)
        previewLabel = label(18f, Color.argb(200, 255, 255, 255))
        texts.addView(bannerLabel); texts.addView(railLabel); texts.addView(previewLabel)
        bottom.addView(texts, LinearLayout.LayoutParams(0, -2, 1f))
        secondaryButton = button("↺", Color.argb(160, 0, 0, 0)) { secondaryPressed() }
        primaryButton = button("Start", BLUE) { primaryPressed() }
        bottom.addView(secondaryButton, LinearLayout.LayoutParams(-2, dp(56f).toInt()).apply { rightMargin = pad })
        bottom.addView(primaryButton, LinearLayout.LayoutParams(dp(150f).toInt(), dp(56f).toInt()))
        root.addView(bottom, FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM))
        setContentView(root)
        render()
    }

    private fun showBanner(text: String?) {
        bannerLabel.text = text ?: ""
        bannerLabel.visibility = if (text.isNullOrEmpty()) View.GONE else View.VISIBLE
    }

    private fun primaryPressed() {
        if (!loaded || finishing) return
        if (isPractice) {
            if (running) running = false else startRun()
        } else if (running) {
            finishing = true
            render()
            LiveShared.engineExecutor.execute { finishUtterance("finish_button") }
        } else {
            startRun()
        }
        render()
    }

    /** LIVE: ↺ restarts the take; PRACTICE: Skip. */
    private fun secondaryPressed() {
        if (isPractice) { skipPressed(); return }
        if (!loaded || finishing) return
        rail.clear(); previewLabel.text = ""; showBanner(null)
        LiveShared.engineExecutor.execute {
            LiveShared.engine?.reset()
            utterance.clear()
            firstStamp = null; deadline = 0.0
            event(JSONObject().put("event", "reset"))
        }
        takePendingTranslation()
        LiveShared.languageExecutor.execute { translator.reset() }
        render()
    }

    private fun render() {
        railLabel.text = rail.joinToString("  ")
        targetLabel.visibility = if (isPractice) View.VISIBLE else View.GONE
        if (isPractice) {
            targetLabel.text = target?.let { "Sign: $it   (${practiceIndex + 1}/${practiceTargets.size})" }
                ?: "Score $practiceScore/${practiceTargets.size}"
            secondaryButton.text = "Skip"
            primaryButton.text = if (running) "Pause" else "Start"
        } else {
            secondaryButton.text = "↺"
            primaryButton.text = when { finishing -> "Working…"; running -> "Finish"; else -> "Start" }
        }
        primaryButton.isEnabled = loaded && !finishing
        primaryButton.alpha = if (primaryButton.isEnabled) 1f else 0.5f
    }
}

/** Hands (and the shoulders/face when seen) drawn over the mirrored preview. */
class SkeletonView(context: Context) : View(context) {
    private var observation: LiveObservation? = null
    private var progress: Double? = null
    private val point = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(0x5C, 0xE1, 0xE6) }
    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.argb(200, 255, 255, 255); strokeWidth = 4f }
    private val ring = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(0x1F, 0x5B, 0xFF); style = Paint.Style.STROKE; strokeWidth = 12f }
    private val bones = listOf(0 to 1, 1 to 2, 2 to 3, 3 to 4, 0 to 5, 5 to 6, 6 to 7, 7 to 8, 5 to 9, 9 to 10, 10 to 11,
        11 to 12, 9 to 13, 13 to 14, 14 to 15, 15 to 16, 13 to 17, 0 to 17, 17 to 18, 18 to 19, 19 to 20)

    fun show(o: LiveObservation, finishProgress: Double?) { observation = o; progress = finishProgress; invalidate() }

    override fun onDraw(canvas: Canvas) {
        val o = observation ?: return
        // PreviewView FILL_CENTER: the frame is scaled to cover the view and centred; the preview is mirrored.
        val scale = max(width / o.width.toFloat(), height / o.height.toFloat())
        val dx = (width - o.width * scale) / 2f
        val dy = (height - o.height * scale) / 2f
        fun x(p: LivePoint) = dx + (1f - p.x) * o.width * scale
        fun y(p: LivePoint) = dy + p.y * o.height * scale
        for (hand in listOfNotNull(o.left, o.right)) {
            for ((a, b) in bones) {
                val p = hand.points[a]; val q = hand.points[b]
                if (p.c > 0f && q.c > 0f) canvas.drawLine(x(p), y(p), x(q), y(q), line)
            }
            for (p in hand.points) if (p.c > 0f) canvas.drawCircle(x(p), y(p), 6f, point)
        }
        progress?.let { canvas.drawArc(width / 2f - 60f, height / 2f - 60f, width / 2f + 60f, height / 2f + 60f, -90f, (360 * it).toFloat(), false, ring) }
    }
}
