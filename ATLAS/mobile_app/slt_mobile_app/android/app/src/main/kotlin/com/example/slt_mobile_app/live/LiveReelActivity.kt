// The Android Live Reel / practice camera screen (port of ios/Runner/LiveReel/LiveReelViewController.swift,
// words only). Portrait/landscape camera with live English captions. Finish speaks once and saves
// the session. Android retains its MediaPipe/TFLite recognizer and optional hand-image mode.
// PRACTICE: the first word committed after each target is judged; Skip moves on.
package com.example.slt_mobile_app.live

import android.Manifest
import android.app.AlertDialog
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ActivityInfo
import android.content.res.Configuration
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
import android.widget.VideoView
import android.widget.PopupMenu
import android.net.Uri
import android.os.Environment
import java.io.File
import androidx.camera.core.UseCaseGroup
import androidx.camera.video.*
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
    private val settings by lazy { getSharedPreferences("slt_shell", MODE_PRIVATE) }
    private var cameraProvider: ProcessCameraProvider? = null
    private var frontCamera = true
    private var showLandmarks = true
    private var showSigns = true
    private var captionText = ""
    private var finishedSentence = ""
    private var framingText = ""
    private var liveHint: TextView? = null
    private var speakButton: Button? = null
    private var reference: VideoView? = null
    private var referenceFile: File? = null
    private var referenceTarget: String? = null
    private var referenceExpanded = false
    private var referenceDialog: AlertDialog? = null
    private var closing = false
    private val framing = LiveFraming()
    private var brightness: Double? = null
    private var referenceGeneration = 0
    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    @Volatile private var generation = 0
    private var lastSeconds = 0.0


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
    private var captions: LiveCaptions? = null

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
        showLandmarks = settings.getBoolean("showLandmarks", true)
        showSigns = settings.getBoolean("showSigns", true)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        buildViews()
        main.postDelayed({ if (!isFinishing) requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_FULL_SENSOR }, 800)
        if (!isPractice && !settings.getBoolean("liveTipsSeen", false)) main.post { showGuide() }
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
        ttsReady = status == TextToSpeech.SUCCESS && (tts?.setLanguage(Locale.US) ?: -1) >= 0
    }

    override fun onPause() {
        super.onPause()
        reference?.pause()
        stopRecording()
        if (running) stopRun(save = !isPractice)
    }

    override fun onDestroy() {
        running = false
        generation++
        referenceGeneration++
        main.removeCallbacksAndMessages(null)
        referenceDialog?.dismiss()
        reference?.stopPlayback()
        stopRecording()
        cameraProvider?.unbindAll()
        tts?.shutdown()
        super.onDestroy()
    }

    override fun onResume() {
        super.onResume()
        if (loaded && isPractice && !roundDone) startRun()
        reference?.start()
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        val banner = bannerLabel.text.toString()
        referenceDialog?.dismiss()
        reference?.stopPlayback(); referenceTarget = null
        buildViews(); showBanner(banner); render()
        previewView.post { startCamera() }
    }

    @Deprecated("Back closes the screen with its result")
    override fun onBackPressed() = close(if (isPractice) "PRACTICE" else "HOME")

    private fun close(navigate: String) {
        if (finishing || closing) return
        closing = true; running = false; generation++
        stopRecording()
        closeResult.put("navigate",navigate)
        if (isPractice) closeResult.put("completed",roundDone).put("score",practiceScore).put("attempts",practiceAttempts)
            .put("total",practiceTargets.size).put("missed",JSONArray(practiceMissed))
        LiveShared.engineExecutor.execute {
            LiveShared.engine?.reset()
            LiveShared.languageExecutor.execute {
                if (!isPractice) saveSession(complete=true)
                main.post { setResult(RESULT_OK,Intent().putExtra(RESULT_JSON,closeResult.toString())); finish() }
            }
        }
    }

    private fun waitForModels() {
        val status = LiveShared.status
        when (status["state"]) {
            "ready" -> { loaded = true }
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
        if (isFinishing || isDestroyed || checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return
        if (previewView.width == 0) { previewView.post { startCamera() }; return }
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            if (isFinishing || isDestroyed) return@addListener
            try {
                val provider = future.get(); cameraProvider = provider
                val selector = ResolutionSelector.Builder().setResolutionStrategy(
                    ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build()
                val rotation = previewView.display.rotation
                val preview = Preview.Builder().setResolutionSelector(selector).setTargetRotation(rotation).build()
                preview.setSurfaceProvider(previewView.surfaceProvider)
                val analysis = ImageAnalysis.Builder().setResolutionSelector(selector).setTargetRotation(rotation)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888).build()
                analysis.setAnalyzer(LiveShared.engineExecutor) { analyze(it) }
                var camera = if (frontCamera) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
                if (!provider.hasCamera(camera)) { frontCamera = !frontCamera; camera = if (frontCamera) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA }
                overlay.mirrored = frontCamera
                val group = UseCaseGroup.Builder().addUseCase(preview).addUseCase(analysis)
                previewView.viewPort?.let { group.setViewPort(it) }
                videoCapture = if (!isPractice && settings.getBoolean("saveVideos", false)) {
                    VideoCapture.withOutput(Recorder.Builder().setQualitySelector(QualitySelector.from(Quality.SD,
                        FallbackStrategy.lowerQualityOrHigherThan(Quality.SD))).build()).also { group.addUseCase(it) }
                } else null
                stopRecording(); provider.unbindAll()
                try { provider.bindToLifecycle(this, camera, group.build()) }
                catch (t: Exception) {
                    if (videoCapture == null) throw t
                    // Some low-end cameras cannot stream Preview + Analysis + Video together.
                    videoCapture = null
                    provider.unbindAll(); provider.bindToLifecycle(this, camera, preview, analysis)
                    showBanner("Video saving is unavailable with live recognition on this camera.")
                }
                if (running) startRecording()
            } catch (t: Exception) { showBanner("Camera unavailable: ${t.message}") }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun startRecording() {
        if (isPractice || recording != null || !settings.getBoolean("saveVideos", false)) return
        val capture = videoCapture ?: return
        val stamp = sessionStamp ?: return
        val dir = File(getExternalFilesDir(Environment.DIRECTORY_MOVIES), "live_reel_videos").also { it.mkdirs() }
        val file = File(dir, "${stamp}_${System.currentTimeMillis()}.mp4")
        recording = capture.output.prepareRecording(this, FileOutputOptions.Builder(file).build())
            .start(ContextCompat.getMainExecutor(this)) { e ->
                if (e is VideoRecordEvent.Finalize && e.hasError()) {
                    file.delete()
                    if (!isFinishing && !isDestroyed) showBanner("Video could not be saved. Recognition is still available.")
                }
            }
    }

    private fun stopRecording() { recording?.stop(); recording = null }

    /** Upright, unmirrored ARGB frame with its longest side limited to LiveEngine.DETECTION_SIDE. */
    private fun upright(proxy: ImageProxy): Bitmap {
        val full = proxy.toBitmap()
        val crop = proxy.cropRect
        val raw = Bitmap.createBitmap(full, crop.left, crop.top, crop.width(), crop.height())
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
            if (processedFrames % 5 == 0) {
                var sum=0.0; var n=0
                for (y in 0 until frame.height step 16) for (x in 0 until frame.width step 16) {
                    val rgb=frame.getPixel(x,y); sum += .2126*Color.red(rgb)+.7152*Color.green(rgb)+.0722*Color.blue(rgb); n++
                }
                brightness = sum/max(n,1)
            }
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
        lastSeconds = seconds
        processedFrames += 1
        computeHistory.addLast(step.computeSeconds); if (computeHistory.size > 40) computeHistory.removeFirst()
        frameTimes.addLast(seconds); if (frameTimes.size > 30) frameTimes.removeFirst()
        if (!isPractice) {
            utterance.addAll(step.committed)
            updateCaptions(step.committed, seconds, step.finishProgress != null)
            for (w in step.committed) {
                event(JSONObject().put("event", "word").put("seconds", seconds).put("word", JSONObject(w.toMap())))
                synchronized(events) { sessionGlosses.add(w.gloss) }
            }
            if (SystemClock.elapsedRealtime() - lastSave >= 5000) saveSession(complete = false)
        }
        val token = generation
        val o = step.observation
        val preview = if (step.finishProgress == null) engine.preview else null
        val median = 1000 * computeHistory.sorted()[computeHistory.size / 2]
        val fps = if (frameTimes.size > 1) (frameTimes.size - 1) / max(frameTimes.last() - frameTimes.first(), 1e-6) else 0.0
        val hands = listOfNotNull(o.left, o.right).size
        val frames = processedFrames
        val advice = if (step.finishProgress == null) framing.update(o,brightness,seconds) else null
        val committed = step.committed.map { it.gloss }
        main.post {
            if (isFinishing || closing || token != generation) return@post
            overlay.show(o, step.finishProgress)
            framingText = advice ?: ""
            liveHint?.text = framingText
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
        rail.clear(); captionText = ""; finishedSentence = ""; showBanner(null)
        val token = ++generation
        LiveShared.languageExecutor.execute { captions?.reset() }
        LiveShared.engineExecutor.execute {
            if (token != generation) return@execute
            LiveShared.engine?.reset()
            framing.reset(); brightness=null
            firstStamp = null; deadline = 0.0
            frameTimes.clear(); computeHistory.clear()
            utterance.clear()
            if (!isPractice && sessionStamp == null) {
                sessionStamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
                sessionStartedUtc = LiveSessions.utcNow()
                sessionStartedAt = SystemClock.elapsedRealtime()
            }
            event(JSONObject().put("event", "start").put("time", System.currentTimeMillis() / 1000.0))
            running = true
            main.post { if (token == generation) { startRecording(); render() } }
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

    private fun liveCaptions(): LiveCaptions? {
        if (captions == null) LiveShared.captions?.let { captions = LiveCaptions(it) }
        return captions
    }

    private fun updateCaptions(committed: List<LiveWord>, seconds: Double, held: Boolean) {
        val token = generation
        LiveShared.languageExecutor.execute {
            if (token != generation) return@execute
            val model = liveCaptions()
            var latest: String? = null
            for (w in committed) latest = model?.push(w.gloss,w.score,seconds)
            if (!held) model?.tick(seconds)?.let { latest = it }
            if (model == null && committed.isNotEmpty()) latest = synchronized(events) { sessionGlosses.joinToString(" ") { it.lowercase() } }
            latest?.let { text -> main.post { if (token == generation && !isFinishing) { captionText = text; render() } } }
        }
    }

    /** Engine thread: flush the sign interval; captions run off the frame path. */
    private fun finishUtterance(reason: String) {
        if (!running) return
        val engine = LiveShared.engine ?: return
        val token = generation
        running = false
        val tail = runCatching { engine.finish() }.getOrDefault(emptyList())
        utterance.addAll(tail)
        synchronized(events) { sessionGlosses.addAll(tail.map { it.gloss }) }
        val words = utterance.toList(); utterance.clear()
        event(JSONObject().put("event",reason).put("words",JSONArray(words.map { JSONObject(it.toMap()) })))
        main.post { stopRecording(); finishing = words.isNotEmpty(); previewLabel.text = ""; render() }
        LiveShared.languageExecutor.execute {
            if (token != generation) return@execute
            val model = liveCaptions()
            for (w in tail) model?.push(w.gloss,w.score,lastSeconds)
            val sentence = if (words.isEmpty()) "" else model?.finish() ?: words.joinToString(" ") { it.gloss.lowercase() }+"."
            model?.reset()
            if (sentence.isNotEmpty()) {
                event(JSONObject().put("event","sentence").put("sentence",sentence).put("rendering_mode",if (model == null) "literal" else "live_captions"))
                synchronized(events) { sessionSentences.add(sentence) }
            }
            saveSession(complete=true)
            main.post {
                if (token != generation || isFinishing) return@post
                finishing = false; captionText = ""; rail.clear(); finishedSentence = sentence
                showBanner(if (sentence.isEmpty()) "No recognized signs to finish." else sentence)
                if (sentence.isNotEmpty()) say(sentence)
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
        generation++
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
                .put("hand_images", LiveShared.engine?.handImages).put("live_captions", LiveShared.captions?.name)
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
                    main.post {
                        loaded = LiveShared.status["ready"] == true
                        refreshModeChip(); render()
                        if (!loaded) showBanner("Models failed to load: ${LiveShared.status["error"] ?: ""}")
                        else if (isPractice) startRun()
                    }
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

    private fun showGuide() {
        if (!isPractice) settings.edit().putBoolean("liveTipsSeen",true).apply()
        AlertDialog.Builder(this).setTitle(if (isPractice) "Practice" else "Live tips")
            .setMessage(if (isPractice) "Watch the example, then sign it. Tap the video to enlarge it. Skip moves to the next sign. Landscape is recommended; portrait works too."
                else "ATLAS recognizes the 100 signs in Glosses. Keep your face and both hands in view. Tap Start, sign, then tap Finish or hold both hands above your shoulders, right hand open and left hand in a fist, for one second. Finish speaks the English once. Landscape is recommended; portrait works too.")
            .setPositiveButton("Got it",null).show()
    }

    private fun showMenu(anchor: View) {
        val menu = PopupMenu(this,anchor)
        menu.menu.add(0,1,0,"Show landmarks").apply { isCheckable=true; isChecked=showLandmarks }
        if (!isPractice) menu.menu.add(0,2,1,"Show signs").apply { isCheckable=true; isChecked=showSigns }
        menu.menu.add(0,3,2,"Guide")
        menu.menu.add(0,4,3,"Recognition mode")
        menu.setOnMenuItemClickListener {
            when(it.itemId) {
                1 -> { showLandmarks = !showLandmarks; settings.edit().putBoolean("showLandmarks",showLandmarks).apply(); render() }
                2 -> { showSigns = !showSigns; settings.edit().putBoolean("showSigns",showSigns).apply(); render() }
                3 -> showGuide()
                4 -> chooseMode()
            }; true
        }; menu.show()
    }

    private fun flipCamera() {
        if (isPractice || finishing) return
        frontCamera = !frontCamera
        val resume = running
        running = false
        LiveShared.engineExecutor.execute {
            LiveShared.engine?.reset()
            main.post { startCamera(); running = resume; render() }
        }
    }

    private fun configureReference(view: VideoView, file: File) {
        view.setOnPreparedListener { player ->
            player.isLooping = true
            player.playbackParams = player.playbackParams.setSpeed(.75f)
            view.start()
        }
        view.setOnErrorListener { _, _, _ -> showBanner("Reference video is unavailable."); true }
        view.setVideoURI(Uri.fromFile(file))
    }

    private fun showReference() {
        val gloss = target ?: return
        if (referenceTarget == gloss) return
        referenceTarget = gloss
        val token = ++referenceGeneration
        reference?.stopPlayback()
        Thread({
            val file = runCatching {
                val catalog = JSONObject(assets.open("flutter_assets/assets/gloss_examples/catalog.json").bufferedReader().use { it.readText() })
                val name = catalog.getJSONObject("examples").getJSONObject(gloss).getJSONObject("assets").getString("loop")
                require(File(name).name == name && name.endsWith(".mp4"))
                File(cacheDir,"reference_$name").also { file ->
                    if (!file.exists()) assets.open("flutter_assets/assets/gloss_examples/$name").use { input -> file.outputStream().use { input.copyTo(it) } }
                }
            }.getOrNull()
            main.post {
                if (token != referenceGeneration || isFinishing || isDestroyed) return@post
                referenceFile = file
                if (file != null) reference?.let { configureReference(it,file) }
                else showBanner("Reference video is unavailable for $gloss.")
            }
        },"slt-practice-reference").start()
    }

    private fun expandReference() {
        val file = referenceFile ?: return
        if (referenceExpanded) return
        referenceExpanded = true; reference?.pause()
        val video = VideoView(this)
        val dialog = AlertDialog.Builder(this).setTitle(target ?: "Reference")
            .setView(video).setPositiveButton("Close",null).create()
        referenceDialog=dialog
        dialog.setOnDismissListener { video.stopPlayback(); referenceExpanded=false; referenceDialog=null; reference?.start() }
        dialog.show()
        video.layoutParams = video.layoutParams.apply { height = (resources.displayMetrics.heightPixels*.6).toInt() }
        configureReference(video,file)
    }

    private fun buildViews() {
        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        previewView = PreviewView(this).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
        overlay = SkeletonView(this).also { it.mirrored=frontCamera }
        root.addView(previewView,FrameLayout.LayoutParams(-1,-1)); root.addView(overlay,FrameLayout.LayoutParams(-1,-1))
        val pad = dp(16f).toInt()
        val top = LinearLayout(this).apply { orientation=LinearLayout.HORIZONTAL; gravity=Gravity.CENTER_VERTICAL; setPadding(pad,pad,pad,0) }
        val back = label(17f,Color.WHITE,true).apply { text="‹ Back"; setPadding(0,pad/2,pad,pad/2); setOnClickListener { close(if (isPractice) "PRACTICE" else "HOME") } }
        top.addView(back)
        top.addView(label(14f,Color.WHITE,true).apply { text=if(isPractice) "PRACTICE" else "LIVE" })
        top.addView(View(this),LinearLayout.LayoutParams(0,1,1f))
        modeChip = label(12f,Color.WHITE).apply { text="" }
        if (!isPractice) top.addView(label(17f,Color.WHITE,true).apply { text="Flip"; setPadding(pad,pad/2,pad,pad/2); setOnClickListener { flipCamera() } })
        top.addView(label(24f,Color.WHITE,true).apply { text="•••"; setPadding(pad,pad/2,0,pad/2); setOnClickListener { showMenu(this) } })
        root.addView(top,FrameLayout.LayoutParams(-1,-2,Gravity.TOP))
        statsLabel=label(11f,Color.WHITE).apply { typeface=Typeface.MONOSPACE; visibility=View.GONE }
        root.addView(statsLabel,FrameLayout.LayoutParams(-1,-2,Gravity.TOP).apply { topMargin=dp(56f).toInt() })
        liveHint=label(14f,Color.WHITE).apply { gravity=Gravity.CENTER; text=framingText }
        root.addView(liveHint,FrameLayout.LayoutParams(-1,-2,Gravity.TOP).apply { topMargin=dp(64f).toInt() })
        if (isPractice) {
            reference=VideoView(this).apply { setOnTouchListener { _, e -> if (e.action == android.view.MotionEvent.ACTION_UP) { expandReference(); performClick() }; true } }
            val landscape=resources.configuration.orientation == Configuration.ORIENTATION_LANDSCAPE
            root.addView(reference,FrameLayout.LayoutParams(dp(if(landscape) 180f else 220f).toInt(),dp(if(landscape) 120f else 160f).toInt(),Gravity.TOP or Gravity.CENTER_HORIZONTAL).apply { topMargin=dp(64f).toInt() })
        }
        val landscape=resources.configuration.orientation == Configuration.ORIENTATION_LANDSCAPE
        val bottom=LinearLayout(this).apply { orientation=if(landscape && !isPractice) LinearLayout.HORIZONTAL else LinearLayout.VERTICAL; gravity=Gravity.CENTER; setPadding(pad,0,pad,pad) }
        val texts=LinearLayout(this).apply { orientation=LinearLayout.VERTICAL; gravity=Gravity.CENTER }
        targetLabel=label(23f,Color.WHITE,true).apply { gravity=Gravity.CENTER }
        bannerLabel=label(22f,Color.WHITE,true).apply { gravity=Gravity.CENTER }
        railLabel=label(20f,Color.WHITE,true).apply { gravity=Gravity.CENTER }
        previewLabel=label(16f,Color.WHITE).apply { gravity=Gravity.CENTER }
        texts.addView(targetLabel); texts.addView(bannerLabel); texts.addView(railLabel); texts.addView(previewLabel)
        bottom.addView(texts,if(landscape && !isPractice) LinearLayout.LayoutParams(0,-2,1f) else LinearLayout.LayoutParams(-1,-2))
        val controls=LinearLayout(this).apply { orientation=LinearLayout.HORIZONTAL; gravity=Gravity.CENTER; setPadding(0,pad/2,0,0) }
        secondaryButton=button("↺",Color.argb(160,0,0,0)) { secondaryPressed() }
        primaryButton=button("Start",BLUE) { primaryPressed() }
        controls.addView(secondaryButton,LinearLayout.LayoutParams(-2,dp(52f).toInt()).apply { rightMargin=pad/2 })
        controls.addView(primaryButton,LinearLayout.LayoutParams(dp(130f).toInt(),dp(52f).toInt()))
        speakButton=button("Speak again",Color.argb(160,0,0,0)) { say(finishedSentence) }.apply { visibility=View.GONE }
        texts.addView(speakButton,LinearLayout.LayoutParams(-2,dp(44f).toInt()))
        bottom.addView(controls)
        root.addView(bottom,FrameLayout.LayoutParams(-1,-2,Gravity.BOTTOM))
        androidx.core.view.ViewCompat.setOnApplyWindowInsetsListener(root) { view, insets ->
            val safe = insets.getInsets(androidx.core.view.WindowInsetsCompat.Type.systemBars() or androidx.core.view.WindowInsetsCompat.Type.displayCutout())
            view.setPadding(safe.left,safe.top,safe.right,safe.bottom)
            insets
        }
        setContentView(root); androidx.core.view.ViewCompat.requestApplyInsets(root); render()
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
        generation++
        captionText = ""; finishedSentence = ""
        LiveShared.languageExecutor.execute { captions?.reset() }
        render()
    }

    private fun render() {
        overlay.landmarksVisible = showLandmarks
        overlay.outlineVisible = !isPractice && (!running || framingText.isNotEmpty())
        overlay.invalidate()
        railLabel.text = rail.joinToString("  ")
        railLabel.visibility = if (isPractice || showSigns) View.VISIBLE else View.GONE
        previewLabel.visibility = if (isPractice || showSigns) View.VISIBLE else View.GONE
        statsLabel.visibility = if (settings.getBoolean("benchmark",false)) View.VISIBLE else View.GONE
        speakButton?.visibility = if (!isPractice && finishedSentence.isNotEmpty() && !running && !finishing) View.VISIBLE else View.GONE
        if (!isPractice && captionText.isNotEmpty()) showBanner(captionText)
        if (!isPractice && !running && !finishing && finishedSentence.isEmpty() && bannerLabel.text.isEmpty()) showBanner("Line yourself up, then tap Start.")
        targetLabel.visibility = if (isPractice) View.VISIBLE else View.GONE
        if (isPractice) {
            showReference()
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
    var mirrored = true
    var landmarksVisible = true
    var outlineVisible = false
    private val outline = Paint(Paint.ANTI_ALIAS_FLAG).apply { color=Color.argb(130,255,255,255); style=Paint.Style.STROKE; strokeWidth=3f; pathEffect=android.graphics.DashPathEffect(floatArrayOf(14f,10f),0f) }
    private var observation: LiveObservation? = null
    private var progress: Double? = null
    private val point = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(0x5C, 0xE1, 0xE6) }
    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.argb(200, 255, 255, 255); strokeWidth = 4f }
    private val ring = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(0x1F, 0x5B, 0xFF); style = Paint.Style.STROKE; strokeWidth = 12f }
    private val bones = listOf(0 to 1, 1 to 2, 2 to 3, 3 to 4, 0 to 5, 5 to 6, 6 to 7, 7 to 8, 5 to 9, 9 to 10, 10 to 11,
        11 to 12, 9 to 13, 13 to 14, 14 to 15, 15 to 16, 13 to 17, 0 to 17, 17 to 18, 18 to 19, 19 to 20)

    fun show(o: LiveObservation, finishProgress: Double?) { observation = o; progress = finishProgress; invalidate() }

    override fun onDraw(canvas: Canvas) {
        if (outlineVisible) {
            canvas.drawOval(width*.39f,height*.17f,width*.61f,height*.37f,outline)
            val p=android.graphics.Path().apply { moveTo(width*.18f,height*.78f); cubicTo(width*.2f,height*.39f,width*.8f,height*.39f,width*.82f,height*.78f) }
            canvas.drawPath(p,outline)
        }
        val o = observation ?: return
        // PreviewView FILL_CENTER: the frame is scaled to cover the view and centred; the preview is mirrored.
        val scale = max(width / o.width.toFloat(), height / o.height.toFloat())
        val dx = (width - o.width * scale) / 2f
        val dy = (height - o.height * scale) / 2f
        fun x(p: LivePoint) = dx + (if (mirrored) 1f-p.x else p.x) * o.width * scale
        fun y(p: LivePoint) = dy + p.y * o.height * scale
        if (landmarksVisible) for (hand in listOfNotNull(o.left, o.right)) {
            for ((a, b) in bones) {
                val p = hand.points[a]; val q = hand.points[b]
                if (p.c > 0f && q.c > 0f) canvas.drawLine(x(p), y(p), x(q), y(q), line)
            }
            for (p in hand.points) if (p.c > 0f) canvas.drawCircle(x(p), y(p), 6f, point)
        }
        progress?.let { canvas.drawArc(width / 2f - 60f, height / 2f - 60f, width / 2f + 60f, height / 2f + 60f, -90f, (360 * it).toFloat(), false, ring) }
    }
}
