// Debug build only. Replays scripts/dump_android_parity_fixtures_v17.py fixtures (pushed to
// <external files>/parity/) through the Kotlin landmark-only runtime and compares the committed words
// frame by frame with Python. Optional extra "stage3" renders each fixture's Python words with the
// TFLite T5. Results: <external files>/parity/results.json and logcat tag SLTParity.
//   adb shell am start -n com.example.slt_mobile_app/com.example.slt_mobile_app.live.LiveParityActivity
package com.example.slt_mobile_app.live

import android.app.Activity
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

class LiveParityActivity : Activity() {
    private lateinit var text: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        text = TextView(this).apply { textSize = 14f; setPadding(32, 32, 32, 32) }
        setContentView(text)
        val stage3 = intent.getBooleanExtra("stage3", false)
        if (intent.hasExtra("threads")) LiteModel.threadsOverride = intent.getIntExtra("threads", 4)
        log("options threads=${LiteModel.threadsOverride ?: LiteModel.defaultThreads()}")
        val priority = intent.getIntExtra("priority", LiveShared.INFERENCE_PRIORITY)
        log("thread priority $priority")
        val work = if (intent.getBooleanExtra("bench", false)) Runnable { bench() } else Runnable { run(stage3) }
        Thread {
            android.os.Process.setThreadPriority(priority)
            try {
                if (intent.getBooleanExtra("big_cores", false)) {
                    // Debug comparison on the Kirin 710: workers inherit this thread's affinity.
                    val task = ProcessBuilder("/system/bin/taskset", "-p", "f0",
                        android.os.Process.myTid().toString()).redirectErrorStream(true).start()
                    if (!task.waitFor(5, java.util.concurrent.TimeUnit.SECONDS)) {
                        task.destroyForcibly()
                        throw IllegalStateException("Affinity command timed out")
                    }
                    val output = task.inputStream.bufferedReader().use { it.readText() }.trim()
                    check(task.exitValue() == 0) { "Affinity failed: $output" }
                    log("affinity $output")
                }
                work.run()
            } catch (t: Throwable) {
                log("FAILED ${t.message}")
            }
        }.start()
    }

    /** Per-model latency through the app's own wrappers (compare with benchmark_model numbers). */
    private fun bench() {
        fun time(name: String, runs: Int = 20, body: () -> Unit) {
            repeat(3) { body() }
            val cpuStart = android.os.Process.getElapsedCpuTime()
            val ms = List(runs) { val t = SystemClock.elapsedRealtimeNanos(); body(); (SystemClock.elapsedRealtimeNanos() - t) / 1e6 }.sorted()
            val cpuPerRun = (android.os.Process.getElapsedCpuTime() - cpuStart).toDouble() / runs
            log("BENCH $name min ${"%.1f".format(ms[0])} median ${"%.1f".format(ms[runs / 2])} ms p90 ${"%.1f".format(ms[runs * 9 / 10])} ms cpu/run ${"%.1f".format(cpuPerRun)} ms")
        }
        try {
            val labels = LiveEngine.loadLabels(this)
            val isolated = intent.getBooleanExtra("isolated", false)
            log("benchmark mode=${if (isolated) "isolated" else "combined"}")
            if (!isolated) {
                val boundary = LiveBoundaryModel(this, "boundary_mediapipe_fp32.tflite")
                val rows = List(64) { FloatArray(LiveFeatures.BOUNDARY_DIMENSION) { k -> (k % 7) * 0.1f } }
                time("boundary") { boundary.logits(rows, BooleanArray(64) { true }) }
            }
            val span = LiveSpanInput(FloatArray(32 * 61 * 5) { (it % 13) * 0.05f }, FloatArray(16 * 3 * 512), FloatArray(48), FloatArray(192))
            if (intent.getBooleanExtra("direct", false)) {
                val model = LiteModel(this, LiveEngine.recognizerAsset(false))
                try {
                    val input = FloatArray(8 * span.landmarks.size)
                    repeat(8) { span.landmarks.copyInto(input, it * span.landmarks.size) }
                    model.setFloat(0, input)
                    time("direct_no_write_b8", runs = 30) { model.runFloat() }
                    time("direct_write_and_run_b8", runs = 30) { model.setFloat(0, input); model.runFloat() }
                } finally {
                    model.close()
                }
                log("BENCH done")
                return
            }
            val landmark = LiveSpanRecognizer(this, LiveEngine.recognizerAsset(false), labels)
            time("span_landmark_only_b8", runs = intent.getIntExtra("runs", 20)) { landmark.logits(List(8) { span }) }
            if (intent.getBooleanExtra("quick", false)) { log("BENCH done"); return }
            val images = LiveSpanRecognizer(this, LiveEngine.recognizerAsset(true), labels)
            time("span_hand_images_b8", runs = 10) { images.logits(List(8) { span }) }
            val encoder = LiveHandEncoder(this, LiveEngine.ENCODER_ASSET)
            val crop = ByteArray(256 * 256 * 3) { (it % 251).toByte() }
            time("mobileclip_crop", runs = 10) { encoder.embed(crop) }
            val stage3 = LiveStage3(this)
            time("stage3_sentence_HELLO_HOW_YOU", runs = 5) { stage3.generate(listOf("HELLO", "HOW", "YOU"), listOf(0.9, 0.8, 0.9)) }
            log("BENCH done")
        } catch (t: Throwable) {
            Log.e("SLTParity", "bench failed", t); log("FAILED ${t.message}")
        }
    }

    private fun log(message: String) {
        // Huawei may suppress Log.i when its global diagnostic flag is off.
        Log.println(Log.INFO, "SLTParity", message)
        runOnUiThread { text.append(message + "\n") }
    }

    private fun points(xy: JSONArray, c: JSONArray) = Array(xy.length()) {
        val p = xy.getJSONArray(it)
        LivePoint(p.getDouble(0).toFloat(), p.getDouble(1).toFloat(), c.getDouble(it).toFloat())
    }

    private fun hand(value: JSONObject?): LiveHand? {
        if (value == null) return null
        return LiveHand(points(value.getJSONArray("xy"), value.getJSONArray("c")), value.getString("chirality"),
            value.getDouble("score").toFloat())
    }

    private fun observation(f: JSONObject) = LiveObservation(
        f.getDouble("seconds"), f.getInt("width"), f.getInt("height"),
        hand(f.optJSONObject("left")), hand(f.optJSONObject("right")),
        points(f.getJSONArray("body_xy"), f.getJSONArray("body_c")),
        points(f.getJSONArray("face_xy"), f.getJSONArray("face_c")),
        f.getBoolean("face_for_features"),
    )

    private fun run(stage3Check: Boolean) {
        val dir = File(getExternalFilesDir(null), "parity")
        try {
            val index = JSONArray(File(dir, "index.json").readText())
            val labels = LiveEngine.loadLabels(this)
            val config = LiveEngine.loadConfig(this, LiveEngine.configAsset(false))
            val runtime = LiveSegmentalRuntime(
                LiveBoundaryStream(LiveBoundaryModel(this, "boundary_mediapipe_fp32.tflite", lookahead = 6)),
                LiveSpanRecognizer(this, LiveEngine.recognizerAsset(false), labels), null, config)
            val stage3 = if (stage3Check) LiveStage3(this) else null
            val results = JSONArray()
            var videoMatches = 0
            var frameMismatches = 0
            var frames = 0
            var observeNanos = 0L
            val perFrame = ArrayList<Double>()
            for (i in 0 until index.length()) {
                val fixture = JSONObject(File(dir, index.getJSONObject(i).getString("file")).readText())
                runtime.reset()
                val rows = fixture.getJSONArray("frames")
                val kotlinWords = ArrayList<String>()
                var firstMismatch: JSONObject? = null
                for (k in 0 until rows.length()) {
                    val row = rows.getJSONObject(k)
                    val o = observation(row)
                    val t = SystemClock.elapsedRealtimeNanos()
                    val words = runtime.observe(o, null).map { it.gloss }
                    val spent = SystemClock.elapsedRealtimeNanos() - t
                    observeNanos += spent; perFrame.add(spent / 1e6); frames += 1
                    kotlinWords.addAll(words)
                    val expected = row.getJSONArray("committed").let { a -> List(a.length()) { a.getString(it) } }
                    if (words != expected) {
                        frameMismatches += 1
                        if (firstMismatch == null) firstMismatch = JSONObject().put("frame", k)
                            .put("python", JSONArray(expected)).put("kotlin", JSONArray(words))
                    }
                }
                kotlinWords.addAll(runtime.finish().map { it.gloss })
                val pythonWords = fixture.getJSONArray("words").let { a -> List(a.length()) { a.getJSONObject(it).getString("gloss") } }
                val match = kotlinWords == pythonWords
                if (match) videoMatches += 1
                val result = JSONObject().put("id", fixture.getString("id")).put("match", match)
                    .put("python", JSONArray(pythonWords)).put("kotlin", JSONArray(kotlinWords))
                firstMismatch?.let { result.put("first_mismatch", it) }
                if (stage3 != null && pythonWords.isNotEmpty()) {
                    val scores = fixture.getJSONArray("words").let { a -> List(a.length()) { a.getJSONObject(it).getDouble("score") } }
                    val t = SystemClock.elapsedRealtimeNanos()
                    val rendered = stage3.renderSentence(pythonWords, scores)
                    result.put("stage3", rendered.sentence).put("stage3_mode", rendered.mode)
                        .put("stage3_ms", (SystemClock.elapsedRealtimeNanos() - t) / 1e6)
                }
                results.put(result)
                log("${i + 1}/${index.length()} ${if (match) "OK " else "DIFF"} ${fixture.getString("id")} py=$pythonWords kt=$kotlinWords")
            }
            perFrame.sort()
            val summary = JSONObject().put("videos", index.length()).put("video_matches", videoMatches)
                .put("frames", frames).put("frame_mismatches", frameMismatches)
                .put("observe_ms_mean", observeNanos / 1e6 / frames)
                .put("observe_ms_median", perFrame[perFrame.size / 2])
                .put("observe_ms_p90", perFrame[(perFrame.size * 9) / 10])
                .put("device", "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}")
            File(dir, "results.json").writeText(JSONObject().put("summary", summary).put("results", results).toString(1))
            log("SUMMARY $summary")
        } catch (t: Throwable) {
            Log.e("SLTParity", "failed", t)
            log("FAILED ${t.javaClass.simpleName}: ${t.message}")
            File(dir, "results.json").writeText(JSONObject().put("error", t.toString()).toString())
        }
    }
}
