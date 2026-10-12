package com.example.slt_mobile_app

import android.content.Intent
import com.example.slt_mobile_app.live.LiveHandImageMode
import com.example.slt_mobile_app.live.LiveReelActivity
import com.example.slt_mobile_app.live.LiveSessions
import com.example.slt_mobile_app.live.LiveShared
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import org.json.JSONArray
import org.json.JSONObject

/** The Flutter shell plus the native Live Reel (same channel and methods as ios/Runner/AppDelegate.swift). */
class MainActivity : FlutterActivity() {
    companion object {
        private const val CHANNEL = "slt_mobile_app/inference_v17"
        private const val LIVE_REQUEST = 7301
        private const val PREFS = "slt_shell"
    }

    private var pendingLive: MethodChannel.Result? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        // Build the Reel models behind the home page (desktop shell warm()).
        LiveShared.warm(this)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "getBenchmarkEnabled" -> result.success(getSharedPreferences(PREFS, MODE_PRIVATE).getBoolean("benchmark", false))
                "setBenchmarkEnabled" -> {
                    val enabled = call.argument<Boolean>("enabled")
                    if (enabled == null) result.error("invalid_arguments", "Missing benchmark setting", null)
                    else { getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean("benchmark", enabled).apply(); result.success(null) }
                }
                "openLiveReel", "openPractice" -> {
                    if (pendingLive != null) { result.error("busy", "The live screen is already open", null); return@setMethodCallHandler }
                    val intent = Intent(this, LiveReelActivity::class.java)
                    if (call.method == "openPractice") {
                        intent.putStringArrayListExtra(LiveReelActivity.EXTRA_PRACTICE, ArrayList(call.argument<List<String>>("glosses") ?: emptyList()))
                    }
                    pendingLive = result
                    @Suppress("DEPRECATION")
                    startActivityForResult(intent, LIVE_REQUEST)
                }
                "liveReelStatus" -> result.success(LiveShared.status)
                "listSessions" -> Thread {
                    val rows = runCatching { LiveSessions.list(this) }.getOrDefault(emptyList())
                    runOnUiThread { result.success(rows) }
                }.start()
                "getHandImageMode" -> result.success(mapOf(
                    "mode" to LiveShared.mode(this).name, "active" to LiveShared.engine?.handImages,
                    "crop_encode_ms" to LiveShared.cropEncodeMs, "budget_ms" to LiveShared.AUTO_CROP_BUDGET_MS))
                "setHandImageMode" -> {
                    val mode = runCatching { LiveHandImageMode.valueOf(call.argument<String>("mode") ?: "") }.getOrNull()
                    if (mode == null) result.error("invalid_arguments", "mode must be AUTO, ON or OFF", null)
                    else LiveShared.setMode(this, mode) { active -> runOnUiThread { result.success(mapOf("active" to active)) } }
                }
                "inferVideo" -> result.error("unavailable", "Recorded-video inference is iPhone only; use Live on Android.", null)
                "shareReport" -> result.error("unavailable", "Report sharing is iPhone only.", null)
                else -> result.notImplemented()
            }
        }
    }

    @Deprecated("Flutter's embedding still delivers results here")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        @Suppress("DEPRECATION")
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != LIVE_REQUEST) return
        val result = pendingLive ?: return
        pendingLive = null
        val json = data?.getStringExtra(LiveReelActivity.RESULT_JSON)?.let { runCatching { JSONObject(it) }.getOrNull() }
            ?: JSONObject().put("navigate", "HOME")
        result.success(toMap(json))
    }

    private fun toMap(json: JSONObject): Map<String, Any?> = json.keys().asSequence().associateWith { key ->
        when (val v = json.get(key)) {
            is JSONArray -> List(v.length()) { v.get(it) }
            JSONObject.NULL -> null
            else -> v
        }
    }
}
