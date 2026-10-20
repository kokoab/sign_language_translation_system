package com.example.slt_mobile_app.live

import org.json.JSONArray
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.util.zip.GZIPInputStream

class LiveCaptionsTest {
    @Test fun matchesDesktopAndIPhoneTimedCaptionStreams() {
        val assets = File("src/main/assets")
        val models = LiveCaptionModels(File(assets,"live_captions_v17.json").readText(), File(assets,"live_captions_v17.bin").readBytes())
        val cases = javaClass.getResourceAsStream("/caption_parity.json.gz")!!.use {
            GZIPInputStream(it).bufferedReader().use { r -> JSONArray(r.readText()) }
        }
        var events = 0
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i); val session = LiveCaptions(models); val steps = case.getJSONArray("events")
            for (j in 0 until steps.length()) {
                val e = steps.getJSONObject(j)
                val actual = when(e.getString("op")) {
                    "push" -> session.push(e.getString("gloss"),e.getDouble("score"),e.getDouble("seconds"))
                    "tick" -> session.tick(e.getDouble("seconds"))
                    "finish" -> session.finish()
                    else -> error("Unknown event")
                }
                val expected = if(e.isNull("text")) null else e.getString("text")
                assertEquals("${case.getString("id")} event $j",expected,actual)
                events++
            }
            session.reset(); assertEquals("",session.text); assertEquals("",session.finish()); assertNull(session.tick(100.0))
        }
        println("Caption parity: ${cases.length()} streams, $events events")
    }
}
