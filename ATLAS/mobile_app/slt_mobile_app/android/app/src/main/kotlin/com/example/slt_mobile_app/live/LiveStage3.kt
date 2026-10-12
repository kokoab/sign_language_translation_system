// Stage 3 on device: evidence-encoded glosses -> English with the T5-efficient-tiny slot model (TFLite).
// Port of ios/Runner/LiveReel/LiveReelStage3.swift (TinyStage3Naturalizer.rephrase + render_sentence /
// render_utterance and the incremental translator of scripts/eval_stage3_multisentence_v17.py).
package com.example.slt_mobile_app.live

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

private fun assetJson(context: Context, name: String): JSONObject =
    JSONObject(context.assets.open(name).bufferedReader().use { it.readText() })

private fun JSONArray.strings(): List<String> = List(length()) { getString(it) }
private fun JSONArray.ints(): List<Int> = List(length()) { getInt(it) }

private fun capitalizeFirst(text: String) = if (text.isEmpty()) text else text.substring(0, 1).uppercase() + text.substring(1)

class LiveStage3(context: Context, threads: Int = 4) {
    val checkpointIdentifier: String
    private val encoder = LiteModel(context, "stage3_t5_encoder_fp32.tflite", threads)
    private val decoder = LiteModel(context, "stage3_t5_decoder_step_fp32.tflite", threads)
    private val wordIDs: Map<String, List<Int>>
    private val pieces: List<String>
    private val special: Set<Int>
    private val eos: Int
    private val start: Int
    private val length: Int
    private val lexicon: Map<String, String>
    private val templates: Map<List<String>, String>
    private val emptyOutput: String
    private val fullUtterance: Boolean
    /** Multi-sentence models emit "N: sentence" per sentence (N = glosses consumed). */
    val countedOutput: Boolean

    // Export argument order: encoder (args_0 input_ids, args_1 attention_mask); decoder (args_0
    // decoder_input_ids, args_1 encoder_hidden, args_2 encoder_mask, args_3 position).

    init {
        val tokens = assetJson(context, "stage3_tokens.json")
        checkpointIdentifier = tokens.optString("checkpoint", "unversioned")
        val ids = tokens.getJSONObject("word_ids")
        wordIDs = ids.keys().asSequence().associateWith { ids.getJSONArray(it).ints() }
        pieces = tokens.getJSONArray("pieces").strings()
        special = tokens.getJSONArray("special_ids").ints().toSet()
        eos = tokens.getInt("eos_id")
        start = tokens.getInt("decoder_start_id")
        length = tokens.getInt("max_length")
        fullUtterance = tokens.optString("utterance_segmentation") == "model"
        countedOutput = tokens.optString("output") == "counted_sentences"
        val manifest = assetJson(context, "stage3_mobile_naturalizer_manifest_v17.json")
        val lex = manifest.optJSONObject("literal_lexicon")
        lexicon = lex?.keys()?.asSequence()?.associateWith { lex.getString(it) } ?: emptyMap()
        emptyOutput = manifest.optString("empty_output", "")
        val rows = manifest.optJSONArray("reviewed_templates") ?: JSONArray()
        val table = HashMap<List<String>, String>()
        for (i in 0 until rows.length()) {
            val row = rows.getJSONObject(i)
            val glosses = row.optJSONArray("glosses") ?: continue
            val english = row.optString("english", "")
            if (row.has("english")) table[glosses.strings()] = english
        }
        templates = if (tokens.optBoolean("reviewed_templates_enabled", true)) table else emptyMap()
    }

    companion object {
        fun bucket(score: Double) = if (score >= 0.60) "hi" else if (score >= 0.40) "mid" else "lo"

        fun spelledText(token: String): String {
            val word = if (token.startsWith("fs-")) token.substring(3) else token
            return if (word.isEmpty()) word else word.substring(0, 1) + word.substring(1).lowercase()
        }

        private val countedPattern = Regex("""(\d+):\s*(.*?)(?=\s+\d+:\s|$)""")

        /** parse_counted: "3: Hello, my friend. 2: How are you?" -> [(3, ...), (2, ...)]. */
        fun parseCounted(text: String, glossCount: Int): List<Part> {
            val trimmed = text.trim(' ', '\t')
            val parts = countedPattern.findAll(trimmed).mapNotNull { m ->
                val count = m.groupValues[1].toIntOrNull() ?: return@mapNotNull null
                Part(count, m.groupValues[2].trim(' ', '\t'))
            }.toList()
            return parts.ifEmpty { listOf(Part(glossCount, trimmed)) }
        }

        /** Gloss groups for the session log; one group when the counts do not cover the words. */
        fun clauses(words: List<LiveWord>, parts: List<Part>): List<List<String>> {
            if (parts.sumOf { it.count } != words.size || parts.any { it.count <= 0 }) {
                return if (words.isEmpty()) emptyList() else listOf(words.map { it.gloss })
            }
            var from = 0
            return parts.map { part -> words.subList(from, from + part.count).map { it.gloss }.also { from += part.count } }
        }
    }

    data class Part(val count: Int, val text: String)
    data class Rendering(var sentence: String, var mode: String)

    /** One throwaway generation after loading so the first Finish does not pay interpreter warm-up. */
    fun warm() { runCatching { generate(listOf("HELLO"), listOf(0.9)) } }

    fun literal(glosses: List<String>): String {
        if (glosses.isEmpty()) return emptyOutput
        return capitalizeFirst(glosses.joinToString(" ") { lexicon[it] ?: it.lowercase() }) + "."
    }

    /** Greedy T5 generation; throws when a word has no token entry. */
    fun generate(glosses: List<String>, scores: List<Double>): String {
        var ids = ArrayList<Int>()
        for ((gloss, score) in glosses.zip(scores)) {
            for (word in listOf(bucket(score), gloss.lowercase())) {
                ids.addAll(wordIDs[word] ?: throw LiveException("No Stage-3 tokens for $word"))
            }
        }
        if (fullUtterance && ids.size >= length) throw LiveException("Finished utterance exceeds model context; refuse to drop words")
        ids = ArrayList(ids.take(length - 1) + eos)
        val inputIDs = IntArray(length) { if (it < ids.size) ids[it] else 0 }
        val mask = IntArray(length) { if (it < ids.size) 1 else 0 }
        encoder.setInt(0, inputIDs)
        encoder.setInt(1, mask)
        val hidden = encoder.runFloat()
        if (hidden.size % length != 0) throw LiveException("Stage-3 length mismatch")
        decoder.setFloat(1, hidden)
        decoder.setInt(2, mask)
        val decoderIDs = IntArray(length)
        val position = IntArray(1)
        val out = arrayListOf(start)
        decoderIDs[0] = start
        while (out.size < length) {
            decoder.setInt(0, decoderIDs)
            position[0] = out.size - 1
            decoder.setInt(3, position)
            val next = decoder.runInt()[0]
            if (next == eos) break
            decoderIDs[out.size] = next
            out.add(next)
        }
        val text = out.drop(1).filter { it !in special }.joinToString("") { pieces[it] }.replace("▁", " ")
        return text.split(' ', '\n', '\t').filter { it.isNotEmpty() }.joinToString(" ")
    }

    /** TinyStage3Naturalizer.rephrase */
    fun rephrase(glosses: List<String>, scores: List<Double>): Rendering {
        templates[glosses]?.let { return Rendering(it, "reviewed_template") }
        val sentence = runCatching { generate(glosses, scores) }.getOrNull()
        if (sentence != null && sentence.isNotEmpty() && sentence.length <= 300) return Rendering(sentence, "t5_efficient_tiny")
        return Rendering(literal(glosses), "literal_fallback")
    }

    /** render_sentence: spelled words go through slot tokens FS0, FS1, ... and are restored after. */
    fun renderSentence(glosses: List<String>, scores: List<Double>): Rendering {
        val slots = ArrayList<Pair<String, String>>()
        val inputs = glosses.map { g ->
            if (g.startsWith("fs-")) { val name = "FS${slots.size}"; slots.add(name to spelledText(g)); name } else g
        }
        val value = rephrase(inputs, scores)
        if (slots.isEmpty()) return value
        fun regex(name: String) = Regex("\\b$name\\b", RegexOption.IGNORE_CASE)
        if (slots.all { regex(it.first).findAll(value.sentence).count() == 1 }) {
            var restored = value.sentence
            for ((name, word) in slots) restored = regex(name).replace(restored, Regex.escapeReplacement(word))
            return Rendering(restored, value.mode + "+slots")
        }
        val names = slots.toMap()
        return Rendering(capitalizeFirst(inputs.joinToString(" ") { names[it] ?: it.lowercase() }) + ".", "slot_literal_fallback")
    }

    /** Counted models: render the words (spelled-slot restoration included) and split into parts. */
    fun renderParts(words: List<LiveWord>): List<Part> {
        if (words.isEmpty()) return emptyList()
        val rendered = renderSentence(words.map { it.gloss }, words.map { it.score })
        return parseCounted(rendered.sentence, words.size)
    }

    /** Composition models punctuate the full utterance; older models retain pause splits. */
    fun renderUtterance(words: List<LiveWord>): Pair<String, List<List<String>>> {
        if (countedOutput) {
            val parts = renderParts(words)
            return parts.joinToString(" ") { it.text } to clauses(words, parts)
        }
        val groups = if (fullUtterance) (if (words.isEmpty()) emptyList() else listOf(words)) else liveSplitClauses(words)
        val parts = groups.map { c -> renderSentence(c.map { it.gloss }, c.map { it.score }).sentence.trim(' ', '\t') }
        return parts.joinToString(" ").trim(' ', '\t') to groups.map { c -> c.map { it.gloss } }
    }
}

/**
 * Incremental translation for counted-output models. After committed words arrive the open buffer is
 * rendered once; its first sentence is locked when two consecutive renders agree on it, the output has
 * at least two sentences, the counts cover the buffer, and rendering that sentence's words alone at N,
 * N-1 or N+1 reproduces it exactly. Finish renders only the open tail. Use from one thread only.
 */
class LiveIncrementalTranslator {
    class Lock(val words: List<LiveWord>, val text: String)
    class Finished(val sentence: String, val clauses: List<List<String>>, val locked: Int, val tailWords: Int)

    val locked = ArrayList<Lock>()
    val buffer = ArrayList<LiveWord>()
    private var previous: LiveStage3.Part? = null
    private val cache = HashMap<String, List<LiveStage3.Part>>()

    fun reset() { locked.clear(); buffer.clear(); previous = null; cache.clear() }

    private fun render(words: List<LiveWord>, stage3: LiveStage3): List<LiveStage3.Part> {
        val key = words.joinToString(" ") { "${LiveStage3.bucket(it.score)} ${it.gloss}" }
        return cache.getOrPut(key) { stage3.renderParts(words) }
    }

    private fun verifiedSplit(first: LiveStage3.Part, stage3: LiveStage3): Int? {
        for (n in listOf(first.count, first.count - 1, first.count + 1)) {
            if (n <= 0 || n >= buffer.size) continue
            val alone = render(buffer.subList(0, n).toList(), stage3)
            if (alone.size == 1 && alone[0].text == first.text) return n
        }
        return null
    }

    /** Add committed words and render the open buffer once. Returns a sentence locked by this call. */
    fun append(words: List<LiveWord>, stage3: LiveStage3): Lock? {
        if (words.isEmpty()) return null
        buffer.addAll(words)
        val parts = render(buffer.toList(), stage3)
        if (parts.size < 2 || parts.sumOf { it.count } != buffer.size || parts[0].count <= 0 || parts[0].count >= buffer.size) {
            previous = null
            return null
        }
        if (parts[0] != previous) { previous = parts[0]; return null }
        val n = verifiedSplit(parts[0], stage3) ?: return null
        val lock = Lock(buffer.subList(0, n).toList(), parts[0].text)
        locked.add(lock)
        repeat(n) { buffer.removeAt(0) }
        previous = null
        return lock
    }

    /** Finish: add the flushed tail, render only the open buffer once, return the utterance, reset. */
    fun finish(tail: List<LiveWord>, stage3: LiveStage3): Finished {
        buffer.addAll(tail)
        val words = buffer.toList()
        val parts = render(words, stage3)
        val sentence = (locked.map { it.text } + parts.map { it.text }).joinToString(" ")
        val clauses = locked.map { l -> l.words.map { it.gloss } } + LiveStage3.clauses(words, parts)
        val result = Finished(sentence, clauses, locked.size, words.size)
        reset()
        return result
    }
}
