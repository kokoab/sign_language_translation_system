// LiteRT (TFLite) wrappers for the Android live chain (port of LiveReelModels.swift):
// boundary student, span recognizer (landmark-only or with hand images, fixed batch 8), MobileCLIP2-S0.
package com.example.slt_mobile_app.live

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

/**
 * One TFLite graph from the app assets, run with LiteRT's CompiledModel on the CPU (XNNPACK).
 * The Interpreter API built its XNNPACK delegate single-threaded whatever setNumThreads said (measured
 * 2026-10-05: 1, 4 and 8 threads all ~380 ms for the landmark span model, benchmark_model 4 threads
 * 123 ms); CpuOptions passes the thread count to XNNPACK. Inputs are the export arguments args_0..N
 * in forward-signature order; every graph has one output.
 */
class LiteModel(context: Context, asset: String, threads: Int = defaultThreads()) {
    companion object {
        /** Up to four threads, never more than the phone's cores (big.LITTLE phones have >= 4 fast-ish). */
        fun defaultThreads(): Int = minOf(4, Runtime.getRuntime().availableProcessors()).coerceAtLeast(1)
        /** Debug bench override. */
        @Volatile var threadsOverride: Int? = null
    }

    private val model = CompiledModel.create(context.assets, asset,
        CompiledModel.Options(Accelerator.CPU).apply { cpuOptions = CompiledModel.CpuOptions(threadsOverride ?: threads, null, null) })
    private val inputs: List<TensorBuffer> = model.createInputBuffers()
    private val outputs: List<TensorBuffer> = model.createOutputBuffers()
    val inputCount: Int get() = inputs.size

    fun setFloat(arg: Int, values: FloatArray) = inputs[arg].writeFloat(values)
    fun setInt(arg: Int, values: IntArray) = inputs[arg].writeInt(values)

    fun runFloat(): FloatArray { model.run(inputs, outputs); return outputs[0].readFloat() }
    fun runInt(): IntArray { model.run(inputs, outputs); return outputs[0].readInt() }

    fun close() {
        inputs.forEach { it.close() }; outputs.forEach { it.close() }; model.close()
    }
}

/** AV boundary student: features [1,64,450] (args_0) + valid [1,64] (args_1) -> O/B/I logits. */
class LiveBoundaryModel(context: Context, asset: String, val lookahead: Int = 6) {
    private val model = LiteModel(context, asset)
    private val features = FloatArray(64 * LiveFeatures.BOUNDARY_DIMENSION)
    private val valid = FloatArray(64)

    fun logits(window: List<FloatArray>, mask: BooleanArray): FloatArray {
        for ((i, row) in window.withIndex()) System.arraycopy(row, 0, features, i * LiveFeatures.BOUNDARY_DIMENSION, row.size)
        for (i in mask.indices) valid[i] = if (mask[i]) 1f else 0f
        model.setFloat(0, features)
        model.setFloat(1, valid)
        return model.runFloat()
    }
}

/** AVBoundaryStream: bounded raw history -> boundary_features -> BIO for the frame `lookahead` back. */
class LiveBoundaryStream(val model: LiveBoundaryModel) {
    private val raws = ArrayList<FloatArray>()
    private val times = ArrayList<Double>()
    private val rows = ArrayList<FloatArray>()
    private var last = Double.NEGATIVE_INFINITY

    companion object { const val FRAMES = 64 }

    fun reset() { raws.clear(); times.clear(); rows.clear(); last = Double.NEGATIVE_INFINITY }

    private fun logSoftmaxWithUnknown(logits: FloatArray): FloatArray {
        val m = logits.max()
        var sum = 0f
        for (v in logits) sum += exp(v - m)
        val lse = m + ln(sum)
        return floatArrayOf(ln(1e-6f)) + FloatArray(logits.size) { logits[it] - lse }
    }

    /** Log-probs (UNK, O, B, I) for the frame `lookahead` steps back, or null while warming up. */
    fun update(raw: FloatArray, seconds: Double): FloatArray? {
        if (seconds - last > 0.26) reset()
        last = seconds
        raws.add(raw); times.add(seconds)
        while (times.size > 2 && times[1] < seconds - 1.2) { raws.removeAt(0); times.removeAt(0) }
        rows.add(LiveFeatures.boundaryRow(raws, times))
        while (rows.size > FRAMES) rows.removeAt(0)
        if (rows.size <= model.lookahead) return null
        val pad = FRAMES - rows.size
        val zero = FloatArray(LiveFeatures.BOUNDARY_DIMENSION)
        val window = List(max(pad, 0)) { zero } + rows
        val mask = BooleanArray(FRAMES) { it >= max(pad, 0) }
        return logSoftmaxWithUnknown(model.logits(window, mask))
    }

    /** Estimates for the last `lookahead` frames with the unseen future masked (clip-end windows). */
    fun flush(): List<FloatArray> {
        if (rows.isEmpty()) return emptyList()
        val n = rows.size
        val look = model.lookahead
        val read = FRAMES - 1 - look
        val zero = FloatArray(LiveFeatures.BOUNDARY_DIMENSION)
        val out = ArrayList<FloatArray>()
        for (k in look downTo 1) {
            val target = n - k
            if (target < 0) continue
            val window = ArrayList<FloatArray>(FRAMES)
            val mask = BooleanArray(FRAMES)
            for (i in 0 until FRAMES) {
                val index = i + target - read
                val ok = index in 0 until n
                window.add(if (ok) rows[index] else zero)
                mask[i] = ok
            }
            out.add(logSoftmaxWithUnknown(model.logits(window, mask)))
        }
        return out
    }
}

/** One span's recognizer inputs (float32, row-major). */
class LiveSpanInput(val landmarks: FloatArray, val embeddings: FloatArray, val valid: FloatArray, val boxes: FloatArray)

/**
 * Span recognizer, fixed batch 8. Landmark-only graph: one input [8,32,61,5]. Hand-image graph: four
 * inputs (landmarks, hand embeddings [8,16,3,512], hand valid [8,16,3], hand boxes [8,16,3,4]).
 */
class LiveSpanRecognizer(context: Context, asset: String, val labels: List<String>) {
    private val model = LiteModel(context, asset)
    val usesHandImages = model.inputCount == 4
    private val landmarks = FloatArray(BATCH * 32 * 61 * 5)
    private val embeddings = FloatArray(BATCH * 16 * 3 * 512)
    private val valid = FloatArray(BATCH * 16 * 3)
    private val boxes = FloatArray(BATCH * 16 * 3 * 4)

    companion object {
        const val BATCH = 8

        /** SpanRecognizer.inputs: span landmarks plus hand evidence at 16 positions of the trimmed span. */
        fun inputs(observations: List<LiveObservation>, hands: List<LiveHandFrame>): LiveSpanInput? {
            val built = LiveSpanLandmarks.build(observations) ?: return null
            val length = built.trimEnd - built.trimStart
            val e = FloatArray(16 * 1536)
            val v = FloatArray(48)
            val b = FloatArray(192)
            for (k in 0 until 16) {
                val position = Math.rint(k.toDouble() * (length - 1).toDouble() / 15.0).toInt() + built.trimStart
                val hand = hands[position]
                System.arraycopy(hand.embeddings, 0, e, k * 1536, 1536)
                for (j in 0 until 3) v[k * 3 + j] = if (hand.valid[j] > 0.5f) 1f else 0f
                System.arraycopy(hand.boxes, 0, b, k * 12, 12)
            }
            return LiveSpanInput(built.features, e, v, b)
        }
    }

    /** [N][100] word logits, in batches of 8 (short batches padded with their last span). */
    fun logits(batch: List<LiveSpanInput>): List<FloatArray> {
        val out = ArrayList<FloatArray>()
        var i = 0
        while (i < batch.size) {
            val chunk = batch.subList(i, minOf(i + BATCH, batch.size))
            for (slot in 0 until BATCH) {
                val x = chunk[minOf(slot, chunk.size - 1)]
                System.arraycopy(x.landmarks, 0, landmarks, slot * x.landmarks.size, x.landmarks.size)
                if (usesHandImages) {
                    System.arraycopy(x.embeddings, 0, embeddings, slot * x.embeddings.size, x.embeddings.size)
                    System.arraycopy(x.valid, 0, valid, slot * x.valid.size, x.valid.size)
                    System.arraycopy(x.boxes, 0, boxes, slot * x.boxes.size, x.boxes.size)
                }
            }
            model.setFloat(0, landmarks)
            if (usesHandImages) { model.setFloat(1, embeddings); model.setFloat(2, valid); model.setFloat(3, boxes) }
            val words = model.runFloat()
            for (slot in chunk.indices) out.add(words.copyOfRange(slot * 100, slot * 100 + 100))
            i += chunk.size
        }
        return out
    }
}

/** MobileCLIP2-S0 image tower: 256x256 RGB crop (/255, NCHW) -> L2-normalised 512-d embedding. */
class LiveHandEncoder(context: Context, asset: String) {
    private val model = LiteModel(context, asset)
    private val chw = FloatArray(3 * 256 * 256)

    fun embed(rgb: ByteArray): FloatArray {
        val plane = 256 * 256
        for (p in 0 until plane) for (c in 0 until 3) chw[c * plane + p] = (rgb[p * 3 + c].toInt() and 0xFF) / 255f
        model.setFloat(0, chw)
        return model.runFloat()
    }

    /** SpanRecognizer.frame_hand: embeddings, valid and boxes for one frame. */
    fun frameHand(o: LiveObservation, frame: LiveRGBAImage): LiveHandFrame {
        val crops = LiveHandCrops.crops(o, frame)
        val hand = LiveHandFrame(valid = crops.valid, boxes = crops.boxes)
        for (view in 0 until 3) {
            val crop = crops.crops[view] ?: continue
            System.arraycopy(embed(crop), 0, hand.embeddings, view * 512, 512)
        }
        return hand
    }

    fun close() = model.close()
}
