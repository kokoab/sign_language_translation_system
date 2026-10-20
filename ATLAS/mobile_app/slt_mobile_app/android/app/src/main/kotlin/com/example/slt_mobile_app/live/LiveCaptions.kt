package com.example.slt_mobile_app.live

import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import kotlin.math.*

// Same packaged weights and caption policy as iOS LiveCaptions.swift. No recognizer weights
// are shared across platforms. One session/queue owns a LiveCaptions instance.
private const val EOS = 101
private const val MASK = -10000f
private val pronouns = "me him her it them us he she i my you they we here there".split(" ").toSet()
private val pastWords = ("went came ate drank gave saw knew took made told thought wrote found felt slept heard " +
    "understood was were did had loved liked wanted needed helped asked worked learned talked stopped tried used waited listened answered signed").split(" ").toSet()
private fun words(s: String) = s.split(' ').filter { it.isNotEmpty() }
private fun bucket(s: Double) = if (s >= .6) 1 else if (s >= .4) 2 else 3
private fun argmax(x: FloatArray): Int { var best = 0; for (i in x.indices) if (x[i] > x[best]) best = i; return best }
private fun logSoftmax(x: FloatArray): FloatArray {
    val top = x.maxOrNull() ?: 0f
    var total = 0f; for (v in x) total += exp(v - top)
    val shift = top + ln(total)
    return FloatArray(x.size) { x[it] - shift }
}
// erf approximation, max absolute error < 1.5e-7; used only for exact-GELU evaluation.
private fun erf(x: Float): Float {
    val a = abs(x.toDouble()); val t = 1 / (1 + .3275911 * a)
    val y = 1 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t - .284496736)*t + .254829592)*t*exp(-a*a)
    return (if (x < 0) -y else y).toFloat()
}
private class Weights(val all: FloatArray, val index: JSONObject) {
    fun shape(key: String): IntArray = index.getJSONArray(key).getJSONArray(1).let { a -> IntArray(a.length()) { a.getInt(it) } }
    fun tensor(key: String): FloatArray {
        val start = index.getJSONArray(key).getInt(0)
        val count = shape(key).fold(1) { a, b -> a*b }
        require(start >= 0 && start.toLong()+count <= all.size) { "Invalid caption tensor $key" }
        return all.copyOfRange(start, start+count)
    }
}
private class Linear(val weight: FloatArray, val bias: FloatArray, val input: Int, val output: Int) {
    constructor(w: Weights, key: String): this(w.tensor("$key.weight"), w.tensor("$key.bias"), w.shape("$key.weight")[1], w.shape("$key.weight")[0])
    fun call(x: FloatArray, rows: Int): FloatArray {
        val out = FloatArray(rows*output)
        for (r in 0 until rows) for (o in 0 until output) {
            var sum = 0f; val wi = o*input; val xi = r*input
            for (i in 0 until input) sum += x[xi+i]*weight[wi+i]
            out[r*output+o] = sum+bias[o]
        }
        return out
    }
}
private class Norm(w: Weights, key: String) {
    val weight = w.tensor("$key.weight"); val bias = w.tensor("$key.bias")
    fun call(x: FloatArray, rows: Int): FloatArray {
        val d = weight.size; val out = FloatArray(x.size)
        for (r in 0 until rows) {
            var mean = 0f; for (i in 0 until d) mean += x[r*d+i]; mean /= d
            var variance = 0f; for (i in 0 until d) { val v = x[r*d+i]-mean; variance += v*v }; variance /= d
            val scale = 1f/sqrt(variance+1e-5f)
            for (i in 0 until d) out[r*d+i] = (x[r*d+i]-mean)*scale*weight[i]+bias[i]
        }
        return out
    }
}
private class Embedding(w: Weights, key: String) {
    val table = w.tensor("$key.weight"); val width = w.shape("$key.weight")[1]
}
private class EncoderLayer(w: Weights, key: String, val heads: Int) {
    private val shape = w.shape("$key.self_attn.in_proj_weight")
    val projection = Linear(w.tensor("$key.self_attn.in_proj_weight"), w.tensor("$key.self_attn.in_proj_bias"), shape[1], shape[0])
    val output = Linear(w, "$key.self_attn.out_proj")
    val linear1 = Linear(w, "$key.linear1"); val linear2 = Linear(w, "$key.linear2")
    val norm1 = Norm(w, "$key.norm1"); val norm2 = Norm(w, "$key.norm2")
    fun call(x: FloatArray, t: Int, causal: Boolean) {
        val d = output.output; val dh = d/heads
        val qkv = projection.call(norm1.call(x,t),t); val scale = 1f/sqrt(dh.toFloat())
        val attended = FloatArray(t*d); val weights = FloatArray(t)
        for (h in 0 until heads) for (i in 0 until t) {
            val last = if (causal) i else t-1; var top = Float.NEGATIVE_INFINITY
            for (j in 0..last) {
                var dot = 0f
                for (e in 0 until dh) dot += qkv[i*3*d+h*dh+e]*qkv[j*3*d+d+h*dh+e]
                weights[j] = dot*scale; top = max(top,weights[j])
            }
            var total = 0f; for (j in 0..last) { weights[j] = exp(weights[j]-top); total += weights[j] }
            for (j in 0..last) { val p = weights[j]/total
                for (e in 0 until dh) attended[i*d+h*dh+e] += p*qkv[j*3*d+2*d+h*dh+e]
            }
        }
        val attention = output.call(attended,t); for (i in x.indices) x[i] += attention[i]
        val hidden = linear1.call(norm2.call(x,t),t); for (i in hidden.indices) hidden[i] = max(0f,hidden[i])
        val feed = linear2.call(hidden,t); for (i in x.indices) x[i] += feed[i]
    }
}
private class Encoder(w: Weights, layers: Int, heads: Int) {
    val token = Embedding(w,"tok"); val conf = Embedding(w,"conf"); val pos = Embedding(w,"pos")
    val layers = List(layers) { EncoderLayer(w,"enc.layers.$it",heads) }; val norm = Norm(w,"norm")
    fun call(x: List<Int>, c: List<Int>, causal: Boolean): FloatArray {
        val d = token.width; val t = x.size; val h = FloatArray(t*d)
        for (i in 0 until t) for (e in 0 until d) h[i*d+e] = token.table[x[i]*d+e]+conf.table[c[i]*d+e]+pos.table[min(i,48)*d+e]
        for (layer in layers) layer.call(h,t,causal)
        return norm.call(h,t)
    }
}
private data class Output(val pre: FloatArray, val form: FloatArray, val post: FloatArray)
private class Tagger(w: Weights, layers: Int, heads: Int, val lookahead: Int?) {
    val encoder = Encoder(w,layers,heads); val offset = Embedding(w,"off"); val mix = Linear(w,"mix.0")
    val pre = Linear(w,"pre"); val form = Linear(w,"form"); val post = Linear(w,"post")
    fun call(x: List<Int>, c: List<Int>, length: Int): Output {
        val t = x.size; val d = encoder.token.width; val h = encoder.call(x,c,lookahead != null)
        val joined = FloatArray(t*3*d)
        for (i in 0 until t) {
            val j = lookahead?.let { min(i+it,length) } ?: i
            val oi = (j-i).coerceIn(0,7)*d
            for (e in 0 until d) { joined[i*3*d+e] = h[i*d+e]; joined[i*3*d+d+e] = h[j*d+e]; joined[i*3*d+2*d+e] = offset.table[oi+e] }
        }
        val z = mix.call(joined,t); for (i in z.indices) z[i] = .5f*z[i]*(1+erf(z[i]/sqrt(2f)))
        return Output(pre.call(z,t),form.call(z,t),post.call(z,t))
    }
}
private class Mover(w: Weights, layers: Int, heads: Int) {
    val encoder = Encoder(w,layers,heads); val source = Linear(w,"src"); val query = Linear(w,"q"); val key = Linear(w,"k")
    fun call(x: List<Int>, c: List<Int>): Pair<Int,Int> {
        val t = x.size; val d = encoder.token.width; val h = encoder.call(x,c,false)
        val src = argmax(source.call(h,t)); val q = query.call(h.copyOfRange(src*d,(src+1)*d),1); val k = key.call(h,t)
        val scores = FloatArray(t) { j -> var sum = 0f; for (i in 0 until d) sum += q[i]*k[j*d+i]; sum }
        return src to argmax(scores)
    }
}
internal data class CaptionTag(val pre: String, val form: String, val post: String)
private fun tokens(tags: List<CaptionTag>): List<String> = tags.flatMap { words(it.pre)+(if (it.form == "<drop>") emptyList() else words(it.form))+words(it.post) }
private fun strings(a: org.json.JSONArray): List<String> = List(a.length()) { a.getString(it) }

class LiveCaptionModels(spec: String, bytes: ByteArray) {
    private val spec = JSONObject(spec)
    val name = this.spec.getString("source")
    val endSeconds = this.spec.getJSONObject("policy").optDouble("end_of_turn_seconds",1.5)
    val freezeSeconds = this.spec.getJSONObject("policy").optDouble("freeze_seconds",3.5)
    private val vocab = this.spec.getJSONObject("vocab")
    private val glosses = strings(vocab.getJSONArray("glosses"))
    private val gid = glosses.withIndex().associate { it.value to it.index+1 }
    val known = glosses.toSet()-"FS0"
    private val pre = strings(vocab.getJSONArray("pre_vocab")); private val form = strings(vocab.getJSONArray("form_vocab")); private val post = strings(vocab.getJSONArray("post_vocab"))
    private val drop = form.indexOf("<drop>").also { require(it >= 0) }
    private val legal = Array(102) { FloatArray(form.size) { MASK } }
    private val pairMask = FloatArray(pre.size*form.size) { MASK }
    private val prePronoun = pre.map { words(it).any(pronouns::contains) }; private val postPronoun = post.map { words(it).any(pronouns::contains) }
    private val live: List<Tagger>; private val draft: Tagger; private val labeler: Tagger; private val mover: Mover
    private val finalCache = object: LinkedHashMap<String,Pair<List<String>,String?>>(128,.75f,true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String,Pair<List<String>,String?>>?) = size > 128
    }
    init {
        require(bytes.size % 4 == 0)
        val expected = this.spec.getString("weights_sha256")
        val actual = java.security.MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it.toInt() and 255) }
        require(expected == actual) { "Caption weights hash mismatch" }
        val floats = FloatArray(bytes.size/4); ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(floats)
        val all = this.spec.getJSONObject("tensors")
        fun w(key: String) = Weights(floats,all.getJSONObject(key))
        val arch = this.spec.getJSONObject("architecture"); val tag = arch.getJSONObject("tagger"); val mv = arch.getJSONObject("mover")
        live = List(2) { Tagger(w("live_$it"),tag.getInt("layers"),tag.getInt("heads"),1) }
        draft = Tagger(w("draft"),tag.getInt("layers"),tag.getInt("heads"),0)
        labeler = Tagger(w("labeler"),tag.getInt("layers"),tag.getInt("heads"),null)
        mover = Mover(w("mover"),mv.getInt("layers"),mv.getInt("heads"))
        legal[0].fill(0f); legal[EOS].fill(0f)
        val forms = vocab.getJSONObject("legal_forms")
        for ((i,g) in glosses.withIndex()) { val a = forms.optJSONArray(g) ?: continue; for (j in 0 until a.length()) legal[i+1][a.getInt(j)] = 0f }
        val pairs = vocab.getJSONArray("allowed_pairs")
        for (i in 0 until pairs.length()) { val p = pairs.getJSONArray(i); pairMask[p.getInt(0)*form.size+p.getInt(1)] = 0f }
    }
    private fun guard(lp: FloatArray, flags: List<Boolean>): FloatArray {
        val best = argmax(lp); if (flags[best] && exp(lp[best]) < .5f) for (i in lp.indices) if (flags[i]) lp[i] = MASK
        return lp
    }
    private fun tags(models: List<Tagger>, g: List<String>, s: List<Double>, ended: Boolean): List<CaptionTag> {
        val n = g.size; val x = g.map { gid[it] ?: gid.getValue("FS0") } + if (ended) listOf(EOS) else emptyList()
        val c = s.map(::bucket) + if (ended) listOf(1) else emptyList()
        if (x.isEmpty()) return emptyList()
        val outputs = models.map { it.call(x,c,if (ended) n else n-1) }
        return List(n) { i ->
            val p = FloatArray(pre.size); val f = FloatArray(form.size); val q = FloatArray(post.size)
            for (o in outputs) {
                val pp = logSoftmax(o.pre.copyOfRange(i*pre.size,(i+1)*pre.size))
                val qq = logSoftmax(o.post.copyOfRange(i*post.size,(i+1)*post.size))
                val ff = logSoftmax(FloatArray(form.size) { o.form[i*form.size+it]+legal[x[i]][it] })
                for (k in p.indices) p[k] += pp[k]; for (k in q.indices) q[k] += qq[k]; for (k in f.indices) f[k] += ff[k]
            }
            for (k in p.indices) p[k] /= models.size; for (k in q.indices) q[k] /= models.size; for (k in f.indices) f[k] /= models.size
            if (s[i] >= .6) f[drop] = MASK
            guard(p,prePronoun); guard(q,postPronoun)
            var best = Float.NEGATIVE_INFINITY; var aBest = 0; var bBest = 0
            for (a in p.indices) for (b in f.indices) { val joint = p[a]+f[b]+pairMask[a*form.size+b]; if (joint > best) { best = joint; aBest = a; bBest = b } }
            CaptionTag(pre[aBest],form[bBest],post[argmax(q)])
        }
    }
    internal fun liveTags(g: List<String>, s: List<Double>, ended: Boolean) = tags(live,g,s,ended)
    internal fun draftTags(g: List<String>, s: List<Double>) = tags(listOf(draft),g,s,false)
    internal fun finalize(g: List<String>, s: List<Double>): Pair<List<String>,String?> {
        val cacheKey = g.joinToString(" ")+"|"+s.joinToString("") { bucket(it).toString() }
        finalCache[cacheKey]?.let { return it }
        val (src,dst) = mover.call(g.map { gid[it] ?: gid.getValue("FS0") }+EOS,s.map(::bucket)+1)
        var g2 = g; var s2 = s; var moved: String? = null
        if (src < g.size) {
            fun <T> move(seq: List<T>): List<T> = seq.toMutableList().apply { val item = removeAt(src); add(min(if (dst <= src) dst else dst-1,size),item) }
            g2 = move(g); s2 = move(s); if (g2 != g) moved = g[src]
        }
        val out = tokens(tags(listOf(labeler),g2,s2,true)).toMutableList()
        if (out.joinToString(" ").lastOrNull() !in listOf('.', '?', '!')) out.add(".")
        return (out.toList() to moved).also { finalCache[cacheKey] = it }
    }
}
private data class Merged(val g: List<String>, val s: List<Double>, val names: List<String>)
private fun titled(s: String): String { var after = false; return buildString { for (ch in s) { append(if (ch.isLetter()) { val v = if (after) ch.lowercase() else ch.uppercase(); after = true; v } else { after = false; ch.toString() }) } } }
private fun merge(g: List<String>, s: List<Double>, known: Set<String>): Merged {
    val gs = mutableListOf<String>(); val ss = mutableListOf<Double>(); val names = mutableListOf<String>(); var previous = false
    for (i in g.indices) { val spelled = g[i].startsWith("fs-")
        if (spelled && gs.lastOrNull() == "FS0" && previous) { names[names.lastIndex] += g[i].drop(3).lowercase(Locale.ROOT); ss[ss.lastIndex] = max(ss.last(),s[i]) }
        else if (spelled || g[i] !in known) { gs.add("FS0"); ss.add(s[i]); names.add(titled(if (spelled) g[i].drop(3) else g[i])) }
        else { gs.add(g[i]); ss.add(s[i]) }; previous = spelled
    }; return Merged(gs,ss,names)
}
private fun isPunctuation(s: String, chars: String = ".,!?;:") = chars.contains(s)
private fun detokenize(t: List<String>): String {
    val out = mutableListOf<String>(); var cap = true
    for (tok in t) {
        if (isPunctuation(tok) || tok == "'s") {
            if (out.isEmpty()) continue
            if (isPunctuation(tok) && isPunctuation(out.last().takeLast(1))) continue
            out[out.lastIndex] += tok; if (isPunctuation(tok,".!?")) cap = true; continue
        }
        var w = if (tok == "fs0") "FS0" else if (tok == "i") "I" else tok
        if (cap && w.isNotEmpty()) w = w.take(1).uppercase(Locale.ROOT)+w.drop(1)
        cap = false; out.add(w)
    }; return out.joinToString(" ")
}
private fun tenseConflict(t: List<String>): Boolean {
    val times = t.withIndex().filter { it.value == "tomorrow" || it.value == "yesterday" }
    if (times.isEmpty()) return false
    for ((i,w) in t.withIndex()) if (w in pastWords || w == "will") {
        val governing = times.lastOrNull { it.index < i }?.value ?: times.first().value
        if ((w in pastWords && governing == "tomorrow") || (w == "will" && governing == "yesterday")) return true
    }; return false
}
private fun revise(live: List<String>, final: List<String>, moved: String?): Boolean =
    moved in setOf("WHAT","WHERE","WHEN","WHO","WHY","HOW","NO") ||
        (final.any { it == "not" || it == "never" } && live.none { it == "not" || it == "never" }) || (tenseConflict(live) && !tenseConflict(final))

class LiveCaptions(private val models: LiveCaptionModels) {
    private val g = mutableListOf<String>(); private val s = mutableListOf<Double>()
    private var frozen = emptyList<String>(); private var open = 0; private var toks = emptyList<String>()
    private var last: Double? = null; private var ended = false
    var text = ""; private set
    fun reset() { g.clear(); s.clear(); frozen = emptyList(); open = 0; toks = emptyList(); last = null; ended = false; text = "" }
    fun push(gloss: String, score: Double, seconds: Double): String {
        if (ended && last?.let { seconds-it >= models.freezeSeconds-1e-6 } == true) {
            frozen = if (toks.lastOrNull()?.let { !isPunctuation(it,".?!") } == true) toks+"." else toks
            open = merge(g,s,models.known).g.size
        }
        g.add(gloss); s.add(score); last = seconds; ended = false; view(false); return text
    }
    fun tick(seconds: Double): String? {
        if (ended || last == null || seconds-last!! < models.endSeconds-1e-6) return null
        ended = true; view(true,false); return text
    }
    fun finish(): String { if (g.isNotEmpty()) view(true); return text }
    private fun view(end: Boolean, tailPeriod: Boolean = true) {
        val m = merge(g,s,models.known); toks = step(m.g,m.s,end,tailPeriod)
        text = detokenize(toks)
        for (name in m.names) text = text.replaceFirst("FS0",name)
    }
    private fun step(gs: List<String>, ss: List<Double>, end: Boolean, tailPeriod: Boolean): List<String> {
        val n = gs.size; val o = min(open,n)
        val tags = models.liveTags(gs.drop(o),ss.drop(o),end)
        val commit = if (end) n else max(o,n-1)
        val nextFrozen = frozen.toMutableList(); var start = o; val out = mutableListOf<String>()
        for (i in o until commit) if (words(tags[i-o].post).any { w -> w.any { it in ".?!" } } || (end && i == n-1)) {
            val live = tokens(tags.subList(start-o,i+1-o))
            val (fin,moved) = models.finalize(gs.subList(start,i+1),ss.subList(start,i+1))
            val clause = if (revise(live,fin,moved)) fin else if (fin.lastOrNull()?.let { isPunctuation(it,".?!") } == true && live.lastOrNull()?.let { isPunctuation(it,".?!") } != true) live+fin.last() else live
            if (end) out.addAll(clause) else nextFrozen.addAll(clause)
            start = i+1
        }
        if (!end) { frozen = nextFrozen.toList(); open = start; out.clear() }
        if (commit > start) out.addAll(tokens(tags.subList(start-o,commit-o)))
        if (!end && commit < n) out.addAll(tokens(models.draftTags(gs.drop(o),ss.drop(o)).drop(commit-o).map { it.copy(post="") }))
        if (end && !tailPeriod && out.lastOrNull() == ".") out.removeAt(out.lastIndex)
        return nextFrozen+out
    }
}
