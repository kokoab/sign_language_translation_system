// Live English captions from the committed sign stream: the phone port of scripts/live_captions_v17.py
// (the desktop Live default since 2026-10-14). Five small transformers tag each sign with its English
// form plus the words before and after it: two live taggers (one-sign lookahead, averaged), a draft
// tagger for the newest sign, and, when a clause closes, a mover (at most one sign moved) and a
// full-context labeler. Rules match the desktop: confident signs (>= 0.6) are never dropped, words
// for signs that were not signed are never added, pronoun/here/there insertions need >= 50%.
// Weights and vocabulary: Models/live_captions_v17.{bin,json}, written by
// ATLAS/python/export/export_live_captions_mobile_v17.py. Float32 on the CPU through Accelerate;
// a few milliseconds per sign. Use one LiveCaptions per conversation, from one queue only.

import Accelerate
import Foundation

private let capEOS = 101, capMaxPosition = 48
private let keepConfidence = 0.6
private let pronounProbability: Float = 0.5
private let masked: Float = -1e4
private let punctuation = ".,!?;:"
private let pastWords: Set<String> = [
    "went", "came", "ate", "drank", "gave", "saw", "knew", "took", "made", "told", "thought", "wrote",
    "found", "felt", "slept", "heard", "understood", "was", "were", "did", "had", "loved", "liked", "wanted",
    "needed", "helped", "asked", "worked", "learned", "talked", "stopped", "tried", "used", "waited",
    "listened", "answered", "signed",
]
private let whSigns: Set<String> = ["WHAT", "WHERE", "WHEN", "WHO", "WHY", "HOW"]
private let pronouns: Set<String> = [
    "me", "him", "her", "it", "them", "us", "he", "she", "i", "my", "you", "they", "we", "here", "there",
]

/// Python's `token in "…"` (substring membership; the empty string is in every string).
private func isIn(_ token: String, _ characters: String) -> Bool {
    token.isEmpty || characters.contains(token)
}

private func words(_ text: String) -> [String] { text.split(separator: " ").map(String.init) }

private func bucket(_ score: Double) -> Int { score >= 0.6 ? 1 : score >= 0.4 ? 2 : 3 }

/// First index of the largest value (torch.argmax).
private func argmax(_ values: [Float]) -> Int {
    var best = 0
    for i in values.indices where values[i] > values[best] { best = i }
    return best
}

private func logSoftmax(_ x: ArraySlice<Float>) -> [Float] {
    let top = x.max() ?? 0
    let total = x.reduce(Float(0)) { $0 + Foundation.exp($1 - top) }
    let shift = top + Foundation.log(total)
    return x.map { $0 - shift }
}

/// Python str.title(): a letter is upper-cased after a non-letter, lower-cased after a letter.
private func titled(_ text: Substring) -> String {
    var out = "", afterLetter = false
    for ch in text {
        if ch.isLetter {
            out += afterLetter ? ch.lowercased() : ch.uppercased()
            afterLetter = true
        } else {
            out.append(ch)
            afterLetter = false
        }
    }
    return out
}

// MARK: - Network

private struct CaptionWeights {
    let values: [Float]
    let index: [String: (offset: Int, shape: [Int])]

    func tensor(_ name: String) throws -> [Float] {
        guard let entry = index[name] else { throw LiveCaptionError.missing(name) }
        let count = entry.shape.reduce(1, *)
        return Array(values[entry.offset..<(entry.offset + count)])
    }

    func shape(_ name: String) throws -> [Int] {
        guard let entry = index[name] else { throw LiveCaptionError.missing(name) }
        return entry.shape
    }
}

enum LiveCaptionError: Error, LocalizedError {
    case missing(String), malformed(String)
    var errorDescription: String? {
        switch self {
        case .missing(let name): return "Caption weight \(name) is missing"
        case .malformed(let what): return "Caption package: \(what)"
        }
    }
}

/// nn.Linear, stored transposed (input × output) for vDSP_mmul.
private struct CaptionLinear {
    let transposed: [Float], bias: [Float], input: Int, output: Int

    init(transposed: [Float], bias: [Float], input: Int, output: Int) {
        self.transposed = transposed; self.bias = bias; self.input = input; self.output = output
    }

    init(_ w: CaptionWeights, _ name: String) throws {
        let shape = try w.shape(name + ".weight")
        output = shape[0]; input = shape[1]
        let weight = try w.tensor(name + ".weight")
        var t = [Float](repeating: 0, count: weight.count)
        vDSP_mtrans(weight, 1, &t, 1, vDSP_Length(input), vDSP_Length(output))
        transposed = t
        bias = try w.tensor(name + ".bias")
    }

    func callAsFunction(_ x: [Float], rows: Int) -> [Float] {
        var out = [Float](repeating: 0, count: rows * output)
        vDSP_mmul(x, 1, transposed, 1, &out, 1, vDSP_Length(rows), vDSP_Length(output), vDSP_Length(input))
        out.withUnsafeMutableBufferPointer { o in
            for r in 0..<rows {
                vDSP_vadd(o.baseAddress! + r * output, 1, bias, 1, o.baseAddress! + r * output, 1, vDSP_Length(output))
            }
        }
        return out
    }
}

private struct CaptionLayerNorm {
    let weight: [Float], bias: [Float]

    init(_ w: CaptionWeights, _ name: String) throws {
        weight = try w.tensor(name + ".weight"); bias = try w.tensor(name + ".bias")
    }

    func callAsFunction(_ x: [Float], rows: Int) -> [Float] {
        let d = weight.count
        var out = [Float](repeating: 0, count: x.count)
        for r in 0..<rows {
            let row = x[(r * d)..<((r + 1) * d)]
            let mean = row.reduce(Float(0), +) / Float(d)
            let variance = row.reduce(Float(0)) { $0 + ($1 - mean) * ($1 - mean) } / Float(d)
            let scale = 1 / (variance + 1e-5).squareRoot()
            for (j, v) in row.enumerated() { out[r * d + j] = (v - mean) * scale * weight[j] + bias[j] }
        }
        return out
    }
}

/// nn.TransformerEncoderLayer(norm_first=True, activation=relu), inference only.
private struct CaptionEncoderLayer {
    let inProjection: CaptionLinear, outProjection: CaptionLinear
    let linear1: CaptionLinear, linear2: CaptionLinear
    let norm1: CaptionLayerNorm, norm2: CaptionLayerNorm
    let heads: Int

    init(_ w: CaptionWeights, _ prefix: String, heads: Int) throws {
        // nn.MultiheadAttention keeps its packed projection as in_proj_weight / in_proj_bias.
        let shape = try w.shape(prefix + ".self_attn.in_proj_weight")
        let packed = try w.tensor(prefix + ".self_attn.in_proj_weight")
        var t = [Float](repeating: 0, count: packed.count)
        vDSP_mtrans(packed, 1, &t, 1, vDSP_Length(shape[1]), vDSP_Length(shape[0]))
        inProjection = CaptionLinear(transposed: t, bias: try w.tensor(prefix + ".self_attn.in_proj_bias"),
                                     input: shape[1], output: shape[0])
        outProjection = try CaptionLinear(w, prefix + ".self_attn.out_proj")
        linear1 = try CaptionLinear(w, prefix + ".linear1")
        linear2 = try CaptionLinear(w, prefix + ".linear2")
        norm1 = try CaptionLayerNorm(w, prefix + ".norm1")
        norm2 = try CaptionLayerNorm(w, prefix + ".norm2")
        self.heads = heads
    }

    func callAsFunction(_ x: inout [Float], rows t: Int, causal: Bool) {
        let d = outProjection.output, dh = d / heads
        let qkv = inProjection(norm1(x, rows: t), rows: t)
        let scale = 1 / Float(dh).squareRoot()
        var attended = [Float](repeating: 0, count: t * d)
        var weights = [Float](repeating: 0, count: t)
        for h in 0..<heads {
            for i in 0..<t {
                let last = causal ? i : t - 1
                var top = -Float.infinity
                for j in 0...last {
                    var dot: Float = 0
                    for e in 0..<dh { dot += qkv[i * 3 * d + h * dh + e] * qkv[j * 3 * d + d + h * dh + e] }
                    weights[j] = dot * scale
                    top = max(top, weights[j])
                }
                var total: Float = 0
                for j in 0...last { weights[j] = Foundation.exp(weights[j] - top); total += weights[j] }
                for j in 0...last {
                    let p = weights[j] / total
                    for e in 0..<dh { attended[i * d + h * dh + e] += p * qkv[j * 3 * d + 2 * d + h * dh + e] }
                }
            }
        }
        let attention = outProjection(attended, rows: t)
        for i in x.indices { x[i] += attention[i] }
        var hidden = linear1(norm2(x, rows: t), rows: t)
        for i in hidden.indices { hidden[i] = max(0, hidden[i]) }
        let feed = linear2(hidden, rows: t)
        for i in x.indices { x[i] += feed[i] }
    }
}

private struct CaptionEmbedding {
    let table: [Float], width: Int
    init(_ w: CaptionWeights, _ name: String) throws {
        table = try w.tensor(name + ".weight"); width = try w.shape(name + ".weight")[1]
    }
    func row(_ i: Int) -> ArraySlice<Float> { table[(i * width)..<((i + 1) * width)] }
}

/// Shared input: token + confidence + position embeddings, then the encoder and its final norm.
private struct CaptionEncoder {
    let token: CaptionEmbedding, confidence: CaptionEmbedding, position: CaptionEmbedding
    let layers: [CaptionEncoderLayer], norm: CaptionLayerNorm

    init(_ w: CaptionWeights, layers count: Int, heads: Int) throws {
        token = try CaptionEmbedding(w, "tok"); confidence = try CaptionEmbedding(w, "conf")
        position = try CaptionEmbedding(w, "pos")
        layers = try (0..<count).map { try CaptionEncoderLayer(w, "enc.layers.\($0)", heads: heads) }
        norm = try CaptionLayerNorm(w, "norm")
    }

    func callAsFunction(_ x: [Int], _ c: [Int], causal: Bool) -> [Float] {
        let d = token.width, t = x.count
        var h = [Float](repeating: 0, count: t * d)
        for i in 0..<t {
            // The desktop model would fail past position 48; a longer clause reuses the last position.
            let a = token.row(x[i]), b = confidence.row(c[i]), p = position.row(min(i, capMaxPosition))
            for e in 0..<d { h[i * d + e] = a[a.startIndex + e] + b[b.startIndex + e] + p[p.startIndex + e] }
        }
        for layer in layers { layer(&h, rows: t, causal: causal) }
        return norm(h, rows: t)
    }
}

/// The experiment's tagger.Tagger: per sign, logits for the words before, its form and the words after.
private struct CaptionTagger {
    let encoder: CaptionEncoder, offset: CaptionEmbedding, mix: CaptionLinear
    let pre: CaptionLinear, form: CaptionLinear, post: CaptionLinear
    /// Lookahead in signs; nil sees the whole clause (the labeler).
    let lookahead: Int?

    init(_ w: CaptionWeights, layers: Int, heads: Int, lookahead: Int?) throws {
        encoder = try CaptionEncoder(w, layers: layers, heads: heads)
        offset = try CaptionEmbedding(w, "off")
        mix = try CaptionLinear(w, "mix.0")
        pre = try CaptionLinear(w, "pre"); form = try CaptionLinear(w, "form"); post = try CaptionLinear(w, "post")
        self.lookahead = lookahead
    }

    func callAsFunction(_ x: [Int], _ c: [Int], length: Int) -> (pre: [Float], form: [Float], post: [Float]) {
        let t = x.count, d = encoder.token.width
        let h = encoder(x, c, causal: lookahead != nil)
        var joined = [Float](repeating: 0, count: t * 3 * d)
        for i in 0..<t {
            let j = lookahead.map { min(i + $0, length) } ?? i
            let o = offset.row(min(max(j - i, 0), 7))
            for e in 0..<d {
                joined[i * 3 * d + e] = h[i * d + e]
                joined[i * 3 * d + d + e] = h[j * d + e]
                joined[i * 3 * d + 2 * d + e] = o[o.startIndex + e]
            }
        }
        var z = mix(joined, rows: t)
        for i in z.indices { z[i] = 0.5 * z[i] * (1 + Foundation.erf(z[i] / Float(2).squareRoot())) }   // exact GELU
        return (pre(z, rows: t), form(z, rows: t), post(z, rows: t))
    }
}

/// The experiment's finalizer.Mover: which sign (if any) to move, and where.
private struct CaptionMover {
    let encoder: CaptionEncoder, source: CaptionLinear, query: CaptionLinear, key: CaptionLinear

    init(_ w: CaptionWeights, layers: Int, heads: Int) throws {
        encoder = try CaptionEncoder(w, layers: layers, heads: heads)
        source = try CaptionLinear(w, "src"); query = try CaptionLinear(w, "q"); key = try CaptionLinear(w, "k")
    }

    /// (source position, destination position) for a clause ending in EOS.
    func callAsFunction(_ x: [Int], _ c: [Int]) -> (source: Int, destination: Int) {
        let t = x.count, d = encoder.token.width
        let h = encoder(x, c, causal: false)
        let src = argmax(source(h, rows: t))
        let q = query(Array(h[(src * d)..<((src + 1) * d)]), rows: 1)
        let k = key(h, rows: t)
        let scores = (0..<t).map { j in (0..<d).reduce(Float(0)) { $0 + q[$1] * k[j * d + $1] } }
        return (src, argmax(scores))
    }
}

// MARK: - Models

/// Loads the packaged models once; shared by every caption session.
final class LiveCaptionModels {
    let name: String
    let endSeconds: Double, freezeSeconds: Double
    private let live: [CaptionTagger], draft: CaptionTagger, labeler: CaptionTagger
    private let mover: CaptionMover
    private let gid: [String: Int]
    let known: Set<String>
    private let preVocab: [String], formVocab: [String], postVocab: [String]
    /// Per token id, the forms it may take (0 allowed, -1e4 not).
    private let legal: [[Float]]
    private let pairMask: [Float]
    private let drop: Int
    private let prePronoun: [Bool], postPronoun: [Bool]
    private var finalCache: [String: ([String], String?)] = [:]

    init(spec specURL: URL, weights weightsURL: URL) throws {
        guard let spec = try JSONSerialization.jsonObject(with: Data(contentsOf: specURL)) as? [String: Any],
              let vocab = spec["vocab"] as? [String: Any],
              let glosses = vocab["glosses"] as? [String],
              let pre = vocab["pre_vocab"] as? [String], let form = vocab["form_vocab"] as? [String],
              let post = vocab["post_vocab"] as? [String],
              let legalForms = vocab["legal_forms"] as? [String: [Int]],
              let pairs = vocab["allowed_pairs"] as? [[Int]],
              let policy = spec["policy"] as? [String: Any],
              let architecture = spec["architecture"] as? [String: [String: Int]],
              let tensors = spec["tensors"] as? [String: [String: [Any]]]
        else { throw LiveCaptionError.malformed("live_captions_v17.json") }
        name = spec["source"] as? String ?? "live_captions"
        endSeconds = policy["end_of_turn_seconds"] as? Double ?? 1.5
        freezeSeconds = policy["freeze_seconds"] as? Double ?? 3.5

        let data = try Data(contentsOf: weightsURL)
        let values = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        func weights(_ model: String) throws -> CaptionWeights {
            guard let entries = tensors[model] else { throw LiveCaptionError.missing(model) }
            var index: [String: (Int, [Int])] = [:]
            for (key, entry) in entries {
                guard entry.count == 2, let offset = entry[0] as? Int, let shape = entry[1] as? [Int] else {
                    throw LiveCaptionError.malformed(key)
                }
                index[key] = (offset, shape)
            }
            return CaptionWeights(values: values, index: index)
        }
        let tagger = architecture["tagger"] ?? [:], moverShape = architecture["mover"] ?? [:]
        let tl = tagger["layers"] ?? 3, th = tagger["heads"] ?? 4
        live = try [CaptionTagger(weights("live_0"), layers: tl, heads: th, lookahead: 1),
                    CaptionTagger(weights("live_1"), layers: tl, heads: th, lookahead: 1)]
        draft = try CaptionTagger(weights("draft"), layers: tl, heads: th, lookahead: 0)
        labeler = try CaptionTagger(weights("labeler"), layers: tl, heads: th, lookahead: nil)
        mover = try CaptionMover(weights("mover"), layers: moverShape["layers"] ?? 3, heads: moverShape["heads"] ?? 4)

        var ids: [String: Int] = [:]
        for (i, g) in glosses.enumerated() { ids[g] = i + 1 }
        gid = ids
        known = Set(glosses).subtracting(["FS0"])
        preVocab = pre; formVocab = form; postVocab = post
        // Same order as the desktop: PAD and EOS rows open, then each gloss's legal forms. The last
        // gloss shares id 101 with EOS, so its row stays fully open there too.
        var lm = [[Float]](repeating: [Float](repeating: masked, count: form.count), count: 102)
        lm[0] = [Float](repeating: 0, count: form.count)
        lm[capEOS] = [Float](repeating: 0, count: form.count)
        for (i, g) in glosses.enumerated() { for f in legalForms[g] ?? [] { lm[i + 1][f] = 0 } }
        legal = lm
        var pm = [Float](repeating: masked, count: pre.count * form.count)
        for pair in pairs where pair.count == 2 { pm[pair[0] * form.count + pair[1]] = 0 }
        pairMask = pm
        guard let dropIndex = form.firstIndex(of: "<drop>") else { throw LiveCaptionError.malformed("<drop>") }
        drop = dropIndex
        prePronoun = pre.map { !Set(words($0)).isDisjoint(with: pronouns) }
        postPronoun = post.map { !Set(words($0)).isDisjoint(with: pronouns) }
    }

    private static func pronounGuard(_ lp: [Float], _ isPronoun: [Bool]) -> [Float] {
        let best = argmax(lp)
        guard isPronoun[best], Foundation.exp(lp[best]) < pronounProbability else { return lp }
        return zip(lp, isPronoun).map { $1 ? masked : $0 }
    }

    private enum TaggerChoice { case live, draft, labeler }

    /// (words before, form, words after) for each sign.
    private func tags(_ choice: TaggerChoice, _ glosses: [String], _ scores: [Double], ended: Bool) -> [(String, String, String)] {
        let n = glosses.count
        let x = glosses.map { gid[$0] ?? gid["FS0"]! } + (ended ? [capEOS] : [])
        let c = scores.map(bucket) + (ended ? [1] : [])
        guard !x.isEmpty else { return [] }
        let models: [CaptionTagger] = choice == .live ? live : choice == .draft ? [draft] : [labeler]
        let outputs = models.map { $0(x, c, length: ended ? n : n - 1) }
        let np = preVocab.count, nf = formVocab.count, nq = postVocab.count
        let count = Float(models.count)
        var result: [(String, String, String)] = []
        for i in 0..<n {
            var lpPre = [Float](repeating: 0, count: np), lpForm = [Float](repeating: 0, count: nf)
            var lpPost = [Float](repeating: 0, count: nq)
            for o in outputs {
                for (k, v) in logSoftmax(o.pre[(i * np)..<((i + 1) * np)]).enumerated() { lpPre[k] += v }
                for (k, v) in logSoftmax(o.post[(i * nq)..<((i + 1) * nq)]).enumerated() { lpPost[k] += v }
                let row = legal[x[i]]
                let formRow = (0..<nf).map { o.form[i * nf + $0] + row[$0] }
                for (k, v) in logSoftmax(formRow[...]).enumerated() { lpForm[k] += v }
            }
            for k in 0..<np { lpPre[k] /= count }
            for k in 0..<nq { lpPost[k] /= count }
            for k in 0..<nf { lpForm[k] /= count }
            if scores[i] >= keepConfidence { lpForm[drop] = masked }
            let p = Self.pronounGuard(lpPre, prePronoun), q = Self.pronounGuard(lpPost, postPronoun)
            var best = -Float.infinity, bestPre = 0, bestForm = 0
            for a in 0..<np {
                for b in 0..<nf {
                    let joint = p[a] + lpForm[b] + pairMask[a * nf + b]
                    if joint > best { best = joint; bestPre = a; bestForm = b }
                }
            }
            result.append((preVocab[bestPre], formVocab[bestForm], postVocab[argmax(q)]))
        }
        return result
    }

    func liveTags(_ g: [String], _ s: [Double], ended: Bool) -> [(String, String, String)] { tags(.live, g, s, ended: ended) }
    func draftTags(_ g: [String], _ s: [Double]) -> [(String, String, String)] { tags(.draft, g, s, ended: false) }

    /// Clause rewrite: at most one sign moved, then full-context rendering.
    func finalize(_ glosses: [String], _ scores: [Double]) -> (tokens: [String], moved: String?) {
        let key = glosses.joined(separator: " ") + "|" + scores.map { String(bucket($0)) }.joined()
        if let cached = finalCache[key] { return cached }
        let n = glosses.count
        let x = glosses.map { gid[$0] ?? gid["FS0"]! } + [capEOS]
        let c = scores.map(bucket) + [1]
        var g2 = glosses, s2 = scores, moved: String?
        let (src, dst) = mover(x, c)
        if src < n {
            g2 = Self.move(glosses, src, dst); s2 = Self.move(scores, src, dst)
            moved = g2 != glosses ? glosses[src] : nil
        }
        var toks = tokensOf(tags(.labeler, g2, s2, ended: true))
        if !(toks.joined(separator: " ").last.map { ".?!".contains($0) } ?? false) { toks.append(".") }
        finalCache[key] = (toks, moved)
        return (toks, moved)
    }

    private static func move<T>(_ seq: [T], _ src: Int, _ dst: Int) -> [T] {
        var out = seq
        let item = out.remove(at: src)
        out.insert(item, at: min(dst <= src ? dst : dst - 1, out.count))
        return out
    }
}

private func tokensOf(_ tags: some Sequence<(String, String, String)>) -> [String] {
    var out: [String] = []
    for (pre, form, post) in tags {
        out += words(pre) + (form == "<drop>" ? [] : words(form)) + words(post)
    }
    return out
}

private func detokenize(_ tokens: [String]) -> String {
    var out: [String] = [], capitalize = true
    for tok in tokens {
        if isIn(tok, punctuation) || tok == "'s" {
            if out.isEmpty { continue }   // orphan punctuation
            if isIn(tok, punctuation) && isIn(String(out[out.count - 1].suffix(1)), punctuation) { continue }
            out[out.count - 1] += tok
            if isIn(tok, ".!?") { capitalize = true }
            continue
        }
        var w = tok == "fs0" ? "FS0" : tok == "i" ? "I" : tok
        if capitalize && !w.isEmpty { w = w.prefix(1).uppercased() + w.dropFirst() }
        capitalize = false
        out.append(w)
    }
    return out.joined(separator: " ")
}

/// fs-GE fs-LO -> one name slot "Gelo"; a gloss the captions do not know is shown as a name too.
private func mergeFingerspelling(_ glosses: [String], _ scores: [Double], known: Set<String>)
    -> (glosses: [String], scores: [Double], names: [String]) {
    var g2: [String] = [], s2: [Double] = [], names: [String] = [], previousSpelled = false
    for (g, s) in zip(glosses, scores) {
        let spelled = g.hasPrefix("fs-")
        if spelled, g2.last == "FS0", previousSpelled {
            names[names.count - 1] += g.dropFirst(3).lowercased()
            s2[s2.count - 1] = max(s2[s2.count - 1], s)
        } else if spelled || !known.contains(g) {
            g2.append("FS0"); s2.append(s)
            names.append(spelled ? titled(g.dropFirst(3)) : titled(g[...]))
        } else {
            g2.append(g); s2.append(s)
        }
        previousSpelled = spelled
    }
    return (g2, s2, names)
}

private func restore(_ text: String, _ names: [String]) -> String {
    var out = text
    for n in names { if let r = out.range(of: "FS0") { out.replaceSubrange(r, with: n) } }
    return out
}

private func meaningKey(_ toks: [String]) -> (negated: Bool, question: Bool, tense: String) {
    let t = Set(toks)
    let future = t.contains("will") || t.contains("going"), past = !t.isDisjoint(with: pastWords)
    return (t.contains("not") || t.contains("never"), toks.contains("?"),
            past && !future ? "past" : future && !past ? "future" : "present")
}

/// A verb tense contradicting its governing time word.
private func tenseConflict(_ toks: [String]) -> Bool {
    let times = toks.enumerated().filter { $0.element == "tomorrow" || $0.element == "yesterday" }
    guard let first = times.first else { return false }
    for (i, t) in toks.enumerated() where pastWords.contains(t) || t == "will" {
        let governing = times.last { $0.offset < i }?.element ?? first.element
        if (pastWords.contains(t) && governing == "tomorrow") || (t == "will" && governing == "yesterday") { return true }
    }
    return false
}

private func needsRevision(_ live: [String], _ final: [String], _ moved: String?) -> Bool {
    if let moved, whSigns.contains(moved) || moved == "NO" { return true }
    if meaningKey(final).negated && !meaningKey(live).negated { return true }
    return tenseConflict(live) && !tenseConflict(final)
}

// MARK: - Session

/// One conversation's captions (LiveCaptions in the desktop script). Feed committed signs; call
/// tick() every frame. Speech is the caller's choice (the phone speaks the Finish caption only).
final class LiveCaptions {
    private let models: LiveCaptionModels
    let endSeconds: Double, freezeSeconds: Double?
    private var glosses: [String] = [], scores: [Double] = []
    private var frozen: [String] = [], open = 0
    private var tokens: [String] = []
    private var lastSignAt: Double?
    private var ended = false
    /// The full caption to display.
    private(set) var text = ""

    init(_ models: LiveCaptionModels, endSeconds: Double? = nil, freezeSeconds: Double? = nil) {
        self.models = models
        self.endSeconds = endSeconds ?? models.endSeconds
        self.freezeSeconds = freezeSeconds ?? models.freezeSeconds
    }

    var isEmpty: Bool { glosses.isEmpty }

    func reset() {
        glosses = []; scores = []; frozen = []; open = 0; tokens = []
        lastSignAt = nil; ended = false; text = ""
    }

    @discardableResult
    func push(_ gloss: String, score: Double, seconds: Double) -> String {
        if ended, let freeze = freezeSeconds, let last = lastSignAt, seconds - last >= freeze - 1e-6 {
            // Long silence: the settled tail is final; this sign starts a new sentence.
            var final = tokens
            if let tail = final.last, !isIn(tail, ".?!") { final.append(".") }
            frozen = final
            open = mergeFingerspelling(glosses, scores, known: models.known).glosses.count
        }
        glosses.append(gloss); scores.append(score)
        lastSignAt = seconds; ended = false
        view(ended: false)
        return text
    }

    /// End of turn: no new sign for endSeconds settles the tail (provisionally). nil: nothing changed.
    func tick(_ seconds: Double) -> String? {
        guard !ended, let last = lastSignAt, seconds - last >= endSeconds - 1e-6 else { return nil }
        ended = true
        view(ended: true, final: true, tailPeriod: false)
        return text
    }

    /// Explicit Finish: the final caption of the whole utterance. The caller resets afterwards.
    func finish() -> String {
        guard !glosses.isEmpty else { return text }
        view(ended: true)
        return text
    }

    private func view(ended: Bool, final: Bool = true, tailPeriod: Bool = true) {
        let merged = mergeFingerspelling(glosses, scores, known: models.known)
        tokens = step(merged.glosses, merged.scores, ended: ended, final: final, tailPeriod: tailPeriod)
        text = restore(detokenize(tokens), merged.names)
    }

    /// CaptionEngine.step over the whole (merged) prefix; the runtime context is 0 earlier signs.
    private func step(_ g: [String], _ s: [Double], ended: Bool, final: Bool, tailPeriod: Bool) -> [String] {
        let n = g.count, o = min(open, n), c0 = o
        let tags = models.liveTags(Array(g[c0...]), Array(s[c0...]), ended: ended)
        let commit = ended ? n : max(o, n - 1)
        var newFrozen = frozen, start = o, toks: [String] = []
        if commit > o {
            for i in o..<commit where words(tags[i - c0].2).contains(where: { $0.contains(where: { ".?!".contains($0) }) })
                || (ended && i == n - 1) {
                let clause = close(g, s, start, i + 1, Array(tags[(start - c0)..<(i + 1 - c0)]),
                                   final: final || !(ended && i == n - 1))
                if ended { toks += clause } else { newFrozen += clause }
                start = i + 1
            }
        }
        if !ended { frozen = newFrozen; open = start; toks = [] }
        if commit > start { toks += tokensOf(tags[(start - c0)..<(commit - c0)]) }
        if !ended && commit < n {
            toks += tokensOf(models.draftTags(Array(g[c0...]), Array(s[c0...]))[(commit - c0)...].map { ($0.0, $0.1, "") })
        }
        if ended && !tailPeriod, toks.last == "." { toks.removeLast() }
        return newFrozen + toks
    }

    private func close(_ g: [String], _ s: [Double], _ a: Int, _ b: Int, _ slice: [(String, String, String)],
                       final: Bool) -> [String] {
        let live = tokensOf(slice)
        let (fin, moved) = models.finalize(Array(g[a..<b]), Array(s[a..<b]))
        if needsRevision(live, fin, moved) { return fin }
        if final, let end = fin.last, isIn(end, ".?!"), !(live.last.map { isIn($0, ".?!") } ?? false) {
            return live + [end]
        }
        return live
    }
}
