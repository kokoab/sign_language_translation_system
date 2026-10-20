package com.example.slt_mobile_app.live

/** iOS framing policy: require a persistent problem for one second before showing a hint. */
class LiveFraming {
    private val since = mutableMapOf<Int,Double>()
    private var checked = false
    private var personSeen = false
    private var shouldersSeen = false
    fun reset() { since.clear(); checked=false; personSeen=false; shouldersSeen=false }
    fun update(o: LiveObservation, brightness: Double?, seconds: Double): String? {
        if (o.faceForFeatures) {
            shouldersSeen = o.body.size >= 2 && o.body[0].c > 0 && o.body[1].c > 0
            personSeen = shouldersSeen || o.face.any { it.c > 0 }
            checked = true
        }
        val hands = listOfNotNull(o.left,o.right)
        val edge = hands.any { h -> h.points.any { it.c > 0 && (it.x < .02f || it.x > .98f || it.y < .02f || it.y > .98f) } }
        val checks = listOf(
            "Too dark. Move somewhere brighter or face the light." to ((brightness ?: 255.0) < 45),
            "Step into view so ATLAS can see you." to (checked && !personSeen && hands.isEmpty()),
            "Step back so both shoulders are in view." to (checked && personSeen && !shouldersSeen),
            "Keep your hands inside the frame." to edge)
        var shown: String? = null
        for ((i,check) in checks.withIndex()) {
            if (!check.second) since.remove(i)
            else { val start=since.getOrPut(i) { seconds }; if (shown == null && seconds-start >= 1) shown=check.first }
        }
        return shown
    }
}
