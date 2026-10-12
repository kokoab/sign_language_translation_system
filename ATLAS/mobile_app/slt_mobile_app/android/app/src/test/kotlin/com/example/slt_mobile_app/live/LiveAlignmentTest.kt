package com.example.slt_mobile_app.live

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.ln

class LiveAlignmentTest {
    @Test fun delayedClosureDoesNotRewindWordEndpoints() {
        val word = LiveWord("MY", 2, 10, 12, 0.1, 0.5, 0.6, 0.9, true)
        word.extendTo(24, 1.2)
        word.extendTo(20, 1.0)
        assertEquals(24, word.endFrame)
        assertEquals(1.2, word.endSeconds, 0.0)
        assertEquals(12, word.commitFrame)
        assertEquals(0.6, word.commitSeconds, 0.0)
        word.extendTo(30, 1.5)
        assertEquals(30, word.endFrame)
        assertEquals(1.5, word.endSeconds, 0.0)
    }

    @Test fun configuredBoundaryDipKeepsWholeGestureCandidate() {
        fun row(state: Int): FloatArray = FloatArray(4) { ln(if (it == state) 0.85f else 0.05f) }
        val bio = listOf(row(1), row(2), row(3), row(3), row(1), row(1), row(1), row(1), row(3), row(3), row(3), row(1))
        val whole = SpanKey(1, 10)
        assertFalse(LiveSegmental.bothSpans(bio).contains(whole))
        assertTrue(LiveSegmental.bothSpans(bio, minLen = 3, maxInnerO = 8).contains(whole))
        assertFalse(LiveSegmental.bothSpans(bio, minLen = 12, maxInnerO = 8).contains(whole))
    }
}
