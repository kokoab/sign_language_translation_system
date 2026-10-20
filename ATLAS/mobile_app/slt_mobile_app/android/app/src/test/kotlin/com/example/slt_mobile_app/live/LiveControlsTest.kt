package com.example.slt_mobile_app.live

import org.junit.Assert.*
import org.junit.Test

class LiveControlsTest {
    private fun hand(open: Boolean, x: Float): LiveHand {
        val p = Array(21) { LivePoint(x,.48f,1f) }
        p[0]=LivePoint(x,.5f,1f)
        p[2]=LivePoint(x-.025f,.46f,1f); p[3]=LivePoint(x-.05f,.42f,1f); p[4]=LivePoint(x-.075f,.38f,1f)
        for ((k,mcp) in listOf(5,9,13,17).withIndex()) {
            val fx=x+(k-1)*.025f
            p[mcp]=LivePoint(fx,.4f,1f); p[mcp+1]=LivePoint(fx,.35f,1f)
            p[mcp+2]=LivePoint(fx,.3f,1f); p[mcp+3]=LivePoint(fx,if(open) .25f else .48f,1f)
        }
        return LiveHand(p)
    }
    @Test fun finishRequiresRightPalmLeftFistAboveFreshShouldersAndHoldsOnce() {
        val gesture=LiveFinishGesture(); val left=hand(false,.3f); val right=hand(true,.7f)
        val body=arrayOf(LivePoint(.3f,.65f,1f),LivePoint(.7f,.65f,1f))
        assertFalse(gesture.update(left,right,emptyArray(),0.0))
        assertFalse(gesture.update(hand(true,.3f),right,body,0.0))
        assertFalse(gesture.update(left,right,body,0.0)); assertTrue(gesture.active)
        assertFalse(gesture.update(left,right,emptyArray(),.9))
        assertTrue(gesture.update(left,right,emptyArray(),1.0))
        assertFalse(gesture.update(left,right,emptyArray(),1.1))
        assertFalse(gesture.update(left,right,emptyArray(),2.0)); assertFalse(gesture.active)
        gesture.reset(); assertEquals(0.0,gesture.progress,0.0)
    }
    @Test fun framingWaitsOneSecondAndPrioritizesLighting() {
        val framing=LiveFraming()
        val empty=LiveObservation(0.0,640,480,null,null,Array(4){LivePoint()},Array(15){LivePoint()},true)
        assertNull(framing.update(empty,20.0,0.0))
        assertNull(framing.update(empty,20.0,.9))
        assertTrue(framing.update(empty,20.0,1.0)!!.startsWith("Too dark"))
        assertTrue(framing.update(empty,100.0,1.1)!!.startsWith("Step into view"))
        framing.reset(); assertNull(framing.update(empty,100.0,2.0))
    }
}
