import Foundation
final class LiveLatestFrameGate<Frame> {
    private let lock = NSLock()
    private var enabled = false, busy = false
    private var generation = 0
    private var pending: Frame?
    func setEnabled(_ value: Bool) {
        lock.lock(); defer { lock.unlock() }
        enabled = value; generation &+= 1; pending = nil
    }
    var activeGeneration: Int? {
        lock.lock(); defer { lock.unlock() }
        return enabled ? generation : nil
    }
    func isCurrent(_ value: Int) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return enabled && generation == value
    }
    func offer(_ frame: Frame, generation expected: Int? = nil) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard enabled, expected == nil || expected == generation else { return false }
        pending = frame
        guard !busy else { return false }
        busy = true
        return true
    }
    func take() -> (Frame, Int)? {
        lock.lock(); defer { lock.unlock() }
        guard enabled, let frame = pending else { return nil }
        pending = nil
        return (frame, generation)
    }
    func complete() -> Bool {
        lock.lock(); defer { lock.unlock() }
        if enabled && pending != nil { return true }
        busy = false
        return false
    }
}


let gate = LiveLatestFrameGate<Int>()
assert(!gate.offer(0))
gate.setEnabled(true)
let old = gate.activeGeneration!
assert(gate.offer(1, generation: old))
assert(gate.take()!.0 == 1)
for i in 2...100 { assert(!gate.offer(i, generation: old)) }
assert(gate.complete())
assert(gate.take()!.0 == 100)
assert(!gate.complete())
assert(gate.offer(101))
gate.setEnabled(false)
assert(gate.take() == nil)
gate.setEnabled(true)
assert(!gate.offer(999, generation: old))
assert(!gate.offer(103))
assert(gate.complete())
assert(gate.take()!.0 == 103)
assert(!gate.complete())
print("Latest-frame queue and rotation epoch checks PASS")
