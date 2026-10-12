// The iPhone app shell: one landscape window, five pages, the reel pipeline behind LIVE/PRACTICE.
// Port of scripts/app_shell_v17.py (page routing, practice drill, history), with the LIVE page
// processing only after START (user request) and the camera off on every other page.

import AVFoundation
import UIKit

final class AppShellViewController: UIViewController, AVSpeechSynthesizerDelegate {
    private let camera = ReelCamera()
    private let sessions = ShellSessionStore()
    private let speech = AVSpeechSynthesizer()
    private var sound: AVAudioPlayer?

    private let cameraHost = UIView()
    private let nav = ShellNavBar()
    private let hud = ReelHUDView(topInset: 50)
    private let overlay = PracticeOverlay(topInset: 50)
    private let home = HomePage()
    private let glosses = GlossesPage()
    private let detail = GlossDetailPage()
    private let practiceSetup = PracticeSetupPage()
    private let practiceResult = PracticeResultPage()
    private let history = HistoryPage()

    private var page = "HOME"
    private var status = "warming up…"
    private var liveRunning = false
    private var finishing = false

    // Practice drill (setup -> running -> done), as in the desktop shell.
    private var practiceState = "setup"
    private var practiceSet: [String] = []
    private var practiceIndex = 0
    private var practiceScore = 0
    private var practiceAttempts = 0
    private var practiceMissed: [String] = []
    private var target: String?
    private var verdict: (gloss: String, at: Date)?

    override var supportedInterfaceOrientations: UIInterfaceOrientationMask { .landscape }
    override var preferredInterfaceOrientationForPresentation: UIInterfaceOrientation { .landscapeRight }
    override var prefersStatusBarHidden: Bool { true }
    override var prefersHomeIndicatorAutoHidden: Bool { true }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = HUD.ink
        speech.delegate = self
        if let url = Bundle.main.url(forResource: "Glass", withExtension: "aiff") {
            sound = try? AVAudioPlayer(contentsOf: url)
            sound?.prepareToPlay()
        }
        cameraHost.backgroundColor = .black
        cameraHost.layer.addSublayer(camera.previewLayer)
        for v in [cameraHost, hud, overlay, home, glosses, detail, practiceSetup, practiceResult, history] as [UIView] {
            view.pin(v)
        }
        nav.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(nav)
        NSLayoutConstraint.activate([
            nav.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor),
            nav.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor),
            nav.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 2),
        ])
        wire()
        camera.load()
        show("HOME")
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        camera.previewLayer.frame = cameraHost.bounds
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        setNeedsUpdateOfSupportedInterfaceOrientations()
        view.window?.windowScene?.requestGeometryUpdate(.iOS(interfaceOrientations: .landscape))
    }

    // MARK: wiring

    private func wire() {
        nav.onSelect = { [weak self] in self?.go($0) }
        home.onAction = { [weak self] in self?.go($0) }
        glosses.onSelect = { [weak self] label in
            self?.detail.show(label)
            self?.showDetail(true)
        }
        detail.onBack = { [weak self] in self?.detail.stop(); self?.showDetail(false) }
        detail.onPractice = { [weak self] label in
            // One sign straight from its tile: a round of exactly that gloss.
            self?.detail.stop()
            self?.startPractice([label])
        }
        practiceSetup.onStart = { [weak self] size in self?.startPractice(size: size) }
        practiceResult.onAgain = { [weak self] in self?.startPractice(size: max(self?.practiceSet.count ?? 5, 1)) }
        practiceResult.onDone = { [weak self] in self?.practiceState = "setup"; self?.go("HOME") }
        overlay.onSkip = { [weak self] in self?.skipPractice() }
        hud.onStart = { [weak self] in self?.toggleLive() }
        hud.onFinish = { [weak self] in self?.finishUtterance(stopping: false) }
        hud.onReset = { [weak self] in self?.resetLive() }
        camera.onReady = { [weak self] in
            guard let self else { return }
            self.status = "ready"
            self.hud.loaded = true
            self.refreshChrome()
        }
        camera.onError = { [weak self] message in
            self?.status = message
            self?.refreshChrome()
        }
        camera.onUpdate = { [weak self] update in self?.received(update) }
    }

    // MARK: routing

    private var practising: Bool { page == "PRACTICE" && practiceState == "running" }

    func go(_ next: String) {
        guard next != page || next == "GLOSSES" else { return }
        if (next == "LIVE" || next == "PRACTICE") && !camera.ready && next != page {
            if next == "LIVE" { return }            // the button reads WARMING UP until then
        }
        // A half-grown candidate must not survive a trip to another page.
        if page == "LIVE" && liveRunning { stopLive(saveSentence: false) }
        if practising && next != "PRACTICE" { endRound(showResult: false); practiceState = "setup" }
        show(next)
    }

    private func show(_ next: String) {
        page = next
        detail.stop()
        showDetail(false)
        nav.setActive(next)
        home.isHidden = next != "HOME"
        glosses.isHidden = next != "GLOSSES"
        history.isHidden = next != "HISTORY"
        practiceSetup.isHidden = !(next == "PRACTICE" && practiceState == "setup")
        practiceResult.isHidden = !(next == "PRACTICE" && practiceState == "done")
        let cameraPage = next == "LIVE" || practising
        cameraHost.isHidden = !cameraPage
        hud.isHidden = !cameraPage
        hud.minimal = practising
        overlay.isHidden = !practising
        if !practising { overlay.stop() }
        if cameraPage { camera.startCapture() } else { camera.stopCapture() }
        UIApplication.shared.isIdleTimerDisabled = cameraPage
        if next == "HISTORY" { history.reload() }
        if next == "HOME" { home.enter() }
        refreshChrome()
    }

    private func showDetail(_ visible: Bool) {
        detail.isHidden = !visible
        glosses.isHidden = page != "GLOSSES" || visible
    }

    private func refreshChrome() {
        nav.setNote(status)
        home.update(ready: camera.ready, status: status, sessions: ShellSessionStore.rows().count)
    }

    // MARK: live page

    private func toggleLive() {
        if liveRunning { stopLive(saveSentence: true); return }
        guard camera.ready else { return }
        liveRunning = true
        hud.running = true
        hud.clear()
        sessions.begin()
        sessions.note("start")
        camera.setProcessing(true)
    }

    private func stopLive(saveSentence: Bool) {
        liveRunning = false
        hud.running = false
        camera.setProcessing(false)
        hud.clearSkeleton()
        if saveSentence {
            finishUtterance(stopping: true)
        } else {
            camera.reset()
            hud.clear()
            sessions.end(complete: false, timing: [:])
        }
    }

    private func resetLive() {
        camera.reset()
        speech.stopSpeaking(at: .immediate)
        finishing = false
        hud.finishing = false
        hud.clear()
        sessions.note("reset")
    }

    /// FINISH (or STOP): flush the open segment, render the clause-split sentence and speak it.
    private func finishUtterance(stopping: Bool) {
        guard camera.ready else { return }
        camera.finish { [weak self] tail, words in
            guard let self else { return }
            self.commit(tail)
            self.hud.endOfStream()
            if words.isEmpty {
                self.hud.setSentence("No recognized signs to finish.")
                if stopping { self.sessions.end(complete: true, timing: self.camera.timing) }
                return
            }
            self.finishing = true
            self.hud.finishing = true
            self.hud.setSentence("Finishing...")
            self.camera.render(words) { sentence, clauses, ms in
                self.finishing = false
                self.hud.finishing = false
                self.hud.setSentence(sentence)
                self.say(sentence)
                self.sessions.sentence(sentence, clauses: clauses, ms: ms)
                if stopping { self.sessions.end(complete: true, timing: self.camera.timing) }
            }
        }
    }

    private func commit(_ words: [LiveWord]) {
        hud.commit(words)
        for w in words {
            sessions.word(w)
            say(w.gloss.hasPrefix("fs-") ? LiveStage3.spelledText(w.gloss) : w.gloss.lowercased())
        }
    }

    private func received(_ update: ReelCamera.Update) {
        if page == "LIVE" && liveRunning {
            for w in update.committed { sessions.word(w) }
            for w in update.committed { say(w.gloss.hasPrefix("fs-") ? LiveStage3.spelledText(w.gloss) : w.gloss.lowercased()) }
            hud.apply(update)
        } else if practising {
            hud.apply(update)
            checkPractice(update.committed)
        }
    }

    // MARK: practice

    private func startPractice(size: Int) {
        startPractice(Array(GlossCatalog.examples.map(\.label).shuffled().prefix(max(1, size))))
    }

    private func startPractice(_ set: [String]) {
        guard camera.ready, !set.isEmpty else { return }
        if page == "LIVE" && liveRunning { stopLive(saveSentence: false) }
        practiceSet = set
        practiceIndex = 0
        practiceScore = 0
        practiceAttempts = 0
        practiceMissed = []
        practiceState = "running"
        target = set[0]
        verdict = nil
        page = ""                    // force the page switch even from PRACTICE
        show("PRACTICE")
        hud.clear()
        hud.running = true
        camera.setProcessing(true)
        refreshRound()
    }

    private func refreshRound() {
        guard let target else { return }
        let missed = verdict.flatMap { Date().timeIntervalSince($0.at) < 3 ? $0.gloss : nil }
        overlay.show(target: target, index: practiceIndex, total: practiceSet.count,
                     score: practiceScore, attempts: practiceAttempts, missed: missed)
    }

    /// The first new word is judged; either way the next attempt starts from nothing.
    private func checkPractice(_ words: [LiveWord]) {
        guard let target, let first = words.first else { refreshRound(); return }
        let clean = first.gloss.hasPrefix("fs-") ? String(first.gloss.dropFirst(3)) : first.gloss
        say(clean.lowercased())
        practiceAttempts += 1
        if clean == target {
            practiceScore += 1
            verdict = nil
            overlay.celebrate(clean)
            sound?.currentTime = 0
            sound?.play()
            advancePractice()
        } else {
            verdict = (clean, Date())
        }
        clearBuffer()
        refreshRound()
    }

    private func skipPractice() {
        guard practising, let target else { return }
        practiceAttempts += 1
        practiceMissed.append(target)
        verdict = nil
        advancePractice()
        clearBuffer()
        refreshRound()
    }

    private func advancePractice() {
        practiceIndex += 1
        if practiceIndex >= practiceSet.count {
            endRound(showResult: true)
            return
        }
        target = practiceSet[practiceIndex]
    }

    private func endRound(showResult: Bool) {
        camera.setProcessing(false)
        hud.running = false
        hud.clearSkeleton()
        target = nil
        overlay.stop()
        guard showResult else { return }
        practiceState = "done"
        practiceResult.show(score: practiceScore, total: practiceSet.count, missed: practiceMissed)
        // Let the celebration finish over the camera before the result page.
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { [weak self] in
            guard let self, self.practiceState == "done", self.page == "PRACTICE" else { return }
            self.page = ""
            self.show("PRACTICE")
        }
    }

    private func clearBuffer() {
        camera.reset()
        hud.clear()
    }

    // MARK: speech

    private func say(_ text: String) {
        speech.speak(AVSpeechUtterance(string: text))
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        hud.speaking = true
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        if !synthesizer.isSpeaking { hud.speaking = false }
    }
}
