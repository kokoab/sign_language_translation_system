# mobile-deployment — log

Measured results, rejected approaches, and progress snapshots. Not read start-to-end —
`rg` this file before re-running an experiment to see if it already failed.

11 entries, newest first. Archived from `PROJECT_GROUND_TRUTH.md` 2026-09-10.

---

## 2026-10-06 — Android live-camera empty-scene smoke passes

Parent opened Flutter home/models ready, LANDMARKS default, native Live and Start.
Frame counter advanced to265 without crash; snapshot10.3FPS/latest85ms, zero hands.
This establishes camera/MediaPipe execution on an empty scene only, not signer
accuracy or complete signed-input latency. Finish/Back stopped it; any empty smoke
sessions remain in history(no deletions). Report and snapshot:
artifacts/reports/android_runtime_pool_isolation_20261006/REPORT.md.
Restored temporary log.tag.SLTParity property to initial empty state. Build success
and paired debug diagnostics verified; production model/runtime unchanged. Next:
native profiling under matched conditions and signer/sustained complete-pipeline
acceptance. No real-time readiness claim; exact token savings remain unavailable.

## 2026-10-06 — parent-led Android runtime experiments complete; no safe speed workaround established

Completed three parent-authored paired microbenchmarks via Luna completion watcher. Boundary-pool
isolation: isolated medians241.8/324.3ms vs combined281.9/291.0ms, no consistent gain. External
run-as affinity denied; debug in-app taskset own inference thread succeeds(ff→f0), but pinned
medians340.5/352.1ms vs default190.3/185.4ms. No production affinity change. Direct-buffer
comparison: wrapper196.1/298.6ms; direct no-write265.8/185.2ms; direct write+run250.5/185.0ms.
Input copies/wrapper bypass do not explain native execution variability. All4threads,-4priority,
30samples/component/run,ABBA. Process CPU/run captures all app threads, not host load percentage.

Only app change is debug LiveParityActivity:isolated/direct/big_cores flags, bounded in-app affinity
command, process CPU timing and Log.println(INFO) completion logging (Huawei Log.i suppressed).
Parent authored/reviewed exact runners and interpreted results; Luna executed/waited, without
new independent fixes. Build caps and cleanup retained; no unrestricted retry. APK installed,
latest date10:30:27, after parent confirmed Huawei's already-authorized install prompt. Separate
push14-15s proves transfer not bottleneck. Successful constrained debug rebuilds14-22s. No
production inference/model/config changes. Saved recognition89/89,frame5161/5161 andStage3
sentence89/89 parity remain; no new protected evaluation. Root observed Flutter home launch;
complete camera/signer/stress acceptance remains open.

Report artifacts/reports/android_runtime_pool_isolation_20261006/REPORT.md; source JSONs
paired_results_v3,affinity_results_v1,direct_results_v1. Updated Android current state. Exact
token savings cannot be measured; supported mailbox completion wake-up worked and parent did
not repeatedly poll runs. Next safe diagnostic:native operator/thread/frequency tracing under
matched conditions. Do not claim real-time readiness or apply pinning/pool removal as a fix.

## 2026-10-06 — parent-reviewed contained build succeeds; phone installer waits behind proximity guard

Parent reviewed worker's debug isolated benchmark branch and exact build command. Standard JDK
version/class/module diagnostics pass; no proven persistent JDK corruption. After stale-daemon
cleanup, contained gradlew help passes and assembleDebug exits0 (1m29s,141tasks). Controls:
--no-daemon --max-workers=2, JAVA_TOOL_OPTIONS ActiveProcessorCount2/Xmx2g, overridden Gradle
JVM args ActiveProcessorCount2/Xmx2G/MaxMetaspaceSize1G, instrumentation disabled, external
180s timeout and task-owned process cleanup. These restrict concurrency, not a measured hard
CPU-percentage ceiling. APK511,928,351bytes; parent ZIP CRC check passes all827members and
finds the debug benchmark-mode string in dex. No production runtime behavior changed.

Streaming adb installs timed out. Parent-authored separate transfer/install runner pushes APK
successfully14.44s/34.1MB/s, but pm install exceeds120s. Package remains at2026-10-05 install.
One-shot device diagnosis finds InstallStaging behind keyguard with focus Emui:ProximityWnd;
phone sensor/lock state must be resolved. Asked user asynchronously to uncover/wake/unlock.
No blanket settings/security change made. Parent authored paired benchmark runner with ABBA
order,4threads,display priority,30samples/run,run-as big-core affinity f0 and blocking logcat
BENCH done/FAILED events; not yet executed while install is unresolved.

User clarified lower-cost agent should watch, parent should own technical decisions. Parent
now reviews changes, authors exact runners, and limits Luna to executing/waiting/completion
messages. Blocking mailbox waits replace parent polling; actual token counters unavailable,
so net token savings remain unmeasured. No desktop notifications. Next: resolve device-side
installer, then run paired experiment and validate any evidence-backed runtime change.

## 2026-10-06 — isolated Android benchmark blocked by Gradle daemon CPU spin

Added a debug-only `isolated` option in `LiveParityActivity.kt` to benchmark the landmark span recognizer without first constructing or warming the boundary model. Combined mode remains the default; production runtime code is unchanged.

Flutter debug build failed with `Incompatible magic value 1784772193` in Java class loading and Gradle lock cleanup reported `Device not configured`. The completed Gradle 8.14 daemon then remained at 919.2% CPU (PID 34615, PPID 1); `jcmd` could not attach. A retry configured for two Gradle workers and disabled instrumentation still drove the retry daemon to 309% CPU, so its task-owned wrapper and daemon were terminated (Flutter exit 143). No APK was produced, installed, or benchmarked; there are no isolated/combined timing results.

Report: `artifacts/reports/android_runtime_pool_isolation_20261006/REPORT.md`. Next safe action: diagnose the class-format failure, then retry only with `--no-daemon`, at most two workers, JVM `-XX:ActiveProcessorCount=2`, and a hard external timeout; verify the CPU bound before device execution.

## 2026-10-06 — user forbids desktop notifications; automatic re-entry unavailable

User clarified completion must notify the agent, never the user. Removed both osascript
desktop notifications from the diagnostic runner. Its completed run is not being relaunched.
Same-session CLI resume already failed due to an active writer; no exposed background
completion hook can wake this assistant after ending a turn. Do not promise automatic
re-entry or use desktop notifications as a substitute. Results remain preserved. No
production app changes. Continue only with supported execution/completion handling.

## 2026-10-06 — completed Android diagnostics; Stage3 parity passes, self-resume failed

After user follow-up, read completed one-shot results once (no periodic status checks).
Stage3 comparison passes89/89 sentences against Python. Two30-sample app batch8 runs:
4threads display median333.5/279.6ms;2threads349.2/402.8ms;1thread667.6/732.5ms;
4threads urgent-display254.6/245.7ms. Standalone big-core averages1/2/4threads383.957/
198.418/108.799ms. Priority/affinity are not matched; sequential runs may have state
confounds. App single-thread slowdown this session means parallel scaling alone is
not established as the cause. No production runtime changes made or improvement claimed.

The command-exit callback attempted same-session Codex exec resume, but the active desktop
writer caused thread-store conflict. Automatic return did not work; do not reuse it as a
verified callback. Result report: artifacts/reports/android_runtime_takeover_20261006/REPORT.md.
Updated PROJECT_GROUND_TRUTH.md to mark Stage3 sentence parity complete. Next safe action:
isolated recognizer versus boundary-pool debug comparison with matched execution conditions,
then saved-fixture and live-camera validation. Existing data/test gates remain intact.

## 2026-10-06 — Android takeover authorized; event-driven diagnostics launched

User authorized continuing the Android MediaPipe/TFLite performance/parity work, explicitly forbidding
polling and requesting a completion-triggered return to this session. Created syntax-checked one-shot
runner `artifacts/generated/android_runtime_takeover_20261006.py`. It compares the installed app's
landmark batch8 with1/2/4 threads and display/urgent-display priority (two runs of30 samples), then
standalone pinned-core benchmark_model; independent Stage3 Python reference comparison uses the
already-saved89 phone sentence results. Device is connected (Huawei MGA-LX9; battery59%,31.0C).
No new model selection/test access. No production app changes yet.

Runner blocks on streamed BENCH done/FAILED logcat events, subprocess exit and bounded timeout;
it does not repeatedly check status. Outputs: `artifacts/reports/android_runtime_takeover_20261006/`.
On completion/failure it emits a macOS notification and invokes installed Codex CLI `exec resume`
for this exact session01a10ec8-5739-7302-b607-2365bf04c192, with the authorized follow-up task and
result paths. Resume output is preserved in resume.log and FINAL_RESULTS.md. This is a same-session
continuation, not an independent delegated task; desktop UI delivery of the resumed answer is not
verified. Next action after completion notification: review results, diagnose the scaling cause,
validate selected changes without polling and report evidence/limitations. Preserve existing changes.

## 2026-10-06 — Android integration status reconciled with completed phone parity

Read-only implementation review requested by user, including pasted 2026-10-05 runtime tuning
note. Confirmed saved `artifacts/generated/android_parity_results_tune_20261005.json` summary:
Huawei MGA-LX9, 89/89 video matches, zero frame commitment mismatches across 5,161 observations.
Debug LiveParityActivity injects Python-extracted observations into Kotlin landmark-only boundary/
segmental runtime; this establishes fixture parity, not live camera/MediaPipe extraction parity.
Observe mean247.29/median210.77/p90524.31ms excludes camera/MediaPipe and Stage3. Stage3 fields
exist for89 fixtures, all t5_efficient_tiny, mean605.59ms/sentence; no saved Python sentence
comparison found in the targeted generated-output search. Do not count rendering as sentence parity.

Current app LiveModels.kt uses LiteRT CompiledModel + CPU CpuOptions (up to4 threads). Its dated
comment documents the Interpreter XNNPACK path staying single-threaded (~380ms batch8 across
requested thread counts). Pasted tuning note reports one-thread stable~364ms, four-thread
min155/median274.8/p90432.3ms versus standalone benchmark~113ms batch8. Multiple model thread
pools are a hypothesis from that note, not a confirmed cause. Correctness passed; in-app scaling
and real-time performance remain unresolved. CameraX KEEP_ONLY_LATEST, native MediaPipe,
Live/Practice activities, finish handling, session storage, T5 and Flutter mode controls exist;
no live-camera acceptance result found in reviewed evidence.

Changed only this log and the Android current-state paragraph in PROJECT_GROUND_TRUTH.md to
replace stale install/parity-pending wording. No app/model changes, builds, training or device
runs performed. Next safe action: diagnose app latency, then verify Stage3 sentence parity,
camera/frontend behavior and sustained complete-pipeline performance; preserve test gates.

## 2026-10-06 — expanded distribution review; offline link-local USB peer reachable

User requested broader investigation after the managed USB bridge failure. Reviewed
Apple docs, original GitHub sources, first-hand Reddit/StackOverflow reports and the
USENIX2026 sideloading study. New candidates: direct link-local USB; Mac loopback-source
offline Wi-Fi; small App Store localhost host plus locally supplied assets; opposite
direction Personal Hotspot USB. Public enterprise certificates were considered, not
downloaded/used: Apple requires initial internet verification and periodic revalidation,
with Allow & Restart on18+; no dependable7day external-research route established.
JS Shell PH App Store listing is free/6.6MB/iOS17+ and claims localhost, persistent
files/camera support; not an installed/tested SLT host. HTML Serve is iPad-only. Do
not describe any web/interpreter host as preserving the complete native pipeline.

Read local AssetCacheTetheratorUtil manual (root needed to change state); no sudo
authentication available and no third-party scripts executed. Inspected ordinary
Internet Sharing separately: selected inactive Thunderbolt source and USB targets,
briefly enabled; no managed bridge established. Restored Wi-Fi source, all targets
off, sharing off. No Wi-Fi AP broadcast or loopback service created.

Crucial follow-up: en7 remained active with Mac169.254.167.131 and USB peer169.254.92.174,
same peer MAC as prior bridge. With Content Caching/tetherator/ordinary sharing off,
turned Mac Wi-Fi off: 2/2 ping replies at2,8,16seconds (6/6) and no upstream default
route. Restored Wi-Fi in finally. This is warm already-paired/developer-enabled device
network reachability, not Safari fetch, cold pairing, replug, reboot or general iOS17+
proof. Interface may depend on earlier tethered/developer-tool activation; origin
not established. Previous report's bridge failure is valid but does not rule out USB.

Evidence: artifacts/reports/iphone_usb_delivery_20261006/linklocal_probe.json and
RECONSIDERED_OPTIONS.md. Added follow-up to REPORT.md/current ground truth. No models,
app, training gates, participant accounts, profiles or certificate trust changed.
Link-local evidence assertions passed after correcting the check to read tetherator
status from stderr (initial assertion wrongly checked stdout only). Sharing restoration
verified in UI; targeted diff --check passed. Regenerated LARGE_FILES.md.
Next: prioritize fresh-initialization and trustworthy-origin Safari tests on this
direct USB path; keep Mac-only AP and one-small-host-install as distinct compromises.

---

## 2026-10-06 — USB phone page delivery passed; upstream-free configuration failed

User confirmed the connected phone displayed the probe and authorized autonomous
completion/reporting. Server recorded a 200 GET from USB peer 192.168.234.2. Native
SLT app, models and datasets unchanged. Two bounded Mac Wi-Fi-off probes restored
Wi-Fi in finally blocks. First: bridge100 disappeared, 0/5 ping replies, local page
unreachable, no default route. Repeat: initial bridge/ping passed, but bridge absent
and ping failed at 5 and 12 seconds; devicectl still reported connected/paired/wired.
After restoring Mac Wi-Fi, USB bridge and ping recovered after a delay. Cache status
remained Active=true/TetheratorStatus=1 even during failure; those flags are insufficient.
This tests no active Wi-Fi/Ethernet on the Mac, not an associated network without WAN
or every offline USB method. Do not claim all USB methods require internet.

Mac Safari plain-HTTP USB-origin diagnostic: secureContext=false, serviceWorkerAvailable
and cameraAPIAvailable=false. Not an iPhone diagnostic. Phone remote inspection showed
Enable Web Inspector on device; no phone inspection utility/security setting changed.
No automated phone reload, reboot, Home Screen installation, camera or full inference
test performed. No new app/model port and no training/test gates consumed.

Reports/evidence: artifacts/reports/iphone_usb_delivery_20261006/REPORT.md and
APP_FEASIBILITY.md, with bounded network JSON evidence and disposable HTML probes.
Evidence assertions passed (offline failure, wired connection, browser diagnostic and
cleanup); targeted git diff --check passed. Regenerated artifacts/LARGE_FILES.md as
required after report creation; preserved unrelated worktree changes.
Restored Mac Wi-Fi on, Content Caching off, USB connection sharing off, All Content
original selector, Safari developer menu off. Stopped server; no port8765 listener,
USB bridge absent. System-managed393KB caching remains; no purge. Phone radios remain
as user left them. Current state paragraph added to PROJECT_GROUND_TRUTH.md.

Decision: tested tethered-sharing setup is not venue-ready; HTML USB success is not
complete offline SLT deployment. Next safe action: discuss the measured transport
failure and separate trusted-HTTPS/full-browser-pipeline feasibility. Offline Android
hotspot remains a conditional candidate, not a selected/verified complete solution.

---

## 2026-10-06 — connected iPhone USB browser transport probe prepared

User authorized trying USB delivery on their currently connected phone. Read-only
inspection found paired iPhone 13, wired transport, iOS 26.7.1 and Developer Mode
already enabled; this is not representative proof for iOS 17 or a new participant.
Initially en6 (iPhone USB) was inactive and Content Caching/USB sharing were off.
Through System Settings temporarily enabled Content Caching with Internet Connection
sharing. Cache selector was observed as All Content at activation (attempted Only
Shared Content did not persist); cache remained zero at the inspected snapshot.
Mac Wi-Fi upstream remains on. No hotspot or SLT app/model installation performed.

Activation created bridge100 at 192.168.234.1 with USB member en7 and an iPhone peer
192.168.234.2. TetheratorStatus=1. Python stdlib HTTP server serves one disposable HTML
page on the USB bridge address only, port 8765, from
/var/folders/p3/wq7yl8sj2m96kncm59f0190h0000gn/T/slt-usb-probe-titn_6og .
Exec server session 9285; no repository files are exposed. Initial 3-second curl
timed out during startup, then server logged a 200 response; repeated local curl
verified the page. Phone Safari fetch is pending user interaction; no phone success
or offline-startup support is claimed. HTTP probe cannot verify camera/service worker.

Next: user opens http://192.168.234.1:8765/ in iPhone Safari with phone Wi-Fi/cellular
off to isolate USB. If successful, test Mac upstream removal separately, then restore
Content Caching/Internet Connection off and original All Content selector; stop server.
Phone radios need restoring after the test. Only this log changed in repository.

---

## 2026-10-05 — offline Android hotspot is a supported network candidate

User has no router, may consider a team phone hotspot but prefers none, and requires
participant setup at the venue rather than beforehand. Android officially supports
LocalOnlyHotspot: nearby devices communicate on a network with no internet access.
This establishes an offline transport candidate, not a working SLT deployment. A team
Android phone could create the network while the Mac supplies assets; client-to-client
reachability on the exact hardware still needs verification. Participant phones need
no hotspot utility. GitHub local-only hotspot projects exist, but none was installed,
audited or selected. Ordinary iPhone Personal Hotspot must not be assumed equivalent.

Mac Internet Sharing documentation describes sharing an upstream connection; an
upstream-free Mac Wi-Fi hotspot has not been verified. Browser camera/service worker
still require a trustworthy origin, and the native Apple Vision/Core ML pipeline still
needs a validated browser alternative before claiming complete-app parity. Offline
hotspot solves transport only. Next safe action: identify the available team phone and
whether one brief offline-hotspot connection is acceptable. No implementation selected;
no app, account or network settings changed. Changed only this log.

Sources: https://developer.android.com/develop/connectivity/wifi/localonlyhotspot ;
https://support.apple.com/en-qa/guide/mac-help/mchlp1540/mac ;
https://github.com/parthbhinde/react-native-local-only-hotspot .

---

## 2026-10-05 — venue has no Wi-Fi; distinguish offline networking from internet

User clarified there is no Wi-Fi at the testing venue and prefers all loading from
Mac storage. Explained Ethernet was a requirement of Apple's proposed tethered internet
sharing, not of model inference or offline operation. That method is not an established
fit for this venue; no wired internet or adapter availability is assumed.

AirDrop uses peer-to-peer Wi-Fi without an access point/internet (Apple security guide),
so it can transfer files at the venue, but file transfer alone does not install a native
app or establish a service-worker PWA. ServiceWorker registration requires an HTTP(S)
URL and trustworthy origin; an AirDropped file:// HTML bundle cannot supply the normal
offline-PWA registration path. Local Mac hosting over a private network is a candidate
only if participant connection to an offline local network is acceptable; trusted HTTPS
and offline restart persistence still need verification. Do not equate a local Wi-Fi
network with internet access, or claim ordinary USB trust gives Safari a Mac web-server
connection. No solution satisfying all constraints is yet verified.

Sources: https://support.apple.com/en-gb/guide/security/sec2261183f4/1/web/1 ;
https://developer.mozilla.org/en-US/docs/Web/API/ServiceWorkerContainer/register .
Changed only this log. Next safe action: clarify whether an isolated local network
created by the team is prohibited, or only reliance on venue Wi-Fi/internet. No app,
network settings, files on phones or models changed.

---

## 2026-10-05 — allowed onboarding fixed; USB-loaded offline web app remains a candidate

User permits USB, unlock/trust-Mac, AirDrop, browser opening, Home Screen addition,
installation/Open confirmation and camera permission. Participant Apple Account login,
Developer Mode and joining Wi-Fi are prohibited; developer-identity trust is only
"maybe". Requires zero budget, complete app, iOS 17+, no mandatory OS update, local
phone inference and offline reopening after app/phone restart and on another day.
Browser-compatible pipeline is acceptable only without regression. One participant
at a time; loading should preferably be supplied locally by Mac/AirDrop.

Native Personal Team/iloader route no longer fits. AirDrop of HTML/IPA alone is not
established as an installable, camera-capable offline PWA. Promising infrastructure
finding: Apple's documented tethered caching shares a Mac's internet connection to
iPhones over USB, with Ethernet internet on the Mac as a requirement. This could
bootstrap a trusted-HTTPS browser app without participant Wi-Fi/account/Developer Mode,
then cache shell/models for disconnected use. This is feasibility only: no iPhone USB
browser connectivity, service worker/camera, full model port or restart persistence test
has run. Apple Content Caching caches supported Apple/iCloud assets, not arbitrary model
files; do not claim it automatically caches our browser models. Strictly local model
serving over USB needs separate endpoint/network/TLS verification.

Apple DTS also documents an iPhone USB-Ethernet adapter reaching a Mac web server on an
isolated Ethernet network; this needs existing/borrowed hardware, not assumed purchases.
Safari cached storage is best effort; offline reopen can be implemented and tested but
unconditional persistence under storage pressure is not guaranteed. No implementation
selected. Questions still needed: existing Mac Ethernet access and whether one-time
HTTPS loading through Mac-provided USB internet is acceptable.

Sources: https://support.apple.com/en-ph/guide/deployment/dep38ff24bed/web ;
https://support.apple.com/guide/deployment/intro-to-content-caching-depde72e125f/web ;
https://developer.apple.com/forums/thread/821159 ;
https://webkit.org/blog/14403/updates-to-storage-policy/ . Changed only this log;
no account/network settings, installs, builds, models or tests changed. Next safe action:
resolve infrastructure constraints, then discuss a small connectivity/offline-shell pilot
before any full browser port or non-regression claim.

---

## 2026-10-05 — low-burden browser/PWA and App Clip deployment feasibility

User now prioritizes no manual downloads/installation or Developer Mode for participants;
asked about local models in a PWA and other alternatives. This preference supersedes
iloader as the leading proposal despite earlier acceptance of participant account login.
Questions sent about Safari QR + camera permission, automatic model downloads on provided
Wi-Fi, and whether a distinct browser pipeline or Mac-side inference is acceptable.
Answers pending; discussion only, no implementation authorized or selected.

Primary-source research confirms local browser inference is possible: ONNX Runtime Web
supports Wasm and model caching; LiteRT.js runs .tflite models via Wasm/WebGPU. Safari 26
ships WebGPU on iOS; older ONNX support tables still mark Safari WebGPU unsupported, so
use the WebKit release evidence plus exact-runtime device verification rather than either
table alone. No claim that our models already work or meet latency/accuracy on Safari.
MediaPipe Tasks has web/Safari support. The current Swift Apple Vision/Core ML pipeline
cannot simply execute as browser JavaScript; a browser candidate needs its own extractor/
runtime contracts. Existing MediaPipe Android landmark-only recognizer is 27.8 MB FP32
TFLite (recognizer only, not full web payload); using it on iPhone is a distinct proposed
deployment family requiring authorization and validation, not replacement of Apple inputs.
No feeding MediaPipe landmarks into Apple-trained models is proposed.

Model cache/Service Worker assets can enable offline operation after initial load, but
Safari storage is best effort and can be evicted. Camera capture requires permission and
a secure context; any browser-to-Mac option must use trusted HTTPS without asking users
to install a self-signed certificate. Mac-side raw-video inference could retain Apple's
model family but is network-dependent and not evidence of phone-local inference/speed.

App Clip is a native low-install-burden candidate: QR/link invocation, no full app install,
native frameworks, automatic asset loading; still requires Developer Program distribution/
App Store review. Local test App Clip experiences require per-device setup and do not
solve this onboarding requirement. Six current source packages in
`mobile_app/slt_mobile_app/ios/Runner/LiveReel/Models` total 153.43 decimal MB (two
boundary models, hand encoder, recognizer, T5 encoder/decoder). Not an App Clip variant,
compressed download, browser-model size or RAM measurement. Apple's App Clip size limits
and allowed additional asset downloads mean packaging/loading must be evaluated separately.

Sources: https://webkit.org/blog/17333/webkit-features-in-safari-26-0/ ;
https://developers.google.com/edge/litert/web ;
https://onnxruntime.ai/docs/tutorials/web/large-models.html ;
https://developers.google.com/edge/mediapipe/solutions/setup_web ;
https://webkit.org/blog/14403/updates-to-storage-policy/ ;
https://www.w3.org/TR/mediacapture-streams/ ;
https://developer.apple.com/documentation/appclip/choosing-the-right-functionality-for-your-app-clip ;
https://developer.apple.com/documentation/appclip/creating-an-app-clip-with-xcode .
GitHub Safari resource issues are evidence of possible implementation risk, not failure
rates for our pipeline. No app, installer, model, account or test changes; only this log.
Next safe action: resolve the browser interaction/network/pipeline preferences, then
discuss a narrowly bounded Safari feasibility pilot before building a complete product.

---

## 2026-10-05 — immediate-use deployment: 50 iPhones/day on one Mac

User clarified testing is immediate after installation and never longer than seven
days per participant; expected throughput is ~50 phones/day with one Mac. Preferred
onboarding is phone passcode + trust/authorization only, or phone-to-phone AirDrop.
User explicitly keeps this discussion focused on deployment, not evaluation design.

Verified distinction: USB trust unlocks communication but does not supply the Apple
certificate/provisioning profile required to execute the app. Free central signing
remains limited to three registered devices; deleting an app does not create a new
device slot. Per-participant free signing (iloader/Sideloadly) remains technically
possible, but adds each person's Apple Account authentication/2FA and Developer Mode/
trust setup. Participant subsequently accepted using their own Apple Account login.
No measured per-phone installation time or 50/day throughput claim is established.
AirDrop transports an IPA/link; it does not bypass signing or act as a stock iOS IPA
installer. Configurator similarly installs appropriately provisioned builds; it does
not create a free unlimited signing entitlement. Paid ad hoc is capped at 100 iPhones
per membership year, whereas TestFlight supports the requested hundreds.

Sources: https://support.apple.com/en-gb/guide/security/sec7c917bf14/web ;
https://developer.apple.com/help/account/basics/about-your-developer-account ;
https://developer.apple.com/documentation/xcode/distributing-your-app-to-registered-devices ;
https://developer.apple.com/help/account/devices/devices-overview/ ;
https://github.com/nab138/iloader . No builds, installs or account changes. Changed
only this log. Recommended discussion route is a prebuilt Release IPA + iloader,
signed separately with each participant's account. Official iloader README supports
importing arbitrary IPAs. Source inspection of `src-tauri/src/account.rs` confirms
optional credential persistence (`save_credentials`), stored-account deletion and
in-memory account invalidation. Prefer no credential persistence and clear the active
session between participants; UI behaviour must be verified in the pilot. No installer
has been installed and our IPA has not been signed using it. Next safe action: discuss
the exact installation workflow, then pilot compatibility/account switching and measure
per-phone time before committing to 50/day. AirDrop can share the package, not replace
each recipient's provisioning/install step.

---

## 2026-10-05 — distribution constraint clarified: one Mac setup visit is acceptable

User allows connecting each phone to the Mac once, but cannot log their own Apple
Account into other people's phones. Several hundred phones remains the intended scale.
This supersedes the no-cable constraint in the preceding research entry below.

Discussion only; no implementation selected. Signing in Xcode/installer is separate
from the phone's iCloud account. The user's free signing account still cannot provision
hundreds of devices. Each participant can instead self-sign with their own free Apple
Account (the SideStore guide explicitly says it need not be the phone's iCloud account).
For one short session, computer-side signing/installation gives up to seven days without
SideStore. For ongoing use, SideStore is now a candidate: initial Mac/USB setup, then
install the shared app IPA in SideStore and refresh both apps on the phone before their
seven-day profiles expire. The documented workflow requires Wi-Fi and LocalDevVPN for
refresh; recognition itself remains offline. No model changes are implied. This has not
been tested with our Runner IPA or across participant phones.

"One Mac visit" is a normal-path goal, not a guarantee: SideStore expiry or invalid
pairing can require another computer setup. Official FAQ/prerequisites/common-issues
and project issue reports were checked; reports are evidence of possible support burden,
not measured failure rates. Each participant must handle their own credentials/2FA;
no request to share the user's account or collect participant passwords is proposed.
References: https://docs.sidestore.io/docs/installation/install ;
https://docs.sidestore.io/docs/installation/prerequisites ;
https://docs.sidestore.io/docs/faq ; https://github.com/SideStore/SideStore/issues/906 .

Changed only this log. Next safe action: discuss short-session installs versus ongoing
SideStore use, then authorize a small pilot before planning hundreds of installations.

---

## 2026-10-05 — iPhone distribution research; hundreds of phones, no cable onboarding

Discussion only. User clarified several hundred participants, in-person testing but
installation by a shareable link without connecting each iPhone to the Mac; the school
cannot provide an institutional developer account. No distribution implementation chosen.

Read current ground truth, targeted mobile-deployment history/current diff, and the
external Flutter app at `/Volumes/secret/SLT/mobile_app/slt_mobile_app`. iOS uses native
Swift Apple Vision/Core ML, bundled models, and local session/video storage. Xcode pins
one development team and `com.kokoab.sltMobileApp`; Podfile/project/framework minimum
is iOS 17. Current local Runner.app embedded profile admits one device and lasts exactly
seven days (2026-09-28 14:09:11 UTC to 2026-10-05 14:09:11 UTC). This is build-artifact
evidence, not a fresh inspection of the installed phone. Signing is independent of model
training; a browser version would require a different runtime and new validation.

Verified Apple Personal Team limits: three test devices per platform, three installed
free-provisioned apps per device, seven-day profiles. AltStore/Sideloadly automate
refresh with a computer; SideStore permits later phone-side refresh but documented
initial setup still uses USB/computer, an Apple Account, trust/Developer Mode and a
local VPN. These do not meet the requested hundreds-of-phones link-only workflow.
TestFlight supports 10,000 external testers, 90-day builds, and first-build beta review;
developer membership is required, testers do not pay. An authorized sponsoring publisher
with an existing membership could remove this team's direct fee. Eligible nonprofit/
educational organizations can request a fee waiver; individuals cannot claim it merely
for student/research status. ResearchKit supplies research-app features, not signing.
University of Minnesota's work-zone research report documents TestFlight distribution;
Reddit anecdotes corroborate provisioning/refresh friction but are not policy evidence.

Primary references checked: https://developer.apple.com/support/compare-memberships/ ;
https://developer.apple.com/help/account/basics/about-your-developer-account ;
https://developer.apple.com/help/app-store-connect/test-a-beta-version/testflight-overview/ ;
https://developer.apple.com/help/account/membership/fee-waivers/ ;
https://docs.sidestore.io/docs/installation/install ; https://github.com/SideStore/SideStore ;
https://github.com/altstoreio/AltStore ; https://sideloadly.io/faq.html ;
https://cts-d10resmod-prd.oit.umn.edu/pdf/mndot-2023-04.pdf .

Changed only this research log. No builds, installations, account changes, model changes,
or test-data access. Next safe action: discuss whether an authorized external publisher/
sponsor is available, or which constraint may change (budget, onboarding, native runtime).

---

## 2026-10-04 — Android app builds (Kotlin native layer in the Flutter app); iPhone landmark-only candidates pass floors

Android: words-only Live Reel ported to Kotlin in `mobile_app/slt_mobile_app/android/app/src/main/kotlin/
com/example/slt_mobile_app/live/` (types, features/crops, segmental decoder, LiteRT 2.2.0 models, MediaPipe
Tasks 0.10.14 vision + finish gesture, Stage 3 T5 + incremental translator, engine, CameraX activity) and
`MainActivity` channel (openLiveReel/openPractice/liveReelStatus/listSessions + Android-only
get/setHandImageMode; inferVideo/shareReport return "iPhone only"). Hand-images mode: AUTO uses a one-time
MobileCLIP crop timing (<= 12 ms/crop -> on), Flutter settings sheet override (Automatic / Hand images /
Landmarks). Assets: FP32 TFLite + pinned .task files (268 MB, noCompress); debug APK 512 MB, ARM only.
Backup before edits: `mobile_app/backup_before_android_live_20261004/`. Build gotchas: exFAT `._*` files in
`res/` break AGP (dot_clean -m android/app/src); an interrupted NDK install leaves a stub without
source.properties (delete it). Parity fixtures (89 tune videos, 5,161 frames; reproduce the landmark-only
replay 89/89): `artifacts/generated/android_parity_fixtures_tune_20261004/` via
`scripts/dump_android_parity_fixtures_v17.py`; debug-only `live.LiveParityActivity` replays them on the
phone; `scripts/android_stage3_reference_v17.py` checks T5 sentences. Not yet installed: phone dropped off USB.
iPhone landmark-only (1-point floors kept): share 0.3 no eligible epoch; encoder-lr 5e-6 epoch 1
94.71/88.24/95.65, tune proxy 33.6%; frozen encoders epoch 2 94.71/88.04/96.27, proxy 38.1% (Android
landmark-only proxy 26.1% -> replay 17.70%). Both get Apple-Vision words-only tune replays; held-out once
for the chosen one. Floor-2 candidate not used.

## 2026-10-04 — Android landmark-only recognizer trained: held-out 9.14% WER, ~70 ms/frame on the phone

Same span recipe as the hand-image recognizer but from the MediaPipe landmark branch with no hand images
(`--landmark-only` in train_span_recognizer; runtime `recognizer_kind: landmark_only`). Selected epoch 4.
Isolated 93.92/85.69/95.79 vs Apple 95.24/89.67/96.31 (pass). Live replay: tune 17.70% (hand-image 13.27,
Apple 10.18); held-out 9.14% (hand-image 8.60, Apple 11.83; gate pass) — local60 4.32%, unseen ASLLRP12 41.7%
(hand-image 54.2, Apple 33.3). Mac 24 ms/frame. TFLite FP32 27.8 MB, 0/4,232 changes; phone 108.8 ms/B8 on
4 big cores, 75 MB. Report `artifacts/reports/mediapipe_rebuild_v17_20261004/LANDMARK_ONLY.md`.
Next: Kotlin app with both modes (auto default by first-launch crop-encode timing + settings override).

## 2026-10-04 — Hand-images-off shortcut fails on continuous signing; landmark-only model needed

User decisions: Android keeps two recognition modes — landmark-only (default on low-end phones) and
landmarks + hand images (on capable phones; option 2 GPU work later); automatic first-launch default plus
a manual settings override; hand-image models bundled in the app. Shortcut test (no retraining): the
MediaPipe span recognizer with every hand view masked keeps isolated accuracy close (Citizen 92.33 /
SemLex 85.79 / local 95.75 vs 94.44 / 87.73 / 97.20 with images; missing clips counted as errors) but the
tuning-pool live replay degrades from 13.3% to 24.8% WER (177 vs 207 correct, deletions 38 vs 15).
Not acceptable; a recognizer trained without hand images is required. New runtime config key
`hand_images` (default true) skips crop encoding and masks hand views; config
`stream_config_mediapipe_v2_no_hand_images.json`. Phone cost of the landmark branch alone at batch 8:
108 ms on 4 big cores (GPU 404 ms) vs 150 ms for the full recognizer. Held-out not used.

## 2026-10-04 — Measured on the 4 GB target phone (Huawei nova Y70): full chain ~8x over budget

Device MGA-LX9: Android 10, Kirin 710 (4xA73 2.0 GHz + 4xA53), Mali-G51 (OpenCL), 3.6 GB. LiteRT
benchmark_model, pinned cores: hand landmarks GPU 19.6 ms (4 big 36.0); palm detector GPU 26.2; pose/face
every 8th frame ~8 ms/frame amortised; boundary 9.2 ms (4 big); span recognizer B8 150.6 ms (int8 113.8);
MobileCLIP2-S0 192.9 ms per crop (GPU delegate only 198/498 ops, 540 ms); T5 encoder 12.9 ms, decoder
42.6 ms/token (~0.9 s per 20-token sentence; int8 ~0.5 s). Our transformer exports run poorly on the Mali
delegate; MediaPipe's models are fully GPU-delegated. Sustained 3 min (hand GPU + span CPU): hand stable
~23 ms, span +20%, battery 35->38 C. Per frame: full chain ~390 ms (~2.5 fps) vs 50 ms budget, ~310 ms of it
the hand-crop encoder; without crops ~80 ms (~12 fps). Decision needed: landmark-only Android recognizer
(retrain without hand crops) vs other options. Report: `artifacts/reports/android_device_bench_20261004/REPORT.md`.
Tools: `artifacts/generated/android_tools/` (adb, benchmark_model), `scripts/bench_android_tflite_v17.py`.

## 2026-10-04 — Android chain converted to LiteRT/TFLite: lossless; speed bound by the hand encoder

`scripts/export_tflite_mediapipe_v17.py` (isolated env `artifacts/generated/litert_env`, litert-torch,
torch 2.13) exported FP32 + dynamic-int8 candidates to `artifacts/tflite/mediapipe_v17_20261004/`.
FP32 parity vs PyTorch: boundary 0/5,161 frame changes; span recognizer (B8) 0/4,232 top-1 changes;
MobileCLIP2-S0 cosine >= 0.9999998 on 600 crops; Stage 3 T5 200/200 identical greedy outputs. int8:
13 frame changes, 6/4,232, broken encoder (min cosine .007), 192/200 — not adopted. End-to-end TFLite
replay (new `--backend tflite` in segmental runtime/replay) is word-identical to PyTorch: tune 89/89
(13.27%), held-out 72/72 (8.60%). Speed, M4 CPU LiteRT 2.2 (4t): boundary 1.2 ms, span batch 16.3 ms,
MobileCLIP 39.7 ms/crop (78 ms 1t), T5 112 ms/sentence; per frame ~85-90 ms vs 50 ms budget, ~64 ms
of it the hand encoder (1.6 crops/frame); landmark-only would be ~20-25 ms. TF 2.16's interpreter is
5-8x slower (~300 ms/frame) and is not the Android runtime. Size: FP32 ~251 MB incl. MediaPipe tasks;
RAM ~378 MB for the TFLite models on CPU. Conclusion: on a 4 GB phone the hand encoder needs the GPU
delegate or the landmark-only fallback (requires a crop-free recognizer). No phone measured.
Environment note: `venv/bin/pip` still targets the deprecated laptop venv; use `venv/bin/python -m pip`.
A stray install there and a broken ai-edge-litert py3.9 wheel in venv were both removed again.
Report: `artifacts/tflite/mediapipe_v17_20261004/REPORT.md`.

## 2026-10-04 — MediaPipe family retrained: all stage gates pass; held-out 8.60% WER

Exact rebuild of the words-only live chain on MediaPipe inputs (current `active/v17` code, Apple
recipes, MPS). Every stage passes the user's 5-point gate (MediaPipe vs Apple, validation top-1, missing
clips counted as errors): landmark base Citizen 91.80 vs 95.77; landmark branch 93.65/95.99 vs 95.50/
96.34 (Citizen/local); hand branch 79.63/58.08 vs 80.69/58.15; unified 94.44/87.73/97.62 vs 96.30/89.06/
97.10 (Citizen/SemLex/local); reel_v2 94.44/87.73/97.51 vs 96.03/89.16/97.03 with phrase 77.61 vs 66.41
and activity 79.05 vs 69.79; span recognizer 94.44/87.73/97.20 vs 95.24/89.67/96.31 (both epoch 4);
boundary student KL .2687 vs .2698 on Apple's exact 43 val videos. End to end (words only, torch,
stream_config_v2 values, MediaPipe early_unsafe): tune 13.27% vs 10.18% WER; held-out 72/186 8.60% vs
11.83% (gate <=16.83: pass) — local60 1.85% vs 8.64%, but unseen ASLLRP12 54.17% vs 33.33% (11/24 vs
17/24), consistent with the continuous hand-detection gap. Held-out replayed once after freezing.
Deviations: MPS not T4; span tuning alpha 1.0 (no proposal twin); Apple four-stream teacher scores reused
by clip ID; Apple-number floors re-derived by rule (354, 346, 93.44/88.09/97.06); boundary and span
training sets matched to Apple's at-the-time sets (538 keys; 6,254 non-prefix spans).
Report: `artifacts/reports/mediapipe_rebuild_v17_20261004/REPORT.md` (gates.json, config, replays).
Code: additive `--extractor mediapipe_full` options in train_stage_1, hand, unified, phrase-adapt,
unfrozen isolated_raw, span, boundary (`--av-raw-dir`, `--keys-file`), replay (`--detector`),
prefix_confusion (`--raw-cache`); Apple defaults unchanged; 98 focused tests pass. Next: TFLite parity.

## 2026-10-04 — MediaPipe extraction stage complete; continuous hand detection gap measured

All inputs for the exact words-only rebuild now exist under `data/local/mediapipe_full_v17_20261003/`:
20,376 isolated landmark+crop+embedding archives (121 no-hands markers); 543/543 Stage-2 phrase
window archives + embeddings (0 window-count and 0 target mismatches vs Apple); 1,675/1,675 20 Hz
continuous captures (`continuous/av_raw`, frame counts identical to Apple for every video) and 665
span-input memos for local_train/tune/test/asllrp_val (97,928 DGS-teacher span keys). 0 failures.

Defect fixed: the first phrase-embedding run grew to a 31 GB footprint (swap 26.6/27.6 GB, laggy
machine) because it lacked the Apple encoder's MPS cap and per-archive `empty_cache`; it was stopped
at 405/543 (atomic writes, no partials) and resumed with batch 32 and a 0.25 MPS cap (~4 GB).
Both embedding commands now bound MPS memory.

Measured input gap (binding context for gates): on continuous 20 Hz video MediaPipe detects a hand in
fewer hand-slot frames than Apple — ASLLRP val .798 vs .931, local train .486 vs .574, YouTube .746 vs
.819 — while detected hands carry all 21 joints (not an edge-joint artifact). Not recoverable by
threshold (12 ASLLRP videos: .81/.83/.82 at .5/.3/.2 vs Apple .96) or by stateless IMAGE-mode hands
(ASLLRP .82 vs VIDEO .80; local .46 vs .48). Isolated archives stay near parity after the v17 3-frame
gap interpolation. Consistent with the 2026-08-09 bakeoff (pre-trim hand detection 38.5% vs 42.7%).
Phrase hand-crop validity .598 vs .662; landmark-valid windows .951 vs .967. Candidate mitigation only
if a gate fails: a second hand pass on a body-centred crop (smaller hands, ASLLRP 1280x720). No
training, no test access. Next: Phase 4 retraining (needs MediaPipe-aware loaders/fingerprints).

## 2026-10-04 — MediaPipe isolated extraction complete and audited

`data/local/mediapipe_full_v17_20261003/` (fingerprint d17b7cd2ecc5614f, pose lite, GPU): 20,497 isolated
jobs = 20,376 landmark+crop archives and 121 no-hands markers (`*.outcome.json`); 0 failures, 0 GPU
aborts, no partial files; 2.39 clips/s overall with 2 shards. Per source (Apple / MediaPipe means):

| source | clips | no hands | hand | two-hand | face | body | jitter |
|---|---:|---:|---|---|---|---|---|
| citizen_train | 1476 | 2 | .569/.571 | .310/.312 | .788/.778 | .533/.604 | .089/.067 |
| citizen_val | 378 | 0 | .569/.579 | .304/.310 | .795/.802 | .393/.567 | .107/.062 |
| semlex_train | 1388 | 2 | .583/.578 | .322/.309 | .785/.752 | .424/.513 | .100/.061 |
| semlex_val | 978 | 4 | .581/.578 | .323/.313 | .770/.747 | .416/.503 | .103/.059 |
| local_train | 13381 | 97 | .655/.656 | .375/.384 | .774/.782 | .524/.635 | .069/.054 |
| local_val | 2896 | 16 | .660/.659 | .383/.390 | .766/.773 | .518/.628 | .068/.054 |

Binding for evaluation: MediaPipe validation must be scored on the full Apple validation lists with
no-hands clips counted as errors (20 validation clips), never on the smaller extracted subset.
Input-quality comparison only; no accuracy claim. Phrase windows, continuous captures and
embeddings run next (sequential, background).

## 2026-10-03 — Android MediaPipe family: full extractor built, pilot passed, extraction running

Plan locked (see high.md). New code, nothing existing modified: `active/v17/mediapipe_full_v17.py`
(MediaPipe-only detector with the AppleVisionDetector interface, schema `slt_mediapipe_full_landmarks_v17`,
fingerprint `d17b7cd2ecc5614f`), `scripts/extract_mediapipe_full_v17.py` (isolated landmarks + crops,
embeddings), `scripts/extract_mediapipe_phrases_v17.py` (543 Stage-2 phrase windows),
`scripts/capture_mediapipe_segmental_v17.py` (20 Hz av_raw + span inputs for 1,675 videos, 97,928
DGS-teacher span keys), `test/test_v17_mediapipe_full.py` (4 pass; existing MediaPipe tests 3 pass).
Pinned task models in `artifacts/model_assets/mediapipe/` (hand reused; pose lite/full, face added).

Calibration on Citizen train frames: MediaPipe handedness = Apple chirality on unmirrored frames
(272/277), so no flip; pose 11-14 = Apple shoulders/elbows (10-25 px vs 230-370 px swapped); face
mesh map (468,473,63,46,293,276,344,40,291,0,17,264,152,34,168), 1-6 px median. Hand VIDEO-mode
tracking is timestamp-independent (626/626 identical across 33/50/200 ms steps; IMAGE mode differs on
41%); pose smoothing is not, so pose/face run in IMAGE mode at the v17 8-frame cadence. No world depth.
Orientation reuses each Apple archive's recorded decision (all container-auto; no coarse rotations).
Crop boxes replay the dense landmark-pass detections of the same frames (Apple Vision is stateless so
its two passes agreed); 595/596 test clips replayed all 16 crop frames.

Pilot, 378 Citizen validation clips, 0 failures (input quality, not accuracy), Apple vs MediaPipe
lite: hand presence .569/.579, two-hand .304/.310, face .795/.802, body .393/.567, shoulder
normalization 81%/100%, hand jitter .107/.062. Pose lite chosen by the predeclared rule (full needed
+5 pp body presence; it had .547 < lite .567).

Runtime defects found and handled: MediaPipe GPU needs SRGBA input; multiprocessing pools deadlock
the GPU (independent shard processes instead); MediaPipe 0.10.14's macOS GPU path leaks a pixel
buffer per hand call and aborts (`kCVReturnAllocationFailed`) after ~6,000 calls, so shards recycle
every ~4,000 calls; per-clip tracking reset makes recycled output identical. Throughput ~1.2-2.5
clips/s with 2 shards (GPU-bound; VS Code helpers and the SSD's FSKit service compete for CPU).
Full isolated extraction (20,497 clips) launched to `data/local/mediapipe_full_v17_20261003/`; log
`artifacts/generated/mediapipe_full_isolated_20261003.log`. Pilot roots
`data/local/mediapipe_full_v17_20261003_pilot_{lite,full}` retained. No training, no test access.
Next: continuous captures and phrase windows after the isolated run, then embeddings, then audit.

## 2026-09-30 — LIVE takes saved as local MP4s (Start → Finish)

User request: save iPhone LIVE recordings locally, no extra compression pass. Changed
`/Volumes/secret/SLT/mobile_app/slt_mobile_app/ios/Runner/LiveReel/LiveReelApp.swift`
(`LiveVideoRecorder`, `LiveReelSessions.stamp`) and `LiveReelViewController.swift`.
Pre-edit copies: `artifacts/reports/live_video_recording_v17_20260930/app_backup_before/`
(mobile_app is not a git repo). Behaviour, as the user chose: Start begins a take; ↺ Reset
discards it and starts a fresh one; the big Finish button now ends the run (the former
Stop path: flush, render/speak the sentence) and saves the take; Back saves a take still
recording. Finish is enabled whenever running (was: only once something was recognized).
The two-hand finish gesture does not split takes. PRACTICE is not recorded. Frames are the
unmirrored, upright 1280x720 capture buffers the model sees, H.264 ~10 Mbps via
AVAssetWriter, written in tmp and moved to `Documents/live_reel_videos/<visit stamp>_<NN>.mp4`
only after `finishWriting` completes (no moov-less files). Session JSON gets a
`video_start` event naming the file. Late frames dropped by the capture output are also
absent from the video. Verified: unsigned generic-iOS Debug build succeeded with no warnings
in the changed files; Mac run of the recorder on 90 synthetic frames gave h264 1280x720,
90/90 frames, 3.0 s, moov present; discarded/empty takes left no file.
Phone check (same day): signed Release installed/launched on angelo 13:44. Session
20260930_134459: takes _01–_03 discarded by ↺ as designed; `_04.mp4` saved (15.3 MB, h264
1280x720, 379 frames, 12.7 s, ~30 fps, 10.1 Mbps), upright and unmirrored, and holds the
exact HELLO GOOD MORNING HOW YOU attempt.

---

## 2026-09-29 — user-selected logo applied to Figma branding and screens

User rejected the generated concepts, then selected Desktop/ATLAS artwork.
Initially applied `Vector Smart Object.jpg`, then superseded it with the user's
`/Users/frnzlo/Desktop/ATLAS/logo transparent.png` (668x686 RGBA, alpha 0–255).
Uploaded the original PNG without editing; image hash
`864b803b6b8e59be4b7e074fccaed4386044c7e5` fills shared component `4:5` with FIT.
Old vector child `4:6` is hidden. Brand lockup/app-icon and Splash/Home instances
inherit the selected artwork; rejected concept boards are labeled not selected.
Updated notes to identify raster artwork accurately; restored pale Splash background
with transparency. The app-icon tile remains white. JPEG brand/screens were visually
checked before the PNG replacement; PNG transparency and Splash/Home hashes verified.
Figma file: https://www.figma.com/design/Fg692rO8j8IjzW0FPQ2HdU?node-id=7-14
No app code, models, or source artwork changed. Next: review the selected logo in
Figma; wordmark font remains provisional. Runtime ground truth unchanged.

## 2026-09-29 — ATLAS logo revision studies created in Figma

User rejected the earlier swoosh/person symbol and supplied a two-hand, minimal,
geometric brief. Initial Figma retry hit the Starter limit; after the user upgraded,
access succeeded. Created three custom two-contour SVG studies (Exchange, Lift,
Counterform) directly on the existing Brand page, with full-color, monochrome,
reversed, and 16/24/32/48px comparisons. Study board: node `12:14`.
Recommended Counterform for its compact opposing-hand geometry; added an editable
two-path master, provisional Inter wordmark lockup, and three app-icon applications
on node `13:14` (master `13:17`, paths `13:18` and `13:19`).
https://www.figma.com/design/Fg692rO8j8IjzW0FPQ2HdU?node-id=13-14

Both boards were screenshot-inspected. A screenshot request initially used an
incorrect node ID and was corrected to the returned `13:14`; final render passed
visual inspection. These are abstract gestures, not a claimed ASL sign; recognition
by sign-language users is untested. Existing screen branding has not been replaced
while the new directions are reviewed. App/runtime state and ground truth unchanged.
Local construction assets: `artifacts/design/atlas_logo_v2_20260929/`; temporary
screenshots and construction script: `/tmp/atlas-figma/`. Repository changes are
these design assets and this log only. Next safe action: user reviews the recommended
mark and alternatives in Figma; refine the chosen direction before propagating it.

## 2026-09-29 — ATLAS Figma rebranding concept created (design only)

User approved all ten reference screens plus branding in a new Figma file, custom
editable vector logo, Inter interface typography, provisional ATLAS wordmark, and
existing app imagery mixed with camera placeholders. File:
https://www.figma.com/design/Fg692rO8j8IjzW0FPQ2HdU

Reviewed the current Flutter shell under
`/Volumes/secret/SLT/mobile_app/slt_mobile_app/lib/shell/` and native
`ios/Runner/LiveReel/LiveReelViewController.swift`. Retained four Home actions,
5/10/20-sign practice sets, 100-sign vocabulary, landscape camera views, and Finish.
The IDE pasted-text attachment path was unavailable; the visible rebranding board
and current app source supplied the design context.

Created Brand, Mobile Screens, and Components pages; seven portrait frames
(390x844), three landscape frames (844x390), a custom SVG-derived vector symbol,
app icon, 15 reusable component masters, 41 variables in two collections, eight
Inter text styles, and a shadow style. Uploaded ten existing gloss JPGs from app
assets. Session data, confidence values, and camera illustrations are mock content.
Brand and screen screenshots were inspected; fixed component shrinkage, logo
instance scaling, and clipped gloss labels. Portrait and camera creation calls
confirmed Inter as their only text font family. No app code, models, or data changed.

Figma Starter MCP call limit rejected the final navigation/audit call before it
executed; no clickable prototype connections were created. The ten requested
editable designs exist. Local state and review screenshots: `/tmp/atlas-figma/`
(temporary, not canonical). Only this repository log changed for this task;
PROJECT_GROUND_TRUTH.md stays unchanged because runtime state is unchanged.
Next safe action: user reviews the Figma designs and replaces the provisional
wordmark font; prototype navigation can be added after Figma access resets.

## 2026-09-07 07:16 PST — all v8 frames checked; remaining source orientation noise isolated

V8render84702 completed. Inspected all44YOU/43NEED frames; false face-directed lifts
are removed. Saved frame_review.json with exact wrist/path and relative hand changes.
Source footage sheets confirm actual single approach and lowering after YOU, and
repeated NEED flex. V8still has YOU orientation rolls copied from detector estimates;
it is not the final clean motion candidate. To isolate those, v9_clean_motion uses
existing calibrated SignWriting movement (same S100/S106 constraints) rather than
noisy source trajectories. Added append_isolated_release by reversing the existing
approach helper; extended regression verifies unchanged core and monotonic lowering
back to initial neutral. All24rig/mesh tests pass. Renderer now exports one approach,
core, one0.3s release; neither boundary is inserted between continuous signs.
V9render session64585 is running. Review all frames and verify intended core movement
before handoff. No new classification training or data admission.

## 2026-09-06 21:21 PST — symbol orientation/motion pilot implemented; SCHOOL proposal ranks second

Actual proposal-runtime SCHOOL clip finished: raw and corrected TOMORROW GO remain.
TOMORROW SCHOOL GO now appears second with jointscore-7.7249 versus-6.0276 winner.
This verifies proposal inclusion but contradicts successful recovery. Do not increase
contextweight solely to fix this known clip; broaderdev measured harms already exist.

Fetched/inspected official handphotos10000/10020/10030/10040/10050/10620 and
orientation_reference_sheet.png. Primary references verified:
https://www.signwriting.org/software/signwriterstudio/help/handchooser.htm
https://www.signwriting.org/video/swvideo2.html
https://www.signbank.org/iswa/265_sg.html (S265 small single forward floor-plane motion)
https://www.signbank.org/iswa/22e/22e_bs.html (S22e single wristflex wall-plane).
Two guessed pageURLs failed; actual references above supply the interpretation.

Added animate_signwriting_pilot to existing avatar module: only exact right-handed
S10040+S26500 and S10620+S22e04 accepted. Rigidly orient a static reference hand;
generate minimum-jerk forward6cm stroke or65degree wristflex, neutral wristposition
[-.18,1.10,.28]. Static reference thumb/proportions and source duration remain inputs;
per-frame wrist/orientation trajectory is now symbolic. Numerical calibration values
are explicit hypotheses, not encodedSignWriting or nativeapproval. Test RED→GREEN
confirms forwardaxis/travel, stationarywristflex and unsupportedpair refusal.
Existing pilot script has --symbol-motion, recomputes armIK, pinsrighash, records
limitations.27 focusedavatar/mesh/context tests and diffcheck pass. Rendering v4motion
in session24067, log signwriting_avatar_pilot_v17_v1/render_v4.log. Full100sign
symbols, freeEnglishASLplanning, nonmanuals and connectedsyntheticmotion still unmet.

## 2026-08-31 20:19 PST — physical-phone errors localized to Stage 2 and its WHERE adapter

The 22 successful post-fix captures in the connected iPhone 13's Files-visible
`Documents/Diagnostics` directory were copied read-only to a temporary host directory
and audited. All seven saved arrays in every capture have the expected finite values,
window/source masks agree with the reports, Apple Vision observed usable hands, and
the orientation path uses AVFoundation's preferred track transform. Replaying the
exact saved landmark, hand-embedding, validity, box, and window-mask tensors through
the pinned PyTorch graphs reproduces all 22 phone/Core ML gloss sequences exactly.
This rules out an iOS label-order error, corrupt NumPy export, stochastic Core ML
behavior, and a global class-index shift.

The deployed checkpoint is a context-adapted Stage-2 head, not the bare temporal CTC
head. Its development-selected residual has weight `1.5` and is allowed to change
only zero-based classes 9 and 86, `WHERE` and `HOME`. On capture `115101`, the bare
head decodes the exact phone tensors as `NEED NEED`; adding the deployed residual
changes the same tensors to `WHERE WHERE`. The residual changes five of the 22 phone
decodes and explains a material part of the observed WHERE collapse. It was selected
on ASLLRP development validation where NEED had no coverage, so its behavior on this
new phone signer is validation overfit rather than independent generalization.

A bounded validation-only A/B was then run on every existing Citizen, SemLex, and
local isolated validation clip for `HELLO`, `MORNING`, `NEED`, and `WHERE`—165 clips
total, with no test access. The selected isolated Stage-1 model scores 151/165; the
bare Stage-2 CTC head scores 120/165; the deployed context-adapted Stage 2 scores
117/165. Per class, the three results are respectively: `HELLO` 29/32, 19/32, 19/32;
`MORNING` 44/45, 39/45, 39/45; `NEED` 37/40, 17/40, 11/40; and `WHERE` 41/48, 45/48,
48/48. In local validation alone, the adapter changes NEED from 8/27 exact to 4/27
and emits WHERE on 19/27 NEED clips. The underlying isolated data/model is therefore
not globally trashed; the main degradation is introduced by the Stage-2 training and
deployment contract.

The Stage-2 coverage audit explains that degradation. Real Stage-2 phrase training
contains HELLO only inside 107 `HELLO HOW YOU` rows and has no real NEED target at all.
Its real validation contains HELLO only inside 27 `HELLO HOW YOU` rows and again no
NEED. By contrast, contextual ASLLRP contributes 50 real WHERE training clips and 16
WHERE validation clips, after which the explicit WHERE residual further boosts that
class. The isolated Stage-1 replay itself is reasonably populated: Citizen/SemLex/local
train counts for HELLO are 14/12/105, MORNING 14/17/149, NEED 15/14/177, and WHERE
14/18/153. This is a Stage-2 supervision/selection imbalance, not evidence that those
four isolated class folders were mislabeled.

There is also a phone temporal-boundary mismatch. `prepareStage2` anchors nonoverlapping
32-frame windows at recording frame zero and does not trim or re-anchor before
windowing. Fourteen of the 22 phone attempts contain 19--30 frames before Apple Vision
first observes a hand. Consequently, a short isolated sign is often split between the
end of a mostly idle first window and a resampled partial second window; this explains
the sensitivity to when Record was tapped and the duplicated `HELLO HELLO`/`NEED NEED`
outputs. Stored portrait/landscape metadata is not the primary cause.

No production change was made during this diagnosis. The recommended next design is
to remove the development-only HOME/WHERE residual from deployment, motion-anchor the
completed clip before forming windows, and run the already strong whole-clip Stage-1
classifier for a single detected sign while retaining Stage 2 for multi-sign clips.
Stage 2 should then be retrained/evaluated with class-balanced one-token replay, random
leading/trailing idle and window-offset augmentation, and an all-100-class isolated
validation gate in addition to genuine phrase gates. The 22 phone captures should be
given intended gloss labels and retained as new-signer diagnostic evidence, with a
subset locked before any phone-data adaptation. Incremental extraction during
recording remains explicitly deferred. No Citizen, SemLex, local, ASLLRP, or other
test split was accessed.

## 2026-08-31 19:54 PST — physical iPhone diagnostics fixed and Release app reinstalled

Two user-operated iPhone 13 runs established the first real-camera evidence for the
Flutter app. A short intended `HELLO` clip was recorded upright at 720x1280 but decoded
as `MORNING`; its report showed 2,330.3 ms Apple Vision extraction, 1,341.3 ms RGB crop
encoding, 26.5 ms median Core ML inference, and nominal thermal state. A later phrase
run emitted `HELLO HELLO HOW YOU`; recognition retained the intended phrase with one
duplicate token, but Stage 3 fell back to `Hello hello how you.` because exact reviewed
templates never delete recognized tokens. That enabled benchmark run showed 6,119.8 ms
extraction, 28.3 ms median, 32.1 ms p90, resident memory 73.6->441.6 MiB, and nominal
thermal state. These timings and memory values describe the pre-fix physical build;
they are not post-optimization measurements.

The live app at
`/Users/frnzlo/Documents/machine_learning/mobile_app/slt_mobile_app` now fixes the
camera-preview distortion at its root: Flutter no longer wraps `CameraPreview` in the
raw sensor aspect ratio a second time, and an orientation-aware cover layout preserves
geometry without stretching. Portrait controls were moved ahead of the scrollable
result so they are no longer clipped, and the benchmark line now renders a real newline.

Every attempt is copied out of iOS temporary storage before extraction to a unique
Files-visible `Documents/Diagnostics/<UTC timestamp>_<UUID>/` directory. Successful
captures retain `recording.mp4`, `report.json`, `tensor_manifest.json`, raw upright
landmarks, exact normalized model landmarks, source/window masks, hand-valid masks,
normalized hand boxes, and exact MobileCLIP2 hand embeddings as float32 NumPy files.
Failed attempts retain the video plus `error.json`. `UIFileSharingEnabled` and
`LSSupportsOpeningDocumentsInPlace` are enabled, so the folders appear under
Files -> On My iPhone -> ASL Translator -> Diagnostics. Reports are saved for every
capture rather than only benchmarks, and the old literal Swift `${...}` filename bug
can no longer overwrite prior evidence. RGB crop images are deliberately not duplicated;
the original video and saved boxes reproduce them.

The completed-file pipeline still forms sequential 32-source-frame windows, with each
window normalized/resampled to the model's fixed 32x61x5 input. Safe latency/memory
changes were applied without changing those model inputs: model/label/naturalizer
resources are cached once, camera files use AVFoundation's preferred transform rather
than a four-rotation Vision sweep, the hand observations from landmark extraction are
reused for RGB crop creation, normal one-shot inference reuses the cold output instead
of running the neural path twice, and per-frame/per-crop autorelease pools bound Apple
framework temporaries. Core ML remains configured with `computeUnits=.all`, leaving
CPU/GPU/Neural Engine scheduling to iOS. Independent parallel Vision windows were not
introduced because the pre-fix run already reached 441.6 MiB and concurrent window
buffers would increase peak memory. Live extraction during recording remains a separate
camera-stream architecture and is deferred until the post-fix completed-file benchmark
is measured.

Stage 3 gained one explicit reviewed rendering for the observed recognizer sequence
`HELLO HELLO HOW YOU` -> `Hello, how are you?`; the gloss output remains unchanged and
visible. The ordinary `HELLO HOW YOU` template already existed. The canonical and app
manifest copies match at SHA-256
`1d855ad74b2c26d68a28dd6fc55630bb00e2127ec6ab57fadc74e685b46b7716`.
This remains bounded reviewed-template naturalization, not an on-device LLM or general
ASL-to-English translator.

Validation passed: Flutter analysis has zero issues, both Flutter tests pass, all eight
focused Stage-3 naturalizer tests pass, and the unsigned arm64 Release iOS build succeeds
at 126.3 MB. The signed Release app version 1.0.0 was then installed and launched on the
connected iPhone 13 `angelo` under bundle ID `com.kokoab.sltMobileApp`. Post-fix
latency, memory stability, prediction behavior, and Files bundle contents still require
the next user-operated recording. PopSign was explicitly removed from the active plan.
No Citizen, SemLex, local, ASLLRP, PopSign, or 2M-Flores test split was accessed.

## 2026-08-24 13:42 PST — exact compact Stage 2 exported and validated in Core ML

The retained compact Stage-2 graph is exported as two FP32 packages. The frozen v17
multimodal window encoder is
`artifacts/coreml/Stage2FrozenEncoderV17FP32.mlpackage`, tree SHA-256
`1146b539800e6f09a743f4a8ee882c9b2cd2b01503ff3f362a11e97e8c827bb9`; the exact
context adapter plus CTC head is
`artifacts/coreml/Stage2CompactContextV17FP32.mlpackage`, tree SHA-256
`e92ba7d8b7c61c52bc776840e953c73abb6b012637991d01582d4fd64067760a`.

Combined cold validation covered 363 samples and 574 windows with zero Core ML versus
PyTorch decode mismatches. It exactly retained 11/24 ASLLRP contiguous, 7/259 local,
and 43/254 ASLLRP contextual edits. The 12.42 ms median and 12.94 ms p90 are Mac-host
Core ML timings only, not iPhone/ANE/thermal evidence. The packages consume
precomputed MobileCLIP2 hand embeddings; the crop-to-embedding MobileCLIP2 network is
not yet implemented in the iOS app. Therefore the Stage-2 Core ML graph is ready, but
camera-to-gloss mobile deployment is not. Full evidence is in
`artifacts/reports/stage2_v17_coreml_export/README.md`. No test split was accessed.

## 2026-08-13 21:27 PST — iPhone 13 simulator Core ML orientation suite passes

The automated simulator harness is complete and the dedicated virtual device is
strictly an iPhone 13: `SLT Orientation Benchmark iPhone 13`, CoreSimulator device
type `com.apple.CoreSimulator.SimDeviceType.iPhone-13`, model identifier `iPhone14,5`,
and UDID `ABE172E1-A940-4937-92D9-1C666E674060`. No iPhone 17 device was created.
Apple no longer offers the exact iOS 26.2 simulator runtime through Xcode's download
catalog, so the harness compiled against the installed iOS 26.2 SDK and used the
nearest compatible runtime, iOS 26.3.1 (`23D8133`). The runtime and virtual device
remain installed for repeatable runs.

The first literal end-to-end simulator attempt correctly failed closed at all eight
angles. Runtime diagnostics show that the iOS 26.3.1 simulator image contains the
Apple `cnn_human_pose.espresso.net` and `.shape` files but omits their matching
`.weights` file; Vision reports `Unable to setup request in
VNDetectHumanBodyPoseRequest`. The Mac framework has matching graph hashes and the
weights, confirming this is a simulator-runtime asset boundary. The physical-device
app's normal Apple Vision extraction path is unchanged. The final simulator harness
therefore performs the unchanged v17 Apple Vision extraction on the macOS host,
serializes and SHA-256-pins each `(32,61,5)` tensor, and runs only Core ML model loading
and inference inside the iPhone 13 simulator. Every report explicitly records
`extractionExecutionEnvironment: host_macos_apple_vision`, `endToEndPipeline: false`,
`hardwarePerformanceClaim: false`, `thermalsInterpretable: false`, and the missing-
weights limitation. This is not end-to-end iOS Vision evidence and makes no physical
iPhone, ANE, memory, thermal, or sustained-latency claim.

Suite `orientation-v17-ios26-3-1-20260813T132843Z` passed. It uses only Citizen's
official validation clip `020030442376253177-HELLO.mp4`, source SHA-256
`d5d3ac36b623c46b0b22a42dbaa36e5e36321bc1b5987ef719dcd96d9d63473b`,
and expanded-canvas 0/17/37/73/90/123/180/270-degree inputs without crop or
anisotropic stretch. All 8/8 host Apple Vision extractions succeeded, all 8/8 iPhone
13 simulator predictions were `HELLO`, and each report contains exactly 200 timed
inferences. Corrections were `0/0/0/270/270/270/180/90`; residual rolls were
`0/17/37/-17/0/33/0/0`, all within 45 degrees. Mean per-angle median simulator
inference was 5.5683 ms and maximum p90 was 6.0731 ms; these are Mac simulator timings
only. The selected checkpoint remains
`a7490409b3dfd76ba1ff432d2392b5e27df33f12e1b088cd36609fe03c082366`
and the Core ML tree remains
`1cfd5e97cb8ebb29b424b1391ceb85ed9d62e5b7e25841b86254d414ccd0fb5e`.

The final result is
`artifacts/reports/orientation_v17_simulator_benchmark/orientation-v17-ios26-3-1-20260813T132843Z/result.json`
with SHA-256
`929881c5658f3ad1e26c4db1eed83fa7f5311ef7e97b36128077b89b3ca4918d`;
its aggregate SHA-256 is
`a74212ff84a97d5a1377bfcde996641b3651076fade25b0d2a8708dac10116ce`.
The host runner, app automation/reporting code, and focused test SHA-256 hashes are
`ced997fb753ad9dc86f5458fbccb2e8e4b32c150e8f99cab04216fbdb43235e4`,
`73f5eafc272355e2d6cc41c531cff8ba519894e5104b23dd22653d4074977e36`,
and `2efc6be3e0ac6b55b24f06e476ccf4237dc238f46bdd94a87e68cbb366a3d7d1`.

Final validation passes: unsigned Release builds for both iPhone Simulator and generic
iPhoneOS; all 84 focused Stage-1, extractor, independent-capture, and simulator tests;
11 simulator evidence JSON files parsed; changed Python compilation; the 1,100-row
capture-pack setup audit with zero errors (audit SHA-256
`de885a97a09698abf42cb666522dd20c8ab826f7c6c2e964916640a59faf7732`);
and `git diff --check`. Citizen and SemLex test splits were not accessed. The app is
compiled and ready for the deferred signed install and full end-to-end run on a real
iPhone 13.

## 2026-08-13 18:51 PST — arbitrary-orientation gate passes; ready for real phone

The final automatic coarse-orientation rule supersedes the 18:20 face-band rule. It
first chooses the horizontal versus vertical anatomical-axis family from the maximum
shoulder/eye-line horizontalness, using body confidence only as a tie-break, and then
uses signed mouth-below-eyes geometry to distinguish the two opposite directions in
that family. This prevents a correctly upright clip from being rotated merely because
its shoulders temporarily disappear during signing. Python and Swift implement the
same rule. Container metadata is still applied first; the probe only chooses a
lossless quadrant, and the trained classifier covers the remaining continuous roll.
No path crops or anisotropically stretches ordinary input video.

All 175 official ASLLVD clips were re-extracted from the exact top/front-camera pixels
and inclusive annotated frame interval with the final rule. All 175 were retained as
already upright, all 175 pass the v17 integrity audit, and zero clips failed. The final
manifest SHA-256 is
`e2d6d18cb4e43e1809b35e97f980f561a04f0affacd5877ab2d0e5e666009faf`;
the audit SHA-256 is
`8809631035963351a0c6f50a349764900ae850625424a4f177f7998d4962715e`.
Before training on this source, the frozen orientation fallback independently scores
110/175 (62.86%) top-1, 152/175 (86.86%) top-5, and 60.58% macro class top-1 over
52 exact variants and six external consultants. The evidence report is
`artifacts/reports/asllvd_asllex_v17_external_baseline.json` with SHA-256
`c80ad4e1b01fcf76e09b77b073af30a74289df5136657ca280fb0f41612a041b`.

Private Kaggle feature dataset version 3 contains only those final derived features
and provenance; raw ASLLVD movies were not uploaded. Kernel
`kokoab/slt-v17-stage1-orientation-asllvd-v1` version 3 completed successfully, pins
the final manifest, and confirms false Citizen/SemLex test-access flags. Its
checkpoint SHA-256 is
`661be8d6db71df8e07c161d57cc12464566f20317620ba6005d5c54fa552b412`,
but it scores only 355/378 (93.92%) on Citizen validation. This is below the
predeclared 359/378 clean-domain floor, so the challenger is rejected immediately;
SemLex and raw-pixel errors are not used to rescue it. Kernel versions 1 and 2 remain
explicitly superseded, and all three relevant orientation kernels now report
COMPLETE. The selected phone candidate remains the continuous-roll augmentation-only
fallback checkpoint
`a7490409b3dfd76ba1ff432d2392b5e27df33f12e1b088cd36609fe03c082366`.
The four-stream RGB/landmark research teacher remains unchanged at 370/378 Citizen
and 882/978 SemLex.

The frozen candidate manifest now pins final extractor SHA-256
`d049ae34d732fa504ad8702a91d3409dcf1debd8415bc5dea588d7f243138f47`;
its own SHA-256 is
`cd0f2be27cabd1b9d9eedc15d4b74cfcd01c09f552e2efba491678ca55323ee9`.
The untouched 1,100-row independent capture pack passes its setup audit with that
lock and remains inference-free and capture-pending. The final Swift orientation
pipeline and fallback Core ML model compile in unsigned Release mode for both generic
iPhoneOS arm64 and the arm64/x86_64 iPhone Simulator. Both bundles contain the
compiled 13 MiB model, frozen 100-class manifest, and exact model-provenance manifest;
all four iOS interface orientations are declared. Real-device timing, memory,
thermals, and independent capture accuracy are deliberately still unmeasured: the
bundle is the instrument that will collect those measurements on the actual phone.

The definitive raw-pixel sweep with that exact final selector completed over the
fixed 100-clip Citizen validation slice at 0, 17, 37, 73, 90, 123, 180, and 270
degrees. All 800/800 conditions extracted successfully. Correct counts are
93/96/85/94/93/91/93/93, respectively: 92.25% eight-angle mean and 85% minimum.
The 90-, 180-, and 270-degree predictions are exactly identical to upright. Every
upright clip remains at correction 0; every 90/180/270 clip receives exactly the
inverse lossless quadrant; 99/100 clips at 37 degrees remain at 0, while all 73- and
123-degree clips choose the nearer 270-degree correction. The exact report is
`artifacts/reports/stage1_v17_raw_orientation_robustness/augmentation_plus_vision_auto_axis_family/metrics.json`
with SHA-256
`c36f3ec9408158714f12b5942c816ec4a01d8c28437bbe04438adb0bc65e4c77`.

Final validation passes: all 78 focused Stage-1, extractor, and independent-capture
tests; 175/175 ASLLVD schema/integrity archives with zero errors; the 1,100-row frozen
capture-pack setup audit with zero errors; Python compilation for all new/changed
training, extraction, acquisition, finalization, evaluation, and Kaggle-runner code;
unsigned Release builds for generic iPhoneOS and the iPhone Simulator; compiled-model
and manifest bundle checks; and `git diff --check`. Core ML exhaustive parity remains
378 validation samples with zero top-1 mismatches and maximum absolute logit error
0.006403. All relevant Kaggle jobs are COMPLETE. Citizen and SemLex test splits were
not accessed.

## 2026-08-13 16:55 PST — detector-space quadrant fix and official supplement acquisition

The fixed 100-clip Citizen validation raw-pixel stress completed for both orientation
retrained checkpoints. The augmentation-only model scores 93/100 at 0 degrees,
86/100 at 37, 79/100 at 90, and 20/100 at 180. The canonicalized model scores
92/100, 89/100, 73/100, and 17/100, respectively. All 800 attempted extractions
returned a usable sample. At 180 degrees Apple Vision's mean body presence is exactly
zero for both, proving the remaining inversion failure occurs before the classifier.
The augmentation-only checkpoint is the better raw-video candidate. Exact reports are
under `artifacts/reports/stage1_v17_raw_orientation_robustness/augmentation_only/`
and `canonical/`; no Citizen or SemLex test data was accessed.

`active/v17/extract_v17.py` now applies container orientation first and, in automatic
mode, probes three frames at four lossless quadrants using Apple Vision face/body
anatomy. It selects the quadrant whose mouth-to-eye vertical relationship and body
confidence are most upright, then performs the main extraction once. The continuously
roll-augmented classifier therefore sees at most a 45-degree residual roll; this is
not a portrait/landscape classifier and does not stretch aspect ratio. Explicit manual
rotations remain authoritative and can bypass the probe. Sixteen extractor tests pass,
including real Apple Vision recovery of 0-, 90-, and 180-degree source rotations. A
new full 100-clip detector-space validation run with this automatic probe is active.

The same four-quadrant anatomy probe is implemented in the generic iOS benchmark
pipeline after AVFoundation applies the file's preferred transform. The app accepts
any native aspect ratio, records the chosen correction and all orientation scores,
and still measures extraction, Core ML latency, memory, thermal state, and expected
label accuracy. Its simulator build succeeds. The model-byte measurement was also
corrected to measure the compiled model directory rather than the whole app bundle.

The ASLLRP route described at this timestamp was subsequently rejected because the
downloaded distribution archives did not contain the metadata's newer recording IDs.
The 17:20 entry records the replacement official ASLLVD acquisition and is canonical.

## 2026-08-13 16:29 PST — arbitrary-roll failure measured; two controlled retrains active

The existing compact part-wise checkpoint was stress-tested on the official Citizen
**validation** landmarks only at continuous synthetic camera rolls. It scores 366/378
(96.83%) at 0 degrees, 363/378 (96.03%) at 17, 347/378 (91.80%) at 37,
151/378 (39.95%) at 73, 40/378 (10.58%) at 90, 4/378 (1.06%) at 123,
2/378 (0.53%) at 180, and 38/378 (10.05%) at 270. This proves the previously
selected classifier was not orientation-robust even though v17 extraction already
preserves aspect ratio and honors right-angle video orientation metadata. Exact
metrics are under
`artifacts/reports/stage1_v17_orientation_robustness/partwise_original/`.

`active/v17/model_v17.py` now has an optional, parameter-free, missing-safe clip-level
camera-roll canonicalizer. It estimates an anatomical horizontal axis from
confidence-weighted shoulders with an eye-line fallback and rotates every landmark XY
channel in isotropic space. Forced onto the old checkpoint, it gives bit-stable class
predictions at 0, 17, 37, 73, 90, 123, 180, and 270 degrees, but only 295/378
(78.04%) at every angle. This is a diagnostic, not a selected result: the old model
was not trained on canonicalized inputs. Exact metrics are under
`artifacts/reports/stage1_v17_orientation_robustness/partwise_forced_canonical/`.

A real train-only Citizen clip was also synthetically rotated in pixel space before
Apple Vision. Hand detections remained nonzero at every tested angle, but coverage
degraded: observed hand frames were 23/19/20/15/16 at 0/37/90/123/180 degrees,
respectively, while body presence dropped to zero at 90 and 180. Therefore landmark
rotation alone is insufficient evidence for raw-video robustness; pixel-level
orientation stress and missing-landmark behavior must be included before handoff.
The clip was read only and no Citizen or SemLex test data was accessed.

The first Kaggle augmentation kernel failed before training because its initial code
overlay contained the new trainer but an old `model_v17.py` that lacked the active
part-wise configuration fields. The private code dataset was corrected to include the
current model and trainer, with an explicit `test_data_included:false` manifest. The
augmentation-only kernel `kokoab/slt-v17-stage1-orientation-robust-v1` version 2 and
the independent augmentation-plus-canonicalization kernel
`kokoab/slt-v17-stage1-orientation-canonical-v1` version 1 are both RUNNING on T4s.
Both retain the exact train/validation-only Citizen+SemLex protocol, class/source
balancing, architecture, and seed; neither frozen test split is present or accessed.

The Apple Vision extractor now also accepts any finite explicit clockwise correction
angle, not only 0/90/180/270. Exact right angles retain lossless transpose/flip paths;
other angles use a single affine resampling pass on an expanded canvas that contains
all four transformed corners, so pixels are neither cropped nor anisotropically
stretched. The transform and its exact floating-point angle are recorded in output
metadata. Fourteen focused extractor tests pass, including arbitrary 37-degree canvas
expansion, non-finite rejection, exact right-angle/mirror equivalence, isotropic
portrait/landscape geometry, and two real Apple Vision tests. This provides an explicit
path for detector-space stress generation and for a phone sensor-derived roll
correction; automatic container orientation metadata remains the default.

`active/v17/evaluate_raw_orientation_v17.py` now defines the corresponding fixed
detector-space evaluation: it rejects Citizen test by construction, selects only clips
already accepted in a train/validation feature inventory, applies expanded-canvas
pixel rolls, re-runs Apple Vision, and reports extraction coverage, top-1, and upright
prediction agreement for each angle. The candidate freeze was advanced only for the
intentional model/extractor runtime changes; checkpoint members, fusion weights, and
their evidence did not change. Candidate-manifest SHA-256 is now
`035bb08476b098c4a47273120cddeae9a42389b60ebc69f1eceb9b4105406ff4`.
The capture pack now pins that hash without changing its 1,100 immutable ledger rows
or schedules. The combined Stage-1, extractor, and capture workflow suite passes 74/74,
and the scoped diff check passes.

The fixed 100-clip raw-pixel validation stress has now completed for the old compact
checkpoint. Apple Vision returned an extractable sample for all 100 clips at every
tested angle, but model top-1 fell from 96/100 upright to 81/100 at 37 degrees,
4/100 at 90, and 14/100 at 180. Upright-prediction agreement was respectively
100%, 81%, 4%, and 14%. Mean hand presence stayed near 0.52--0.56, while body
presence fell from 0.385 upright to 0.099 at 90 and zero at 180. This confirms both
effects at realistic scale: the classifier lacks roll invariance, and auxiliary
landmarks become selectively missing after raw pixel rotation. Exact results are in
`artifacts/reports/stage1_v17_raw_orientation_robustness/partwise_original/metrics.json`.
The evaluator opened 100 Citizen validation videos already admitted by the frozen
validation feature inventory. It did not access Citizen or SemLex test data.

Both controlled orientation kernels completed successfully and their outputs were
pulled locally. The augmentation-only checkpoint SHA-256 is
`a7490409b3dfd76ba1ff432d2392b5e27df33f12e1b088cd36609fe03c082366`;
it retained epoch 108 and completed 138 epochs. It scores 362/378 (95.77%) on clean
Citizen validation and 839/978 (85.79%) on SemLex validation. In landmark-roll stress
its top-1 is 358/378 at 37 degrees, 355/378 at 90, 350/378 at 123, and 348/378 at
180; the worst nonzero-angle prediction agreement with upright is 94.71%.

The augmentation-plus-canonicalization checkpoint SHA-256 is
`32659e40f9b26b3fd63bc25d3ad5bfb0293bc19f36648ecc2bd71afd0fbba639`;
it also completed 138 epochs. It scores 360/378 (95.24%) on clean Citizen validation
and 848/978 (86.71%) on SemLex validation. Its predictions are exactly invariant in
landmark space at all eight evaluated angles, with 360/378 top-1 and 100% prediction
agreement at each angle. Neither model replaces the prior 366/378 and 853/978 compact
clean-accuracy checkpoint on clean-domain evidence alone; raw-pixel roll stress is
running as the declared orientation selection gate. Training provenance for both
confirms Citizen train/validation plus approved SemLex train only, 50/50 source/class
balancing, seed 1701, and false Citizen/SemLex test-access flags.

## 2026-08-13 15:46 PST — portrait-iPhone collection is executable and variant-gated

The independent portrait-iPhone protocol is now an executable, fail-closed collection
workflow rather than only prose plus an empty ledger header.
`scripts/build_portrait_iphone_eval_v17.py` has three explicit phases: generate the
100-row exact-variant review sheet, build reproducible capture schedules only after
all variants are approved, and audit either the untouched setup or the completed
pre-inference set. A valid pack contains at least five new pseudonymous signers,
two independently randomized 100-class repetitions per signer (1,000 target slots),
and the recommended 20 OOV slots per signer (100 OOV slots). It writes a 1,100-row
attempt-preserving ledger, per-session prompt schedules, and hashes of every source
and schedule. Recaptures append a numbered attempt instead of deleting an objective
failure.

The structural audit pins every class index, canonical label, exact Citizen raw gloss,
and ASL-LEX code; checks exact signer/repetition/class and OOV coverage; rejects changed
input/schedule hashes, unsafe paths, duplicate accepted paths/content hashes,
model-derived QC reasons, unresolved attempts, incomplete device metadata, nonportrait
declarations, prompts not confirmed hidden, and target rows whose performed gloss does
not exactly confirm the pinned variant. The pre-inference phase can report ready only
when exactly one objectively accepted attempt exists for every planned slot. It never
runs a model and records both model/test access as false.

The real review sheet is frozen at
`active/v17/portrait_iphone_variant_review_v17.csv` with exactly 100 pending rows and
SHA-256 `1a42ca6716305f5fdc3582e4b032554dd034a3687a102c12789fe5a6beef9d10`.
Each row links the exact local ASL-LEX entry but does not copy its reference video.
The official ASL-LEX license permits personal searches but prohibits saving,
displaying, or reusing reference videos without permission, so the workflow is
links-only (`https://asl-lex.org/download.html`). Capture is intentionally blocked
until an ASL-fluent reviewer marks every row approved with a pseudonymous reviewer ID
and timezone-aware timestamp. English-label agreement or a normalized/numeric variant
is not approval.

The expanded protocol and commands are in
`docs/guides/PORTRAIT_IPHONE_EVAL_V17.md`; the capture schema is in
`active/v17/portrait_iphone_capture_template.csv`. Six new workflow tests pass. The
existing 43 focused Stage-1 tests also pass unchanged, both new Python files compile,
and scoped `git diff --check` passes. Relevant SHA-256 hashes are script
`dc49e64c2ace80875fd0dc767cdbdeedf232aa4b3ca35084ffc217aab067dd1c`,
tests `34a5af3ff9c2e5f5e82d7812f28bcc4d69bd597ed51262ff2b4cc790f3978f31`,
guide `11cb804be7430ecbc81788b1d6ca82f682a1a360e4616c57382cfc918be5ddad`,
and ledger schema `0c960c82c96446efb9053a16b432d0b4c516296bc61cc7068986c316d505ad74`.

Kaggle remains reachable through the CLI, but no job was launched: this gate requires
genuinely new iPhone captures and human variant confirmation rather than more cloud
training. A narrow 2026 primary-source web check found no public replacement for the
capture: PopSign/PopSignAI still use the same one-handed Pixel-4A PopSign v1.0 corpus
(`https://openreview.net/forum?id=yEf8NSqTPu`,
`https://doi.org/10.1145/3742413.3789164`), FSboard is fingerspelling rather than the
100 isolated lexical variants (`https://arxiv.org/abs/2407.15806`), and ASL-100-RGBD
is Kinect capture rather than portrait iPhone
(`https://www.sign-lang.uni-hamburg.de/lrec/pub/20034.html`). None supplies the new
iPhone signers, two-handed coverage, and exact frozen variants required here.

Neither frozen test split, any model checkpoint, nor any existing dataset was accessed
or changed. The next safe action is the 100-row ASL-fluent review; after approval, run
`build-pack` with five genuinely new signer pseudonyms and immediately run the setup
audit before capture.

## 2026-08-12 05:14 PST — zero-deployment-cost masked-pose trial staged

A SHuBERT/MS-MAE-inspired multi-stream masking component is now implemented without
their full models or external corpora. During pretraining, four-frame spans are
sampled independently for left hand, right hand, face, and body at nominal ratio
0.35. All five public v17 channels are hidden for the selected nodes; the unchanged
part-wise encoder reconstructs masked XYZ with Smooth-L1, presence with binary cross
entropy, and observed confidence with MSE. Only approved Citizen-train and SemLex-
train clips are loaded. Validation and both tests are absent from pretraining.

The temporary 78,385-parameter reconstruction decoder is discarded. Exactly 249
encoder tensors (6,591,808 parameters) are strict-loaded into the unchanged
6,791,717-parameter part-wise classifier before ordinary full fine-tuning. Thus the
existing seed-1701 part-wise run is the architecture/seed control and this treatment
adds zero inference parameters or preprocessing. The loader fails closed on model
config, schema, both manifest hashes, and the exact encoder-key set.

All 41 focused Stage-1 tests pass, including independent part-span coverage,
finite masked reconstruction, and strict encoder-only loading. A real-data two-step
pretraining smoke plus two-batch/full-validation downstream smoke passed; the latter
loaded all 249 encoder tensors, retained 6,791,717 parameters, and kept all validation
and test access flags false. Source hashes are
`015a84413d4c591b197ccbf88f1bd937ff77f461245d520455d64c8503f6d46f`,
`75ca0f66b5e632fc1a26ebdc4bb4c75621942734e38712f271e99dc5ee77aa90`, and
`f2b95df892c69ac357eaef9f78e427f3851c9e3edb356d5f155ea772c4274892`.
The staged private overlay archive under
`artifacts/generated/kaggle_stage1_masked_pose_overlay_v1/` has SHA-256
`efb9bbbbfbbfc0d54ababc067238172f78607e5671f6a07d015d1ae1b02ba559`;
its manifest declares no test data. The sequential pretrain/fine-tune runner under
`artifacts/generated/kaggle_stage1_masked_pose_kokoab_v1/` is now active. Private code
dataset `kokoab/slt-v17-stage1-masked-pose-code-v1` is ready and private kernel
`kokoab/slt-v17-stage1-masked-pose-v1` version 1 is RUNNING on a T4. No local heavy
process is active.
