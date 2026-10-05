# Android runtime investigation — 2026-10-06

Correctness: saved landmark-only phone replay matches Python on89/89 videos and5,161 frame commitments; Stage3 sentences match89/89. These are fixture parity, not live-camera accuracy.

Production inference behavior and model/config selections remain unchanged. Debug APK is built and installed with isolated, affinity, direct-buffer and CPU-time instrumentation.

## Boundary-pool isolation (ABBA,30samples/run)

Four threads, display priority(-4), default Android scheduling. External run-as affinity was denied; both modes equally use default scheduling.

| Run | Mode | Median ms | P90 ms |
|---|---|---:|---:|
| paired_v3_1_isolated | isolated | 241.8 | 365.0 |
| paired_v3_2_combined | combined | 281.9 | 383.2 |
| paired_v3_3_combined | combined | 291.0 | 442.5 |
| paired_v3_4_isolated | isolated | 324.3 | 452.1 |

No consistent isolation gain: isolated medians241.8/324.3ms, combined281.9/291.0ms. Competing boundary-model pool is not established as the cause.

## In-app affinity (ABBA,30samples/run)

Four threads, display priority, isolated model. Debug taskset ran inside the app on the inference thread before creating model workers. Both pinned logs confirm ff→f0.

| Run | Affinity | Median ms | P90 ms |
|---|---|---:|---:|
| affinity_v1_1_default | default | 190.3 | 422.1 |
| affinity_v1_2_pinned | f0 | 340.5 | 401.9 |
| affinity_v1_3_pinned | f0 | 352.1 | 400.3 |
| affinity_v1_4_default | default | 185.4 | 440.3 |

Pinning degraded the median in this paired test. No production pinning applied. Worker inheritance is intended; worker-specific affinity/cycle placement was not measured in this run.

## Direct buffers versus wrapper (ABBA,30samples/component/run)

All isolated, four threads, display priority, default scheduler. Direct model gets the same repeated span input; no-write reuses its prepared buffer. CPU/run is process CPU across all app threads, not wall time or host CPU percentage.

| Run | Component | Median ms | P90 ms | Process CPU ms/run |
|---|---|---:|---:|---:|---:|
| direct_v1_1_wrapper | span_landmark_only_b8 | 196.1 | 413.9 | 914.7 |
| direct_v1_2_direct | direct_no_write_b8 | 265.8 | 360.9 | 993.7 |
| direct_v1_2_direct | direct_write_and_run_b8 | 250.5 | 334.6 | 919.0 |
| direct_v1_3_direct | direct_no_write_b8 | 185.2 | 299.3 | 771.5 |
| direct_v1_3_direct | direct_write_and_run_b8 | 185.0 | 273.7 | 758.6 |
| direct_v1_4_wrapper | span_landmark_only_b8 | 298.6 | 394.1 | 1100.5 |

Repeated input writes and wrapper bypass do not explain the slowdown: second direct pair185.2ms no-write vs185.0ms write+run; variability persists inside native model execution. These are sequential phone microbenchmarks with limited repetitions; causation/real-time readiness is not established.

## Build and installation recovery

Initial volume disconnect/class-format failures left a Gradle daemon spinning at~919%CPU; original and retry processes were stopped. Subsequent builds succeeded with --no-daemon,--max-workers=2,JAVA_TOOL_OPTIONS ActiveProcessorCount2/Xmx2g, overridden Gradle JVM arguments, Kotlin in-process and external180s deadline. These restrict concurrency; they are not a hard measured CPU percentage ceiling. Later instrumentation builds took14-22seconds. No Gradle process remained in the final host process snapshot.

APK CRC validation passed all827members. Separate APK transfer takes14-15seconds; Huawei package installation sometimes presents an unknown-source confirmation. Parent observed and confirmed the already-authorized installation, without changing blanket device settings. Latest install date10:30:27; device shows Installation successful.

Instrumentation fixes: am start -W avoids premature pidof; app uses Log.println(INFO) to bypass Huawei Log.i suppression while global diagnostic flag is off. Per-tag log.tag.SLTParity was temporarily enabled for tests; no global Huawei logging switch changed.

## Scope, execution and next safe action

Changed app code only in debug LiveParityActivity.kt. Parent reviews code, authors runners and interprets experiments; Luna executes exact commands and waits on completion messages. Parent mailbox waits do not repeatedly poll process status. Exact token savings are unavailable.

No training, protected-test access, dependency upgrade, or production runtime workaround. Existing LiteRT CompiledModel is retained. Standalone4bigcore mean108.799ms is historical same-session shell microbenchmark under different placement; not an app frame estimate.

Next diagnostic needs native operator/thread/frequency tracing under matched conditions. LiteRT2.2 source/options were reviewed; do not pass raw XNNPACK runtime worker flags into delegate flags, which are different contracts. No unsupported flag was applied. Primary source: https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/litert/runtime/compiled_model.cc ; https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/tflite/delegates/xnnpack/xnnpack_delegate.h .

Complete native live-camera sustained latency and signer evaluation remain required. The50ms target is not demonstrated; current measured decoder observe mean247.29ms excludes MediaPipe/camera/Stage3.

## Live-camera empty-scene smoke

Parent launched Flutter home, observed models ready and LANDMARKS default, opened
native Live, pressed Start and observed frame counter advance to265 without crash.
Final UI snapshot showed10.3FPS/85ms latest frame, zero hands; these are UI snapshot
values, not an aggregate signed-input latency benchmark. Scene had no signer.
Stopped with Finish and Back. Empty smoke sessions may be retained in app history;
none deleted. Camera/MediaPipe execution smoke passes; signer accuracy, complete
active-sign latency and sustained thermal evaluation remain open. Temporary
SLTParity per-tag logging property restored to its initial empty value.
