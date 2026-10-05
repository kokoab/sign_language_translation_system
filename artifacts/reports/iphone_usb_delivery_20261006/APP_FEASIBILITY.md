# Complete offline browser app feasibility

Date: 2026-10-06. Companion to [USB experiment](REPORT.md).

## Conclusion

**A successful HTML transfer is not a deployed offline SLT application.** The
current application uses native camera capture, Apple Vision landmarks, Core ML
models, native practice/live screens and local session files. A browser equivalent
requires implementation and validation; copying the Flutter web directory or
AirDropping an HTML/IPA bundle cannot preserve the current complete pipeline.

No browser app, model port, alternate inference family or native application was
implemented or installed during this experiment. No accuracy gates were consumed.

## Code findings

Inspected task-relevant files under
`/Volumes/secret/SLT/mobile_app/slt_mobile_app/`:

| File | Existing dependency | Browser consequence |
| --- | --- | --- |
| `lib/services/slt_inference_service.dart:42` | Flutter MethodChannel `slt_mobile_app/inference_v17`; native inference/live/practice calls | Requires a browser implementation of these operations. |
| `lib/shell/app_shell.dart:7` | `dart:io`, platform-specific behavior | Requires web-compatible platform/file handling. |
| `ios/Runner/LiveReel/LiveReelCore.swift:137` | Apple Vision hand, body and face landmark requests | Safari JavaScript cannot directly call this native extractor. Replacing it changes the model-input contract. |
| `ios/Runner/LiveReel/LiveReelModels.swift:11` | Native Core ML loading/compilation | Needs a browser runtime and compatible model exports. |
| `ios/Runner/LiveReel/LiveReelStage3.swift:12` | Core ML T5 encoder and decoder | Translation must also execute locally; replacing only recognition is incomplete. |
| `ios/Runner/LiveReel/ReelCamera.swift:26` | AVCaptureSession and native preview | Needs browser camera capture, orientation and timing parity. |
| `ios/Runner/LiveReel/ReelCamera.swift:213` | Session JSON under app Documents | Needs browser storage, history and export equivalents. |
| `web/index.html` | Standard Flutter bootstrap/manifest scaffold | Does not supply an implementation of the native inference channel. |

Existing MediaPipe/TFLite Android work is a possible source for a browser family,
not evidence of browser parity. It must retain its own extractor/model contract;
do not feed MediaPipe features into Apple Vision-trained models. The user accepts
a browser version only without regression, which has not been established.

## Secure browser loading

The plain HTTP diagnostic opened in **Mac Safari**, at the same USB-hosted origin,
reported:

```json
{
  "origin": "http://192.168.234.1:8765",
  "secureContext": false,
  "serviceWorkerAvailable": false,
  "cameraAPIAvailable": false
}
```

This is a desktop browser measurement, not an iPhone result. It agrees with the
[camera specification](https://www.w3.org/TR/mediacapture-streams/), which exposes
camera capture through secure contexts. The connection needs a trustworthy origin
for camera capture and service-worker registration. A successful HTTP page alone
does not establish either capability.

Service workers require appropriate HTTP(S) URLs and a trustworthy origin;
AirDropped `file://` HTML cannot establish the normal offline-PWA registration
path. [Registration requirements](https://developer.mozilla.org/en-US/docs/Web/API/ServiceWorkerContainer/register)
also mean that using the word localhost does not solve this: on an iPhone,
localhost refers to the iPhone, not the Mac.

A venue-ready browser setup would need certificate trust and reliable hostname
resolution on the local transport, prepared before the venue visit. Neither was
implemented or verified. No self-signed certificate, configuration profile or trust
override was installed on the phone.

## Offline persistence

The application shell, scripts, runtime, model weights, labels and translation
assets must all be stored on the phone before disconnection. Readiness should be
based on complete asset verification, not merely that the first page appeared.
Once established, a service worker can serve cached assets without the Mac.

WebKit provides storage quota and persistence APIs. Persistence requests are
granted according to browser heuristics, including Home Screen use; available
quota is not a storage guarantee. [WebKit storage policy](https://webkit.org/blog/14403/updates-to-storage-policy/)
supports implementing this flow, but does not prove this app survives every storage
condition. Offline reopen, restart and next-day use remain untested requirements.

## Transport alternatives

| Alternative | Current status |
| --- | --- |
| Tested USB tethered sharing with Mac upstream | Phone HTML fetch verified; no complete app or HTTPS bootstrap. |
| Tested USB sharing with Mac Wi-Fi off and no Ethernet | Failed in two runs; cannot recommend for this venue. |
| Other offline USB setup | Unverified; do not equate ordinary USB trust with browser networking. |
| Team Android local-only hotspot | Officially supports nearby communication without internet; no hardware trial performed. |
| AirDrop of model/data files | Possible file transport; cannot by itself establish the browser application's secure origin and service worker. |
| Existing native app signed with a free Personal Team | Still conflicts with rejected account authentication/Developer Mode requirements for participant onboarding. |

Android's [local-only hotspot API](https://developer.android.com/develop/connectivity/wifi/localonlyhotspot)
provides an offline network. It does not itself provide TLS, Safari caching or the
SLT inference implementation. If accepted, the Mac could supply assets across that
network; client reachability and all browser requirements still need checking on
the exact hardware. This candidate has not been selected.

## Next decision

Keep delivery infrastructure separate from model/app parity. A browser route
should not be presented as the complete current application until the full local
pipeline and offline lifecycle are measured. The minimal next experiment, if that
route is selected, is a trusted-HTTPS offline shell with camera and asset persistence
over a working venue transport, before porting models or promising deployment to
hundreds of participant phones.
