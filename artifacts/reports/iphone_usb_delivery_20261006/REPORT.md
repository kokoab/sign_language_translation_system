# iPhone USB delivery experiment

Date: 2026-10-06, Asia/Manila. Scope: deployment transport and browser requirements;
no SLT application installation, model conversion, training or accuracy evaluation.

**Follow-up:** expanded work found an active direct link-local USB interface that
remained reachable with Mac Wi-Fi off and sharing disabled (6/6 ping replies).
This is a different address/path from the bridge tested below. Browser delivery and
fresh-phone initialization are unverified. Read
[reconsidered options](RECONSIDERED_OPTIONS.md) before treating this report as a
decision to abandon USB.

## Finding

**Safari on the connected iPhone successfully loaded a page from the Mac over USB.
The tested configuration did not keep that USB network working when the Mac's
Wi-Fi was turned off. It is not an established solution for the offline venue.**

This corrects the earlier uncertainty about whether Safari can reach the Mac over
an ordinary phone cable: it can after enabling Apple's tethered connection sharing.
USB pairing alone did not create that network.

| Check | Result | Evidence and limits |
| --- | --- | --- |
| USB device pairing | Pass | iPhone 13, wired transport, paired and connected. |
| Safari fetch from Mac | Pass | Server received `192.168.234.2 GET / HTTP/1.1` with status 200; user confirmed the page displayed. |
| Dedicated cable network | Pass with Mac upstream available | `bridge100`, Mac `192.168.234.1`, iPhone peer `192.168.234.2`, USB member `en7`. |
| Mac Wi-Fi disabled, first run | Fail | Bridge disappeared; 0/5 ping replies; local HTTP endpoint became unreachable. |
| Mac Wi-Fi disabled, repeat | Fail | Bridge briefly survived at the initial snapshot; absent at 5 and 12 seconds; iPhone remained connected by wire. |
| Recovery after Mac Wi-Fi restored | Pass after recovery delay | Bridge and phone reachability returned; not immediate in the repeat's first post-restoration check. |
| Camera/offline-PWA APIs on plain HTTP IP | Fail in Mac Safari diagnostic | Insecure context; service worker and camera API unavailable. This diagnostic was not run inside iPhone Safari. |
| iPhone reload/restart/next-day persistence | Not tested | Remote Safari inspection unavailable; no PWA installed. |
| Complete SLT application in Safari | Not implemented | Native Apple Vision/Core ML pipeline has no browser implementation. |

## Equipment and configuration

Observed environment: iPhone 13, iOS 26.7.1; MacBook Air, macOS 27.0 build 26A428.
These are the values reported by the local tools. This is one already-paired phone,
with Developer Mode already enabled. The experiment did not enable Developer Mode,
but it does not establish identical onboarding on an unpaired iOS 17 device.

The Mac initially used Wi-Fi for its upstream connection. Its Ethernet interfaces
were inactive. Content Caching and USB Internet Connection sharing were initially
off. The test enabled both through System Settings → General → Sharing → Content
Caching. An attempted Only Shared Content selection did not persist; the actual
activated setting was All Content. No cache files or participant data were read.

The server bound only to the USB bridge address, port 8765, and exposed one
disposable test directory, not the SLT repository. It served HTML created on the
Mac, not downloaded app/model assets. Missing favicon requests produced 404s;
TLS-looking requests to the HTTP-only server produced 400s. Neither installed
HTTPS or demonstrated camera/PWA support.

[Apple's documented tethered-sharing setup](https://support.apple.com/en-ph/guide/deployment/dep38ff24bed/web)
requires Ethernet internet on the Mac. This experiment worked initially with a
Wi-Fi upstream, outside that documented setup. That observation is limited to
this equipment; it is not a support guarantee.

## Offline experiment

Two bounded scripts disabled Mac Wi-Fi and restored it in a `finally` block.
There was no active wired upstream. In the first run, the default route was absent,
the USB bridge disappeared, and all five phone pings failed. Content Caching still
reported Active=true and TetheratorStatus=1, showing that these status flags alone
do not prove a usable phone network.

The repeat checked immediately, at 5 seconds and at 12 seconds. Its initial ping
passed, then both later checks failed with the bridge absent. Device information
still reported connected, paired and wired. This rules out treating the failure
as simply an unplugged cable. After upstream restoration, later pings passed again.

**The tested condition was Wi-Fi radio off with no Ethernet connection.** It does
not prove that sharing fails on every isolated network, nor that every USB method
requires internet. An associated Wi-Fi network without WAN access, alternative
network configuration, and cold activation entirely offline were not tested.
They must not be presented as working alternatives on the basis of this report.

## Automation limits

Mac Safari's Apps and Devices Inspection showed **Enable Web Inspector on device**.
There was no enabled phone-control surface available to automate iPhone Safari.
No inspector utility was installed and no phone security setting was changed.
Consequently, the phone fetch is backed by server evidence and the user's
confirmation; subsequent radio-off checks measure Mac/phone network reachability,
not an automated iPhone browser reload.

No phone reboot, Home Screen installation, camera permission prompt, model
download, inference benchmark or next-day test was performed.

## Restored state

- Mac Wi-Fi is on again.
- Content Caching is off; USB Internet Connection sharing is off.
- Original All Content preference is retained; ordinary Internet Sharing is off.
- Mac Safari's Show features for web developers preference is restored to off.
- Probe server stopped; port 8765 has no listener; temporary USB bridge is gone.
- Approximately 393 KB was automatically cached during the experiment. It remains
  in the system-managed cache; no cache purge or unrelated deletion was performed.
- Phone radios remain as the user left them. They were not controlled by the agent.

## Evidence files

- [First offline probe](network_probe.json)
- [Repeated offline probe](network_repeat.json)
- [Browser observations and phone fetch](browser_diagnostic.json)
- [Cleanup verification](cleanup.json)
- [Original test page](index.html) and [browser diagnostic page](diagnostics.html)
- [Complete-app feasibility assessment](APP_FEASIBILITY.md)

## Decision

Do not select the tested tethered-sharing configuration for a venue with neither
Wi-Fi nor Ethernet. It proves a useful cable transport when the Mac has an upstream,
but does not meet the user's venue conditions. This is a failed configuration test,
not proof that all offline USB implementations are impossible.

The remaining candidate that has an officially supported offline transport is a
team Android local-only hotspot, if the user accepts a brief network join. It still
needs trusted HTTPS and a validated browser app. No complete zero-budget route
satisfying every current constraint has been verified.
