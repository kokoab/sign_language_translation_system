# Deployment alternatives reconsidered

2026-10-06, Asia/Manila. Expanded primary-documentation, GitHub, Reddit and research
paper review, plus an additional reversible network experiment. No native app,
certificate/profile, helper app or model port installed.

## New measured result: direct link-local USB is still a candidate

The earlier experiment measured the failure of the managed `192.168.234.x` USB
bridge. Further inspection found `en7` remained active using self-assigned addresses:
Mac `169.254.167.131`, USB peer `169.254.92.174` (same peer MAC observed on the earlier
phone bridge). Content Caching and tethered sharing were disabled. Ordinary Internet
Sharing was also off during the new offline probe.

At 2, 8 and 16 seconds after turning off Mac Wi-Fi, `en7` remained active and the
peer answered 2/2 pings each time: **6/6 replies without an upstream default route**.
Wi-Fi was restored in a finally block. Evidence: [linklocal_probe.json](linklocal_probe.json).

This changes the next experiment: try browser delivery at the direct link-local
address before abandoning USB. It does not retroactively turn the previous bridge
failure into a pass. The phone was already paired, Developer Mode was already on,
and earlier tethered-sharing/developer tooling had activated USB interfaces. Fresh
pairing, cable replug, device reboot, iOS17 and Developer Mode-off operation are
unverified. Do not assume an unprepared participant exposes this same interface.
No Safari page fetch at the new link-local address was automated or confirmed.

During exploration, ordinary Internet Sharing was briefly enabled with inactive
Thunderbolt Bridge as its source and both displayed iPhone USB targets selected.
It did not create the earlier bridge. The link-local result was measured afterward,
with that sharing service off. Source restored to Wi-Fi, targets off, sharing off.
Do not attribute the link-local interface's origin conclusively to that brief test.

## Choices, separated by the participant compromise

| Route | Participant action | Potential benefit | Remaining gate |
| --- | --- | --- | --- |
| Direct link-local USB → secure web app | Cable, trust Mac, open page, add Home Screen, camera permission | Closest to current no-router/no-extra-app preference | Warm network only verified; Safari, fresh setup, HTTPS, full app port and offline lifecycle unverified. |
| Offline source → ordinary USB Internet Sharing | Same cable/browser flow | Separate method from Content Caching; may support repeatable setup | Inactive Thunderbolt source did not establish a managed bridge here; loopback-source USB combination remains an inference. |
| Mac creates offline Wi-Fi → web app | Briefly join Mac network, open/save app | No router, extra phone or venue internet | Community loopback workaround; administrator configuration, exact Mac compatibility, TLS and browser parity unverified. |
| Team Android local-only hotspot → Mac assets | Brief network join, open/save app | Officially supported offline transport | Requires available Android phone and browser/TLS validation. |
| Small App Store host + local app/model bundle | Install one free host; team imports/prepares local files | Phone hosts its own localhost; no developer signing of our browser assets | Host installation still needed at venue; importing, camera, ML runtime and complete-app behavior untested. |
| Participant Personal Hotspot over USB → Mac server | Enable Personal Hotspot and trust Mac | Alternative cable direction, no joining someone else's Wi-Fi | Carrier/cellular eligibility and local reachability vary; browser port still required. |
| Team phone cellular USB upstream → Mac → participant USB | Participant cable/browser flow | No venue Wi-Fi; model/app assets may still come from Mac | Requires team cellular service and testing two-phone sharing; not fully upstream-free. |
| Public enterprise certificate signing | Install/trust enterprise app; internet verification, possibly restart | Can preserve native Vision/Core ML rather than porting it | Certificate access, revocation, unauthorized distribution and offline onboarding problems; not dependable research distribution. |
| Existing paid publisher sponsors TestFlight/App Store | Ordinary Apple beta/store onboarding | Preserve current native app, no fee paid by this team | Requires willing existing publisher; venue download/onboarding constraints remain. |
| Free Personal Team signing | Account authentication and Developer Mode | Existing native app, seven-day testing period | Explicitly conflicts with the user's current rejected actions. |

## Mac-only offline network implementation evidence

A [GitHub implementation](https://gist.github.com/zhuhuilin/01656866b3e73a677a434c21183b40d2)
creates a loopback network service with a static address and shares it through Wi-Fi.
Its author reports Sonoma/Sequoia success. This is unverified on this macOS27 machine.
Source inspection found administrator requirements and broad cleanup operations;
none of these scripts was executed. A reviewed minimal implementation would need
to preserve the current configuration rather than run their reset routine.

The author's [Reddit discussion](https://www.reddit.com/r/MacOS/comments/1p4hznt/how_to_enable_macos_internet_sharing_without/)
also contains a report of an access point not being detected. This is community
evidence of a testable workaround, not an official compatibility guarantee.
Applying a loopback source to USB rather than Wi-Fi is a proposed combination,
not something those sources demonstrate.

## App Store host: a different way to use AirDropped assets

[JS Shell's Philippine App Store listing](https://apps.apple.com/ph/app/js-shell/id6763982889)
is free, lists6.6MB and iOS17+. Its developer describes a device-local HTTP server,
WKWebView camera permission forwarding, persistent Documents files and ZIP extraction.
These are vendor claims, not measured SLT results. Its actual download size and
large-project import flow were not measured. A single prepared launcher could be
investigated, with app/model assets transferred locally after host installation.

This supplies an already-signed execution host rather than making AirDrop install
a native IPA. It still requires one initial App Store installation and a compatible
browser version of SLT. Silent offline installation of the host on arbitrary
participants' phones is not established. Do not promise a whole-app AirDrop flow
without solving that initial installation.

Other findings: HTML Serve is currently iPad-only despite broad search snippets;
it is not an iPhone recommendation. a-Shell is a free interpreter candidate but
not established as a camera-capable complete SLT host. Pyto documentation lists
Vision/CoreML bridging, but its App Store listing includes paid tiers, so a zero-cost
complete native-framework route through it is not established.

## Alternative USB direction

Apple supports [iPhone Personal Hotspot over USB](https://support.apple.com/en-ca/guide/iphone/iph45447ca6/ios).
A [developer's firsthand report](https://stackoverflow.com/questions/28206154/view-local-development-server-on-usb-connected-ios-device)
describes opening the Mac's USB address from phone Safari. This is an older
field report, not proof for current equipment without cellular service.

A newer [Reddit Expo report](https://www.reddit.com/r/expo/comments/1pbtm48/solved_connecting_ios_to_expo_via_usb_when/)
reports reachability trouble in that direction and instead uses reverse sharing.
Its claim of offline operation does not establish fresh setup with all upstream
interfaces absent. Both directions deserve an exact-hardware test, not blanket
acceptance of a forum recipe.

## Native public-certificate installers

Scarlet's own documentation describes direct-install, computer and custom-certificate
methods. ESign-style tools likewise do not eliminate signing: another certificate
owner pays for the underlying distribution credential. The current availability
of a usable public certificate was not established, and none was downloaded.

[USENIX Security2026 research](https://www.usenix.org/system/files/conference/usenixsecurity26/sec26_prepub_liu-yijing.pdf)
documents enterprise-certificate misuse and Ad Hoc certificate resale as different
mechanisms. Public enterprise signing can avoid registering each device, but is
not a reliable authorized channel for external research participants. Revocation
can interrupt a test even if it lasts less than seven days.

[Apple's enterprise installation guidance](https://support.apple.com/en-us/118254)
requires internet to establish certificate trust and periodic reverification;
iOS18+ uses an Allow & Restart trust step. Thus, distributing a signed IPA from
the Mac does not solve fully offline first launch. DNS-based revocation bypasses
advertised on Reddit change participants' network verification behavior and are
not equivalent to a clean, dependable deployment. None was applied.

## Other routes reconsidered

- [LiveContainer](https://github.com/LiveContainer/LiveContainer): its normal setup
  still needs AltStore/SideStore or another signing route. Running guest apps does
  not eliminate installation/signing of the host. No full SLT guest parity proven.
- [TrollStore](https://github.com/opa334/TrollStore): supports exact older iOS ranges,
  including17.0, not a general17+ fleet or this phone's26.7.1.
- Shortcuts: Apple documents Run JavaScript as an action on a Safari webpage,
  requiring its script permission setting. It does not expose the current native
  SLT pipeline; a WebClip/bookmark is not a signed native application.
- Configurator/MDM/Expo/Firebase/Diawi-style distribution helpers do not create the
  native signing authorization they need. Existing App Store hosts are a separate
  option; their installation/licensing still needs a supported path.
- EU/Japan alternative distribution does not create a universal free iOS17+ IPA
  installation channel. Apple membership, regional and OS requirements remain.
- App Clips could preserve native frameworks but still need a publisher and do not
  solve the current zero-membership/offline first-delivery constraints.

## Prioritized next work

First investigate the measured direct USB link-local path: trustworthy-origin
Safari delivery and reproducible fresh-phone initialization. If that fails, the
Mac-only offline Wi-Fi workaround avoids purchasing or bringing a router. The small
App Store host is a separate compromise that could avoid network-based delivery
after installation. All browser routes still need complete local pipeline parity.

No route is selected. No additional participant action is silently assumed. The
earlier result should be read as failure of one managed bridge, not abandonment
of all USB delivery.
