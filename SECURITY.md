# Security Policy

## Supported versions

This is a hobby project. Only the latest commit on `main` is supported.

| Version | Supported |
| --- | --- |
| `main` (latest) | Yes |
| Anything older | No |

## Reporting a vulnerability

Please **do not** open a public issue for a security vulnerability.

Report it privately through GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
feature — the **Security** tab of this repository, then **Report a
vulnerability**.

Please include what the issue is, how to reproduce it, and what an attacker
could achieve. A capture URL or a minimal reproduction helps considerably.

As a personal project, response is best-effort rather than guaranteed. Expect an
acknowledgement within about a week.

## Threat model — what is and is not in scope

Being honest about this is more useful than a generic policy, because the
attack surface here is unusually small.

**The application has no backend, no server, no database and no accounts.** It
is a static bundle of HTML, JavaScript and CSS. It makes **no network requests
at runtime** — the application source contains no `fetch`, `XMLHttpRequest`,
WebSocket, `EventSource` or telemetry call, and no CDN is referenced. Loading
the built site and playing it issues exactly three HTTP requests: the HTML, one
JavaScript bundle and one stylesheet. (The bundle does contain Vite's
module-preload polyfill, which references `fetch`; it is inert for this build,
but the honest claim is "no requests are observed" rather than
"the string `fetch` does not appear".)

It loads no external assets, because it has none. It does not read or write
cookies, and stores no credentials.

Consequently the classic web vulnerability classes largely do not apply: there
is no server to attack, no authentication to bypass, no user data at rest, and
no third-party origin in the trust path.

**In scope:**

- A supply-chain compromise of a build or runtime dependency
  (`three`, `postprocessing`, `n8ao`, `three-mesh-bvh`).
- A flaw in the GitHub Actions workflow that could allow code execution or
  secret exfiltration during a build.
- Any code path that unexpectedly introduces a network request or persists data.
- Cross-site scripting via URL parameters — the capture harness parses
  `?shot=`, `?cam=` and related values, and although these are all coerced to
  numbers or matched against fixed allowlists, a bypass would be a genuine
  finding.

**Out of scope:**

- Denial of service by asking the renderer to draw something expensive. You can
  make your own GPU busy; this is a game engine and that is expected.
- Browser or GPU driver crashes caused by WebGL, which belong upstream.
- Missing security headers on GitHub Pages, which is not under this project's
  control.
- The absence of Subresource Integrity, which is not applicable — no external
  resources are loaded.
