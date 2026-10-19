# aquarat/reims-vgpu: Asahi Linux (arm64) host build

Reims is [steelbrain-bot/reims-vgpu](https://github.com/steelbrain-bot/reims-vgpu)
(see [README.md](README.md)); all credit for the device, its design and the
Metal-to-Vulkan path goes to its authors. This fork adds what is needed to
run it on Apple Silicon hosts running Asahi Linux, with macOS 26 guests under
KVM. Nothing here is submitted upstream, and this file describes only this
fork.

This fork's `master` is steelbrain-bot/reims-vgpu `master` plus the commits
below. The device runs either inside QEMU or as its own program
(`reims-vgpu-backend`, below). For the in-process build `vendor/qemu` pins
[aquarat/qemu-reims-vgpu](https://github.com/aquarat/qemu-reims-vgpu) at
6740a49344, the last commit with the in-process shims; that repository's
`master` has only the out-of-process device.
metal2vulkan comes from
[aquarat/metal2vulkan](https://github.com/aquarat/metal2vulkan): steelbrain's
metal2vulkan plus two commits: a narrower vector load from a union (macOS 26
Core Animation shaders) and a byte pointee kept as bytes at the load (packed
vertex attributes read through a byte-record GEP).

This build is the GPU of a macOS CI runner setup that runs production CI
daily: one runner slot with the GPU serves an iOS app's UI tests. The host
side (kernel patches, images, runners) is in
[aquarat/experiment-macos-arm64-on-asahi-linux-arm64](https://github.com/aquarat/experiment-macos-arm64-on-asahi-linux-arm64).

## Commits on top of upstream

- arm64 Linux build: `c_char` for Vulkan extension name pointers (`u8` on
  aarch64 Linux, `i8` elsewhere) and in the QEMU ABI test; native FP16
  reported as unsupported on CPU Vulkan devices (Anees Iqbal), since
  llvmpipe cannot compile the native-FP16 shader path.
- macOS 26 guests: the macOS 26 IOSurface mapper kext, its request ring read
  at the wrapped slot, the device brought up before it answers device info,
  memoryless and heap-placed textures, framebuffer fetch of any colour
  attachment, RGB10A2Unorm and R16Float formats, private heap textures on
  hosts without Metal, and work stranded by a pipeline delete.
- Correctness and speed under load: a refused packet's completion word
  released in channel order, a generation base per guest task namespace,
  dependency-graph compaction at admission, compute sampling a draw's
  resident texture instead of re-uploading its pages, and a drain that ends
  its tranche when a vCPU waits and takes busy channels in turns.
- Logging: overridable, capped log sink paths; the AIR of a failed
  translation captured in the m2v cache.
- Copy path: guest data moves directly between guest RAM and mapped Vulkan
  memory instead of through an intermediate buffer. A packed buffer laid
  over the guest runs that hold a buffer's bytes, one page walk for every
  span copier, sampled linear levels gathered straight into staging and
  storage readbacks scattered straight into their surface mapping. A
  binding the resident already holds sends no zero-filled placeholder, and
  the translation cache's bucket digest uses a fast fold instead of SipHash.
- Rendering on macOS 26: render pipelines with no fragment function (Skia's
  stencil passes), a slot's type checked before it is asked for a texture
  view, and metal2vulkan's byte-record attribute loads, so Compose/Skia
  apps draw their real UI.
- Out of process: `reims-vgpu-backend`, the device as its own program,
  serving QEMU's `apv-gpu-proxy` device over a unix socket, shared memory
  and eventfds; the hot register words in memory both processes map; a
  driver quarantine per host, keyed by the driver build.
- A display refresh-rate cap (`REIMS_VGPU_MAX_REFRESH_HZ`) and optional
  surface pacing (`REIMS_VGPU_FRAME_PACING`), below.
- `vendor/qemu`: aquarat/qemu-reims-vgpu 6740a49344, for the in-process
  build.

`git log` has the details of each.

## Running on Asahi

Use Mesa's Honeykrisp (Asahi Vulkan) driver with the 16-bit varying patches
from the host repository (`patches/mesa/`, built by
`scripts/build-mesa-honeykrisp.sh`) and point `VK_DRIVER_FILES` at it. Stock
Honeykrisp aborts on the iOS simulator's 16-bit varyings. llvmpipe
(`lvp_icd.aarch64.json`) renders the desktop but cannot compile a shader the
iOS simulator uses.

Guest RAM must be a shared memfd (`memory-backend-memfd,share=on`), so the
device can map scattered guest pages.

### Out of process

```sh
cargo build --release -p reims-vgpu-backend
target/release/reims-vgpu-backend --socket /run/gpu.sock &
qemu-system-aarch64 -M vmapple,gfx-device=apv-gpu-proxy,aux-ram-share=on \
    -global apv-gpu-proxy.socket=/run/gpu.sock ...
```

with a QEMU built from aquarat/qemu-reims-vgpu `master`, which contains no
Reims, Rust or Vulkan code. The backend takes the same `REIMS_VGPU_*`
settings and `VK_DRIVER_FILES` from its environment. If the host GPU driver
crashes, only the backend dies: QEMU stops the guest with the reason, and the
driver call that was in flight is quarantined so later runs refuse it until
the driver changes. The host repository's `launch-kvm.sh` runs it all with
`GPU_BACKEND_BIN`, and `MODE=oop scripts/build-qemu.sh` builds both programs.

### Refresh-rate cap

`REIMS_VGPU_MAX_REFRESH_HZ` (default 60; `0` = the native 120;
`reims-vgpu-backend --max-refresh-hz` is the same setting) caps the refresh
rate the display modes advertise and the VBL cadence.

`REIMS_VGPU_FRAME_PACING=on` (`--frame-pacing on`; off by default) also paces
guest surfaces to that rate: work that writes a guest IOSurface completes at
most once per refresh interval per surface, holding its channel's later
completion words in order. A headless guest's other compositors — a sleeping
display's, the iOS simulator's — otherwise redraw as fast as the guest's CPU
allows. The simulator's queues dozens of frames ahead, so pacing it makes the
channel's completions trail by up to about a second. On the simulator
unit-test benchmark it made no clear difference (161.7–166.8 s paced, against
157.8 and 163.6 s), and it cost about 10 % of UI-test time, so it stays off.

## Performance

macOS 26.4 guest running the iOS 26.4 simulator, Vulkan on Honeykrisp,
this fork at 71029b4d00 with QEMU 6740a49344 (drain on a worker thread):

- Simulator unit tests (4 vCPU / 8 GB): 155.8 s, against 154.4 s and
  160.0 s with no paravirtual GPU (`gfx-device=none`), and 204–225 s for
  earlier builds that drained on QEMU's main loop.
- An app's 14 offline XCUITests (8 vCPU / 12 GB): 640–766 s over five runs,
  against 1119–1170 s with the drain on the main loop.
- Longest drain tranche over a UI-test session: 2.6–3.5 s (6.2–9.7 s before
  dependency-graph compaction); vCPU wait for the device lock 17–19 s per
  session (100–135 s).
- 5 of 5 sustained stress runs with no guest panic and no host GPU hang.

Out of process (40c5bc1c38 as `reims-vgpu-backend`, QEMU c6d87a8406, Mesa
with both patches; in production since 2026-10-19): simulator unit tests
157.8 and 163.6 s against 156.8–158.6 s for the same tree in process (one
more out-of-process run took 209.7 s, a host-side slowdown that did not
repeat); the app's suite 690–801 s over three sustained runs with no backend
loss. A backend killed mid-job fails that job in 2 s with the reason.

Charts, data and method: [PERFORMANCE.md](https://github.com/aquarat/experiment-macos-arm64-on-asahi-linux-arm64/blob/master/docs/PERFORMANCE.md)
in the host repository.

## Known gaps

- A cold app launch takes 20–40 s before its UI settles. The drain is
  saturated by the per-draw path (one queue submission per draw), not by the
  GPU or shader compilation.
- `memcpy` is still about two thirds of the drain's CPU time. Honeykrisp
  cannot import guest RAM (no `VK_EXT_external_memory_host`), so guest data
  is copied into and out of Vulkan memory once each way.
- Honeykrisp needs the 16-bit flat fragment input patch as well
  (`patches/mesa/0002-*` in the host repository); without it a Compose
  app's shaders abort the driver.

## Taking upstream changes

```sh
git remote add upstream https://github.com/steelbrain-bot/reims-vgpu.git
git fetch upstream master && git merge upstream/master
```

Merges keep upstream's `.gitmodules` URL changes in view: keep ours
(aquarat/qemu-reims-vgpu). `vendor/qemu` stays at 6740a49344 for the
in-process build; aquarat/qemu-reims-vgpu `master` no longer carries the
in-process shims, so bumping the submodule to it would break that build.
