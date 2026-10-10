# aquarat/reims-vgpu: Asahi Linux (arm64) host build

Reims is [steelbrain-bot/reims-vgpu](https://github.com/steelbrain-bot/reims-vgpu)
(see [README.md](README.md)); all credit for the device, its design and the
Metal-to-Vulkan path goes to its authors. This fork adds what is needed to
run it on Apple Silicon hosts running Asahi Linux, with macOS 26 guests under
KVM. Nothing here is submitted upstream, and this file describes only this
fork.

This fork's `master` is steelbrain-bot/reims-vgpu `master` plus the commits
below, and `vendor/qemu` points at
[aquarat/qemu-reims-vgpu](https://github.com/aquarat/qemu-reims-vgpu)
`master` (upstream QEMU, steelbrain's vmapple work and Asahi KVM support).
metal2vulkan comes from
[aquarat/metal2vulkan](https://github.com/aquarat/metal2vulkan): steelbrain's
metal2vulkan plus one commit (a narrower vector load from a union, used by
macOS 26 Core Animation shaders).

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
- `vendor/qemu`: tracks aquarat/qemu-reims-vgpu `master`.

`git log` has the details of each.

## Running on Asahi

Use Mesa's Honeykrisp (Asahi Vulkan) driver with the 16-bit varying patch
from the host repository (`patches/mesa/`, built by
`scripts/build-mesa-honeykrisp.sh`) and point `VK_DRIVER_FILES` at it. Stock
Honeykrisp aborts on the iOS simulator's 16-bit varyings. llvmpipe
(`lvp_icd.aarch64.json`) renders the desktop but cannot compile a shader the
iOS simulator uses.

Guest RAM must be a shared memfd (`memory-backend-memfd,share=on`), so the
device can map scattered guest pages.

## Performance

macOS 26.4 guest running the iOS 26.4 simulator, Vulkan on Honeykrisp,
this fork at 2a85d74105 with QEMU 2cd151d3b4 (drain on a worker thread):

- Simulator unit tests (4 vCPU / 8 GB): 155.8 s, against 154.4 s and
  160.0 s with no paravirtual GPU (`gfx-device=none`), and 204–225 s for
  earlier builds that drained on QEMU's main loop.
- An app's 14 offline XCUITests (8 vCPU / 12 GB): 640–766 s over five runs,
  against 1119–1170 s with the drain on the main loop.
- Longest drain tranche over a UI-test session: 2.6–3.5 s (6.2–9.7 s before
  dependency-graph compaction); vCPU wait for the device lock 17–19 s per
  session (100–135 s).
- 5 of 5 sustained stress runs with no guest panic and no host GPU hang.

Charts, data and method: [PERFORMANCE.md](https://github.com/aquarat/experiment-macos-arm64-on-asahi-linux-arm64/blob/master/docs/PERFORMANCE.md)
in the host repository.

## Known gaps

- Pipelines with no fragment function are refused, so screenshots of
  Compose/Skia apps are mostly flat colour. UI tests that use accessibility
  are not affected. Being fixed.
- `memcpy` is 70 % of the drain's CPU time after compaction; part of it is an
  extra copy through an intermediate buffer, which can go.

## Taking upstream changes

```sh
git remote add upstream https://github.com/steelbrain-bot/reims-vgpu.git
git fetch upstream master && git merge upstream/master
# after updating aquarat/qemu-reims-vgpu master:
git -C vendor/qemu fetch origin master && git -C vendor/qemu checkout origin/master
git add vendor/qemu && git commit -m "vendor/qemu: bump"
```

Merges keep upstream's `.gitmodules` URL changes in view: keep ours
(aquarat/qemu-reims-vgpu, branch master). `scripts/sync-upstream.sh` in the
host repository automates both forks.
