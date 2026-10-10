# aquarat/reims-vgpu: Asahi Linux (arm64) host build

This fork's `master` is **steelbrain-bot/reims-vgpu `master`** plus a few
commits for arm64 Linux hosts, and `vendor/qemu` points at
**aquarat/qemu-reims-vgpu `master`** (upstream QEMU + steelbrain's vmapple
work + Asahi KVM support). Nothing here is submitted upstream.

Commits on top of upstream:

- `device`: report native FP16 as unsupported on CPU Vulkan devices
  (Anees Iqbal). llvmpipe cannot compile the native-FP16 shader path.
- `vulkan`: use `c_char` for extension name pointers (it is `u8` on aarch64
  Linux, `i8` elsewhere).
- `vendor/qemu`: track aquarat/qemu-reims-vgpu `master`.

On Asahi hosts, run with `VK_DRIVER_FILES=…/lvp_icd.aarch64.json` (llvmpipe).
Mesa's Honeykrisp compiler asserts on the guest's 16-bit varyings.

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
in aquarat/experiment-macos-arm64-on-asahi-linux-arm64.

## Taking upstream changes

```sh
git remote add upstream https://github.com/steelbrain-bot/reims-vgpu.git
git fetch upstream master && git merge upstream/master
# after updating aquarat/qemu-reims-vgpu master:
git -C vendor/qemu fetch origin master && git -C vendor/qemu checkout origin/master
git add vendor/qemu && git commit -m "vendor/qemu: bump"
```

Merges keep upstream's `.gitmodules` URL changes in view: keep ours
(aquarat/qemu-reims-vgpu, branch master). `scripts/sync-upstream.sh` in
aquarat/experiment-macOS-arm64-on-asahi-linux-arm64 automates both forks.
