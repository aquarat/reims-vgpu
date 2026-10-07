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
