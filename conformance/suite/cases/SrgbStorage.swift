import Metal
import Foundation
import IOSurface

// S. A compute kernel writing, reading and re-writing a `BGRA8Unorm_sRGB`
// texture, over a buffer, in private memory and over an IOSurface.
//
// A kernel that writes an sRGB texture stores linear values and the texture
// holds them encoded; a kernel that reads one gets the stored codes decoded.
// Both halves are invisible to a screenshot when they go wrong — an unencoded
// store and an undecoded read are each just "a bit dark" or "a bit light" —
// and both have gone wrong on this device: a write to a linear sRGB texture
// was refused outright (`linear_tex_fmt_storage`, 243 times in one simulator
// UI test, whose home screen then lost its icons), and an sRGB surface view
// was bound as its UNORM sibling, so writes landed unencoded and reads saw
// codes as values.
//
// Each case predicts its bytes from the sRGB transfer function and reads them
// back without a shader in between where the backing allows it — the buffer's
// own bytes, the IOSurface's own bytes — so the readback cannot apply or undo a
// transfer function of its own. A private texture is copied out by a blit,
// which moves bytes.
//
// # The expectation allows one code
//
// Metal encodes through a twelve-bit table: 0.5 is stored as 187 on an M-series
// device, where the transfer function rounded to the nearest code says 188
// (187.51). A device encoding by the formula is right too, so every channel is
// held to within one code of the formula, except 0 and 1, which every encoder
// stores exactly. An unencoded store misses by far more than that — 0.5 lands
// on 128 and 0.002 on 1 instead of 7 — and an R/B swap moves whole values,
// since no texel carries the same value in red and blue.
//
// The round trip is exact: Metal's decode is a step of its own encode table, so
// a code read and written back is the same code, and a device that drifts by a
// code per pass is a device on which a repeated filter walks the picture away.

let srgbStorePipe = pipeline("srgb_store_ladder")

let srgbRoundTripPipe = pipeline("srgb_round_trip")

/// The values `srgb_store_ladder` writes, by column modulo four. Red takes
/// `ladder[i]`, green `ladder[i + 1]`, blue `ladder[i + 2]`, alpha 0.5.
let srgbLadder: [Double] = [0.0, 0.002, 0.5, 1.0]

/// The eight-bit sRGB code of linear `v`, by the transfer function.
func srgbEncode(_ v: Double) -> Int {
    let c = v <= 0.0031308 ? 12.92 * v : 1.055 * pow(v, 1.0 / 2.4) - 0.055
    return Int((c * 255.0).rounded())
}

/// The linear value of eight-bit sRGB code `c`, by the transfer function.
func srgbDecode(_ c: Int) -> Double {
    let x = Double(c) / 255.0
    return x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4)
}

enum SrgbBacking: String, CaseIterable {
    case buffer, `private`, iosurface
}

/// One `BGRA8Unorm_sRGB` texture and the means to put bytes in and take them
/// out without a shader.
struct SrgbTarget {
    let tex: MTLTexture
    let w: Int
    let h: Int
    /// Tight-packed BGRA bytes, row-major.
    let read: () -> [UInt8]
    let write: ([UInt8]) -> Void
}

func makeSrgbTarget(_ backing: SrgbBacking, _ w: Int, _ h: Int,
                    _ label: String) -> SrgbTarget? {
    let tight = w * 4
    let d = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm_srgb, width: w, height: h, mipmapped: false)
    d.usage = [.shaderRead, .shaderWrite]
    switch backing {
    case .buffer:
        let align = dev.minimumLinearTextureAlignment(for: .bgra8Unorm_srgb)
        let bpr = alignUp(tight, align)
        guard let buf = dev.makeBuffer(length: bpr * h, options: .storageModeShared) else {
            report(label, false, "buffer allocation failed"); return nil
        }
        d.storageMode = .shared
        guard let tex = buf.makeTexture(descriptor: d, offset: 0, bytesPerRow: bpr) else {
            report(label, false, "makeTexture(buffer) nil for \(w)x\(h) pitch \(bpr)"); return nil
        }
        let base = buf.contents().bindMemory(to: UInt8.self, capacity: bpr * h)
        return SrgbTarget(
            tex: tex, w: w, h: h,
            read: { (0..<h).flatMap { y in (0..<tight).map { base[y * bpr + $0] } } },
            write: { bytes in
                for y in 0..<h { for x in 0..<tight { base[y * bpr + x] = bytes[y * tight + x] } }
            })
    case .iosurface:
        let bgra: UInt32 = 0x4247_5241   // 'BGRA' as an OSType
        let props: [IOSurfacePropertyKey: Any] = [
            .width: w, .height: h, .bytesPerElement: 4, .pixelFormat: bgra,
        ]
        guard let surface = IOSurface(properties: props) else {
            report(label, false, "IOSurface(properties:) nil for \(w)x\(h)"); return nil
        }
        d.storageMode = .shared
        guard let tex = dev.makeTexture(descriptor: d, iosurface: surface, plane: 0) else {
            report(label, false, "makeTexture(iosurface:) nil for \(w)x\(h)"); return nil
        }
        let bpr = surface.bytesPerRow
        return SrgbTarget(
            tex: tex, w: w, h: h,
            read: {
                surface.lock(options: .readOnly, seed: nil)
                defer { surface.unlock(options: .readOnly, seed: nil) }
                let base = surface.baseAddress.bindMemory(to: UInt8.self, capacity: bpr * h)
                return (0..<h).flatMap { y in (0..<tight).map { base[y * bpr + $0] } }
            },
            write: { bytes in
                surface.lock(options: [], seed: nil)
                defer { surface.unlock(options: [], seed: nil) }
                let base = surface.baseAddress.bindMemory(to: UInt8.self, capacity: bpr * h)
                for y in 0..<h { for x in 0..<tight { base[y * bpr + x] = bytes[y * tight + x] } }
            })
    case .private:
        d.storageMode = .private
        guard let tex = dev.makeTexture(descriptor: d) else {
            report(label, false, "makeTexture(private) nil for \(w)x\(h)"); return nil
        }
        let size = MTLSize(width: w, height: h, depth: 1)
        let origin = MTLOrigin(x: 0, y: 0, z: 0)
        return SrgbTarget(
            tex: tex, w: w, h: h,
            read: {
                let out = dev.makeBuffer(length: tight * h, options: .storageModeShared)!
                memset(out.contents(), 0xEE, tight * h)
                let cb = queue.makeCommandBuffer()!
                let blit = cb.makeBlitCommandEncoder()!
                blit.copy(from: tex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: origin,
                          sourceSize: size, to: out, destinationOffset: 0,
                          destinationBytesPerRow: tight, destinationBytesPerImage: tight * h)
                blit.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                let p = out.contents().bindMemory(to: UInt8.self, capacity: tight * h)
                return Array(UnsafeBufferPointer(start: p, count: tight * h))
            },
            write: { bytes in
                let src = dev.makeBuffer(bytes: bytes, length: tight * h,
                                         options: .storageModeShared)!
                let cb = queue.makeCommandBuffer()!
                let blit = cb.makeBlitCommandEncoder()!
                blit.copy(from: src, sourceOffset: 0, sourceBytesPerRow: tight,
                          sourceBytesPerImage: tight * h, sourceSize: size, to: tex,
                          destinationSlice: 0, destinationLevel: 0, destinationOrigin: origin)
                blit.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
            })
    }
}

/// Run `pipe` over `tex` with the readback kernels' extent and witness
/// bindings. `false` means the dispatch never ran.
func srgbDispatch(_ pipe: MTLComputePipelineState, _ tex: MTLTexture, _ w: Int, _ h: Int) -> Bool {
    let ran = dev.makeBuffer(length: 4, options: .storageModeShared)!
    memset(ran.contents(), 0, 4)
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe)
    enc.setTexture(tex, index: 0)
    var extent = SIMD2<UInt32>(UInt32(w), UInt32(h))
    enc.setBytes(&extent, length: 8, index: 3)
    enc.setBuffer(ran, offset: 0, index: 4)
    let tg = 8
    enc.dispatchThreadgroups(
        MTLSize(width: (w + tg - 1) / tg, height: (h + tg - 1) / tg, depth: 1),
        threadsPerThreadgroup: MTLSize(width: tg, height: tg, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return ran.contents().bindMemory(to: UInt32.self, capacity: 1)[0] != 0
}

/// A fill of known codes: each texel a different BGRA quadruple, red and blue
/// never equal, alpha a code too (alpha is linear and must read back as itself).
func srgbCodeFill(_ w: Int, _ h: Int) -> [UInt8] {
    let codes: [UInt8] = [0, 1, 10, 64, 128, 188, 200, 255]
    var bytes = [UInt8](repeating: 0, count: w * h * 4)
    for i in 0..<(w * h) {
        bytes[i * 4 + 0] = codes[i % codes.count]            // B
        bytes[i * 4 + 1] = codes[(i + 3) % codes.count]      // G
        bytes[i * 4 + 2] = codes[(i + 5) % codes.count]      // R
        bytes[i * 4 + 3] = codes[(i + 6) % codes.count]      // A
    }
    return bytes
}

/// A kernel writes the ladder; the stored bytes are its sRGB codes.
func srgbStoreCase(_ backing: SrgbBacking, _ w: Int, _ h: Int) {
    let label = "srgb_store_ladder_\(backing.rawValue)_\(w)x\(h)"
    guard let t = makeSrgbTarget(backing, w, h, label) else { return }
    t.write([UInt8](repeating: 0xEE, count: w * h * 4))
    guard srgbDispatch(srgbStorePipe, t.tex, w, h) else { refused(label); return }
    let got = t.read()
    var bad: [(Int, Int)] = []
    var firstDetail = ""
    for y in 0..<h {
        for x in 0..<w {
            let i = x % 4
            let linear = [srgbLadder[i], srgbLadder[(i + 1) % 4], srgbLadder[(i + 2) % 4]]
            // Memory order is B, G, R, A.
            let want = [srgbEncode(linear[2]), srgbEncode(linear[1]), srgbEncode(linear[0]),
                        Int((0.5 * 255.0).rounded())]
            let exact = [linear[2], linear[1], linear[0], 0.5].map { $0 == 0.0 || $0 == 1.0 }
            let at = (y * w + x) * 4
            let have = (0..<4).map { Int(got[at + $0]) }
            let ok = (0..<4).allSatisfy { c in
                exact[c] ? have[c] == want[c] : abs(have[c] - want[c]) <= 1
            }
            if !ok {
                bad.append((x, y))
                if firstDetail.isEmpty {
                    firstDetail = "first_bad=(\(x),\(y)) want_bgra=\(want) got_bgra=\(have)"
                }
            }
        }
    }
    report(label, bad.isEmpty,
           bad.isEmpty ? "\(w * h) texels within one code of the transfer function"
                       : "\(bad.count)/\(w * h) wrong \(firstDetail) \(badMap(bad, w, h))")
}

/// Known codes put in by the CPU read back decoded through a plain `read`.
func srgbReadCase(_ backing: SrgbBacking, _ w: Int, _ h: Int) {
    let label = "srgb_read_decodes_\(backing.rawValue)_\(w)x\(h)"
    guard let t = makeSrgbTarget(backing, w, h, label) else { return }
    let fill = srgbCodeFill(w, h)
    t.write(fill)
    guard let got = readBack(readPipe, t.tex, w, h) else { refused(label); return }
    var bad: [(Int, Int)] = []
    var firstDetail = ""
    for y in 0..<h {
        for x in 0..<w {
            let at = (y * w + x) * 4
            let b = Int(fill[at]), g = Int(fill[at + 1]), r = Int(fill[at + 2]), a = Int(fill[at + 3])
            // `read_texels` reports round(value * 255) per channel, RGBA.
            let want = [srgbDecode(r), srgbDecode(g), srgbDecode(b)].map {
                Int(($0 * 255.0).rounded())
            } + [a]
            let v = got[y * w + x]
            let have = [Int(v & 0xFF), Int((v >> 8) & 0xFF), Int((v >> 16) & 0xFF), Int(v >> 24)]
            let ok = (0..<4).allSatisfy { c in
                c == 3 ? have[c] == want[c] : abs(have[c] - want[c]) <= 1
            }
            if !ok {
                bad.append((x, y))
                if firstDetail.isEmpty {
                    firstDetail = "first_bad=(\(x),\(y)) codes_bgra=[\(b),\(g),\(r),\(a)] "
                        + "want_rgba=\(want) got_rgba=\(have)"
                }
            }
        }
    }
    report(label, bad.isEmpty,
           bad.isEmpty ? "\(w * h) texels decoded"
                       : "\(bad.count)/\(w * h) wrong \(firstDetail) \(badMap(bad, w, h))")
}

/// A `read_write` kernel reads every texel and writes it back: the codes do not
/// move.
///
/// Metal admits `access::read_write` only on devices of read-write texture tier
/// 2, so a device reporting less skips rather than fails.
func srgbRoundTripCase(_ backing: SrgbBacking, _ w: Int, _ h: Int) {
    let label = "srgb_round_trip_\(backing.rawValue)_\(w)x\(h)"
    if dev.readWriteTextureSupport != .tier2 {
        skip(label, "readWriteTextureSupport is not tier 2")
        return
    }
    guard let t = makeSrgbTarget(backing, w, h, label) else { return }
    let fill = srgbCodeFill(w, h)
    t.write(fill)
    guard srgbDispatch(srgbRoundTripPipe, t.tex, w, h) else { refused(label); return }
    let got = t.read()
    var bad: [(Int, Int)] = []
    var firstDetail = ""
    for y in 0..<h {
        for x in 0..<w {
            let at = (y * w + x) * 4
            let want = Array(fill[at..<at + 4])
            let have = Array(got[at..<at + 4])
            if want != have {
                bad.append((x, y))
                if firstDetail.isEmpty {
                    firstDetail = "first_bad=(\(x),\(y)) want_bgra=\(want) got_bgra=\(have)"
                }
            }
        }
    }
    report(label, bad.isEmpty,
           bad.isEmpty ? "\(w * h) texels unchanged"
                       : "\(bad.count)/\(w * h) moved \(firstDetail) \(badMap(bad, w, h))")
}

/// Every sRGB storage case, over every backing.
func srgbStorageCases(_ w: Int, _ h: Int) {
    for backing in SrgbBacking.allCases {
        srgbStoreCase(backing, w, h)
        srgbReadCase(backing, w, h)
        srgbRoundTripCase(backing, w, h)
    }
}
