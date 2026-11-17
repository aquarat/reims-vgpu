import Metal
import Foundation

// Fragment shaders that write a storage texture: `texture2d<float,
// access::write>` and `access::read_write` bound to a render encoder, the shape
// of the macOS 26 login window's full-screen RGBA16Float pass.
//
// Every value is exact in RGBA16Float and lands on an exact unorm8 in the
// readback (0.25 -> 64, 0.5 -> 128, 0.75 -> 191, 1 -> 255), so a case either
// matches everywhere or names the texels it lost. The stored texture is never
// the pass's attachment: an attachment of `bgra8Unorm` takes the incidental
// colour output, and only the storage texture is read back.

private let storageWidth = 64
private let storageHeight = 32
private let storageA: [Float] = [0.25, 0.5, 0.75, 1.0]
private let storageB: [Float] = [0.25, 0.0, 0.0, 0.0]

private func storageTexture(renderable: Bool) -> MTLTexture? {
    let d = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba16Float, width: storageWidth, height: storageHeight, mipmapped: false)
    d.usage = renderable ? [.shaderRead, .shaderWrite, .renderTarget] : [.shaderRead, .shaderWrite]
    d.storageMode = .private
    return dev.makeTexture(descriptor: d)
}

private func attachmentTexture() -> MTLTexture? {
    let d = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm, width: storageWidth, height: storageHeight, mipmapped: false)
    d.usage = [.renderTarget, .shaderRead]
    d.storageMode = .private
    return dev.makeTexture(descriptor: d)
}

private func colourPass(_ target: MTLTexture, clear: MTLClearColor) -> MTLRenderPassDescriptor {
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = target
    pass.colorAttachments[0].loadAction = .clear
    pass.colorAttachments[0].clearColor = clear
    pass.colorAttachments[0].storeAction = .store
    return pass
}

/// Clear `tex` through a render pass of its own, which is also what makes its
/// contents defined before a storage draw loads them.
private func clearStorage(_ tex: MTLTexture, _ c: [Float]) {
    let cb = queue.makeCommandBuffer()!
    let pass = colourPass(tex, clear: MTLClearColor(
        red: Double(c[0]), green: Double(c[1]), blue: Double(c[2]), alpha: Double(c[3])))
    cb.makeRenderCommandEncoder(descriptor: pass)!.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
}

/// One full-target draw per entry of `draws`, all in one encoder, each storing
/// through `pipe` with its own colour, separated by a fragment-to-fragment
/// barrier over the storage texture when `barrier` is set. `false` when the
/// resources would not build, so a case reports that rather than a readback of
/// a texture nothing drew into.
private func storageDraws(_ draws: [(MTLRenderPipelineState, [Float])],
                          into storage: MTLTexture,
                          barrier: Bool) -> Bool {
    guard let target = attachmentTexture(),
          let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                                     options: .storageModeShared) else { return false }
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeRenderCommandEncoder(
        descriptor: colourPass(target, clear: MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)))!
    enc.setVertexBuffer(verts, offset: 0, index: 0)
    enc.setFragmentTexture(storage, index: 0)
    for (i, (pipe, colour)) in draws.enumerated() {
        if i > 0 && barrier {
            enc.memoryBarrier(resources: [storage], after: .fragment, before: .fragment)
        }
        var c = colour
        enc.setRenderPipelineState(pipe)
        enc.setFragmentBytes(&c, length: 16, index: 0)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return true
}

private func unorm(_ c: [Float]) -> UInt32 {
    let q = c.map { UInt8(max(0, min(255, ($0 * 255).rounded()))) }
    return pack(q[0], q[1], q[2], q[3])
}

private func expectEverywhere(_ label: String, _ tex: MTLTexture, _ want: UInt32, _ detail: String) {
    guard let got = readBack(readPipe, tex, storageWidth, storageHeight) else {
        refused(label)
        return
    }
    let bad = got.enumerated().filter { $0.element != want }
    if bad.isEmpty {
        report(label, true, detail)
    } else {
        let first = bad[0]
        report(label, false,
               "\(bad.count)/\(got.count) texels wrong; first at \(first.offset) "
               + "got=\(hex(first.element)) want=\(hex(want)) "
               + badMap(bad.map { ($0.offset % storageWidth, $0.offset / storageWidth) },
                        storageWidth, storageHeight))
    }
}

func fragmentStorageWriteCases() {
    guard let write = makeRenderPipeline("storage_write_fs", .bgra8Unorm),
          let accumulate = makeRenderPipeline("storage_accumulate_fs", .bgra8Unorm) else {
        report("fragment_storage_pipeline", false, "storage fragment pipelines would not build")
        return
    }
    report("fragment_storage_pipeline", true, "write and read_write fragment pipelines built")

    // Write-only into a texture nothing has rendered or written before. Every
    // texel is covered, so the result does not depend on what the texture held.
    if let tex = storageTexture(renderable: false) {
        guard storageDraws([(write, storageA)], into: tex, barrier: false) else {
            report("fragment_storage_write_fresh_rgba16f", false, "resources would not build")
            return
        }
        expectEverywhere("fragment_storage_write_fresh_rgba16f", tex, unorm(storageA),
                         "a fragment store covered a never-written RGBA16Float texture")
    }

    // Write-only into a texture a render pass cleared first: the store must
    // replace the cleared contents, not land beside them.
    if let tex = storageTexture(renderable: true) {
        clearStorage(tex, [1, 0, 1, 1])
        guard storageDraws([(write, storageA)], into: tex, barrier: false) else {
            report("fragment_storage_write_after_clear_rgba16f", false, "resources would not build")
            return
        }
        expectEverywhere("fragment_storage_write_after_clear_rgba16f", tex, unorm(storageA),
                         "a fragment store replaced a render pass's clear")
    }

    writeThenSampleCase(write)

    // read_write needs tier-2 read-write textures for RGBA16Float.
    guard dev.readWriteTextureSupport == .tier2 else {
        skip("fragment_storage_read_write_rgba16f", "device has no tier-2 read-write textures")
        skip("fragment_storage_ordered_draws_rgba16f", "device has no tier-2 read-write textures")
        return
    }

    // read_write: the shader loads what an earlier pass left and stores the sum.
    if let tex = storageTexture(renderable: true) {
        clearStorage(tex, storageB)
        guard storageDraws([(accumulate, storageA)], into: tex, barrier: false) else {
            report("fragment_storage_read_write_rgba16f", false, "resources would not build")
            return
        }
        let want = zip(storageA, storageB).map { $0 + $1 }
        expectEverywhere("fragment_storage_read_write_rgba16f", tex, unorm(want),
                         "a read_write store loaded the texture's prior contents")
    }

    // Two overlapping full-target draws in one encoder, the second loading what
    // the first stored, ordered by a fragment-to-fragment texture barrier.
    if let tex = storageTexture(renderable: true) {
        clearStorage(tex, [0, 0, 0, 0])
        guard storageDraws([(write, storageB), (accumulate, storageA)], into: tex, barrier: true) else {
            report("fragment_storage_ordered_draws_rgba16f", false, "resources would not build")
            return
        }
        let want = zip(storageA, storageB).map { $0 + $1 }
        expectEverywhere("fragment_storage_ordered_draws_rgba16f", tex, unorm(want),
                         "an overlapping draw across a texture barrier loaded the earlier store")
    }
}

/// Stored in one pass, sampled in the next: the sample must see the store.
private func writeThenSampleCase(_ write: MTLRenderPipelineState) {
    guard let sample = makeRenderPipeline("tex_fs", .bgra8Unorm),
          let tex = storageTexture(renderable: true),
          let target = attachmentTexture(),
          let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                                     options: .storageModeShared) else {
        report("fragment_storage_write_then_sample", false, "resources would not build")
        return
    }
    clearStorage(tex, [1, 0, 1, 1])
    guard storageDraws([(write, storageA)], into: tex, barrier: false) else {
        report("fragment_storage_write_then_sample", false, "resources would not build")
        return
    }
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeRenderCommandEncoder(
        descriptor: colourPass(target, clear: MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)))!
    enc.setRenderPipelineState(sample)
    enc.setVertexBuffer(verts, offset: 0, index: 0)
    enc.setFragmentTexture(tex, index: 0)
    enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    // `read_texels` packs by channel name, so the bgra8 target reads back as
    // the same RGBA word the storage texture would.
    expectEverywhere("fragment_storage_write_then_sample", target, unorm(storageA),
                     "a later pass sampled what a fragment store wrote")
}
