//! The sampled draw rail's bind of a two-plane colour texture in guest memory.
//!
//! [`crate::runtime::two_plane`] places the planes and converts them; this hands
//! the converted texels to the engine as CPU bytes, with the content identity
//! that lets an unchanged pair skip the engine's own hash.

use super::vulkan::{LinearSampleIdentity, SampledSourceRequest};
use crate::model::{DeviceState, TaskResource};
use crate::protocol::pixel_format::SampledByteFormat;
use crate::runtime::decode::resource::Descriptor;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::two_plane::{self, TwoPlaneBindRefusal};

/// The sampled source of a type-12 `resource`, or `None` after naming why on
/// the failure channel (once per task, reference and reason).
pub(super) fn sampled_source<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    resource: &TaskResource,
) -> Option<(u32, u32, u32, SampledSourceRequest)> {
    let loaded = match resource.decoded() {
        Ok(Descriptor::DualPlaneTexture(dual)) => two_plane::guest_texels(
            state,
            host,
            task_id,
            texture_ref,
            dual,
            crate::runtime::render_writeback::SettleSite::LinearMemoRead,
        ),
        _ => Err(TwoPlaneBindRefusal::Undecoded),
    };
    match loaded {
        Ok(texels) => {
            crate::runtime::drain::note_store_route("sampled_two_plane");
            let format = SampledByteFormat::from_source(
                texels.conversion.output_layout(),
                texels.conversion.pixel_format(),
            );
            Some((
                texels.width,
                texels.height,
                0,
                SampledSourceRequest::Bytes(
                    texels.bytes,
                    // The guest-linear row of the identity keyspace: these
                    // texels are a conversion of guest-written bytes at `gva`,
                    // and the generation changes exactly when they do.
                    texels.plane0_gva.map(|gva| LinearSampleIdentity {
                        key: gva,
                        generation: texels.generation,
                    }),
                    format,
                    crate::backend::vulkan::engine::SampledByteOrigin::LinearTexture,
                ),
            ))
        }
        Err(refusal) => {
            let discriminant = (u64::from(task_id) << 32) | u64::from(texture_ref);
            crate::observe::Emit::decline("draw_sample_two_plane", &refusal)
                .field("task", task_id)
                .field("ref", texture_ref)
                .fail_once(discriminant);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::vulkan::{resolve_sampled_source, SampledSourceRequest};
    use crate::protocol::pixel_format::TexelLayout;
    use crate::runtime::two_plane::tests::guest_two_plane_texture;

    /// The sampled rail binds a type-12 `RGB8_420_2P` texture as its converted
    /// texels, with a content identity, where it used to decline the object
    /// type.
    #[test]
    fn a_two_plane_texture_binds_as_its_converted_texels() {
        let luma: Vec<u8> = (0..8u8).map(|i| 30 + 20 * i).collect();
        let chroma = [60u8, 200, 70, 190];
        let (mut state, mut host, task, texture_ref) =
            guest_two_plane_texture(540, (4, 2), &luma, &chroma);
        let Some((w, h, _, SampledSourceRequest::Bytes(bytes, identity, format, _))) =
            resolve_sampled_source(&mut state, &mut host, task, texture_ref, None, true)
        else {
            panic!("a two-plane texture binds as CPU bytes");
        };
        assert_eq!((w, h), (4, 2));
        assert_eq!(format.layout(), TexelLayout::Rgba8);
        assert_eq!(format.srgb_source(), None, "codes are linear");
        assert_eq!(bytes[0..4], [200, luma[0], 60, 0xff]);
        assert!(identity.is_some_and(|id| id.generation != 0));
    }

    /// A converted format's refusal is the two-plane one, by name, and the
    /// bind declines rather than sampling a plane.
    #[test]
    fn a_converted_two_plane_texture_declines_by_name() {
        let (mut state, mut host, task, texture_ref) =
            guest_two_plane_texture(500, (4, 2), &[0; 8], &[0; 4]);
        let cap = crate::observe::FailCapture::start();
        assert!(
            resolve_sampled_source(&mut state, &mut host, task, texture_ref, None, true).is_none()
        );
        let lines = cap.lines();
        assert!(
            lines.iter().any(|l| l.contains("draw_sample_two_plane")
                && l.contains("reason=two_plane_range_unknown")),
            "{lines:?}"
        );
    }

    /// `RGB10A8_2P_XR10` binds as `RGBA16Float`, which keeps its extended range;
    /// the guest's own descriptor, whose alpha plane is unrecovered, declines.
    #[test]
    fn an_xr10_texture_binds_as_half_floats_and_the_guests_own_declines() {
        use crate::runtime::two_plane::tests::{guest_type12_texture, CENSUS_XR10_352};
        let g = &CENSUS_XR10_352;
        let color = vec![0u8; 4 * 352 * 128];
        let alpha = vec![0xffu8; 352 * 128];
        let at = 4 * 352 * 128 / 0x4000 * 0x4000;
        let (mut state, mut host, task, texture_ref) = guest_type12_texture(
            &g.with_byte_alpha(),
            g.allocation,
            &[(0, &color), (at as u64, &alpha)],
        );
        let Some((w, h, _, SampledSourceRequest::Bytes(bytes, _, format, _))) =
            resolve_sampled_source(&mut state, &mut host, task, texture_ref, None, true)
        else {
            panic!("an XR10 texture with a byte alpha plane binds");
        };
        assert_eq!((w, h), (352, 128));
        assert_eq!(format.layout(), TexelLayout::Rgba16Float);
        // Colour code 0 in every channel reads -384/510; alpha 255 reads 1.
        let code0 = crate::protocol::two_plane::xr10_code_half(0).to_le_bytes();
        assert_eq!(
            bytes[0..8],
            [code0[0], code0[1], code0[0], code0[1], code0[0], code0[1], 0x00, 0x3c]
        );

        let (mut state, mut host, task, texture_ref) =
            guest_type12_texture(&g.body(), g.allocation, &[]);
        let cap = crate::observe::FailCapture::start();
        assert!(
            resolve_sampled_source(&mut state, &mut host, task, texture_ref, None, true).is_none()
        );
        assert!(cap
            .lines()
            .iter()
            .any(|l| l.contains("reason=two_plane_alpha_layout_unrecovered")));
    }
}
