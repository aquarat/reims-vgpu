//! The compute stager's bind of a two-plane colour texture.
//!
//! A compute function samples or reads a two-plane texture the way a fragment
//! function does: as the texels Metal reads, one image. So this stages the
//! conversion [`crate::runtime::two_plane`] produces, in the single-plane format
//! that holds it ([`TwoPlaneConversion::output_pixel_format`]), and never a plane. A
//! storage binding refuses: the image is a converted copy, and a write to it
//! would reach neither plane.
//!
//! [`TwoPlaneConversion::output_pixel_format`]: crate::protocol::two_plane::TwoPlaneConversion::output_pixel_format

use super::{ComputeStatus, RailStage, StagedTexture, TextureWriteback};
use crate::model::DeviceState;
use crate::runtime::decode::resource::Descriptor;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::objects;
use crate::runtime::two_plane::{self, TwoPlaneBindRefusal, TwoPlaneTexels};

/// The one compute status every two-plane refusal reports under. The specific
/// reason is the `two_plane_*` line [`refuse`] writes; the status is the
/// stager's, and spelling a `two_plane_*` slug here too would have two types
/// claiming one slug.
const STATUS_SLUG: &str = "compute_stage_two_plane";

fn refuse(task_id: u32, texture_ref: u32, refusal: TwoPlaneBindRefusal) -> ComputeStatus {
    crate::observe::Emit::decline("compute_stage_two_plane", &refusal)
        .field("task", task_id)
        .field("ref", texture_ref)
        .fail_once((u64::from(task_id) << 32) | u64::from(texture_ref));
    ComputeStatus::Unsupported(STATUS_SLUG)
}

fn staged<R: RailStage>(
    binding: u32,
    texture_ref: u32,
    texels: TwoPlaneTexels,
) -> StagedTexture<R> {
    StagedTexture {
        binding,
        pixel_format: texels.conversion.output_pixel_format(),
        storage_selector: None,
        width: texels.width,
        height: texels.height,
        mip_levels: 1,
        bytes: Vec::clone(&texels.bytes),
        is_storage: false,
        writeback: TextureWriteback::None,
        rail: R::stage(texture_ref, None, None),
    }
}

/// Stage a type-12 object, whose planes are in guest memory.
pub(super) fn stage_guest<R: RailStage, M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    binding: u32,
    is_storage: bool,
) -> Result<StagedTexture<R>, ComputeStatus> {
    let resource = objects::resolve_resource(state, host, task_id, texture_ref)
        .map_err(|_| refuse(task_id, texture_ref, TwoPlaneBindRefusal::Undecoded))?;
    let Ok(Descriptor::DualPlaneTexture(dual)) = resource.decoded() else {
        return Err(refuse(task_id, texture_ref, TwoPlaneBindRefusal::Undecoded));
    };
    if is_storage {
        return Err(refuse(
            task_id,
            texture_ref,
            TwoPlaneBindRefusal::StorageBind {
                pixel_format: dual.pixel_format(),
            },
        ));
    }
    let texels = two_plane::guest_texels(
        state,
        host,
        task_id,
        texture_ref,
        dual,
        crate::runtime::render_writeback::SettleSite::ComputeStageTexture,
    )
    .map_err(|refusal| refuse(task_id, texture_ref, refusal))?;
    crate::runtime::drain::note_store_route("compute_stage_two_plane_guest");
    Ok(staged(binding, texture_ref, texels))
}

/// What the surface rail knows of a texture of a two-plane format over an
/// IOSurface mapping.
pub(super) struct SurfaceTexture {
    pub mapping_id: u32,
    pub width: u32,
    pub height: u32,
    pub pixel_format: u16,
    /// The format a texture view over it reads as, when one does.
    pub view_pixel_format: Option<u16>,
}

/// Stage a texture of a two-plane format over an IOSurface mapping.
pub(super) fn stage_surface<R: RailStage, M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    binding: u32,
    is_storage: bool,
    texture: SurfaceTexture,
) -> Result<StagedTexture<R>, ComputeStatus> {
    let pixel_format = texture.pixel_format;
    if let Some(view) = texture
        .view_pixel_format
        .filter(|&view| view != pixel_format)
    {
        return Err(refuse(
            task_id,
            texture_ref,
            TwoPlaneBindRefusal::ViewFormat { pixel_format, view },
        ));
    }
    if is_storage {
        return Err(refuse(
            task_id,
            texture_ref,
            TwoPlaneBindRefusal::StorageBind { pixel_format },
        ));
    }
    let planes = state
        .mappings
        .get(&texture.mapping_id)
        .ok_or(ComputeStatus::MissingTexture(
            "compute_stage_tex_mapping_gone",
        ))?;
    let planes = two_plane::surface_planes(planes, pixel_format, texture.width, texture.height)
        .map_err(|refusal| refuse(task_id, texture_ref, refusal))?;
    let texels = two_plane::load_surface(state, host, task_id, texture.mapping_id, &planes)
        .map_err(|refusal| refuse(task_id, texture_ref, refusal))?;
    crate::runtime::drain::note_store_route("compute_stage_two_plane_surface");
    Ok(staged(binding, texture_ref, texels))
}

#[cfg(test)]
mod tests {
    use super::super::tests::NeutralStage;
    use super::super::{stage_texture_raw, ComputeStatus};
    use crate::protocol::pixel_format::MTL_FORMAT_RGBA8_UNORM;
    use crate::runtime::two_plane::tests::guest_two_plane_texture;

    #[test]
    fn compute_stages_a_two_plane_texture_converted_and_refuses_storage() {
        let luma: Vec<u8> = (0..8u8).map(|i| 30 + 20 * i).collect();
        let chroma = [60u8, 200, 70, 190];
        let (mut state, mut host, task, texture_ref) =
            guest_two_plane_texture(540, (4, 2), &luma, &chroma);
        let staged = stage_texture_raw::<NeutralStage, _>(
            &mut state,
            &mut host,
            task,
            texture_ref,
            3,
            false,
        )
        .expect("a sampled two-plane texture stages");
        assert_eq!(staged.pixel_format, MTL_FORMAT_RGBA8_UNORM);
        assert_eq!((staged.width, staged.height, staged.mip_levels), (4, 2, 1));
        assert_eq!(staged.bytes[0..4], [200, luma[0], 60, 0xff]);

        let cap = crate::observe::FailCapture::start();
        let refused =
            stage_texture_raw::<NeutralStage, _>(&mut state, &mut host, task, texture_ref, 3, true)
                .err();
        assert_eq!(
            refused,
            Some(ComputeStatus::Unsupported("compute_stage_two_plane"))
        );
        assert!(cap
            .lines()
            .iter()
            .any(|l| l.contains("reason=two_plane_storage_bind")));
    }
}
