from typing import Tuple
import torch
from diffusers import AutoencoderKL
from einops import rearrange
from torch import Tensor


from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.autoencoders.video_autoencoder import (
    Downsample3D,
    VideoAutoencoder,
)

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


def vae_encode(
    media_items: Tensor,
    vae: AutoencoderKL,
    split_size: int = 1,
    vae_per_channel_normalize=False,
    tile_size: int = 1024,
    tile_stride: int = 512,
    batch_size: int = 1,
) -> Tensor:
    """
    Encodes media items (images or videos) into latent representations using a specified VAE model.
    The function supports processing batches of images or video frames and can handle the processing
    in smaller sub-batches if needed.

    Args:
        media_items (Tensor): A torch Tensor containing the media items to encode. The expected
            shape is (batch_size, channels, height, width) for images or (batch_size, channels,
            frames, height, width) for videos.
        vae (AutoencoderKL): An instance of the `AutoencoderKL` class from the `diffusers` library,
            pre-configured and loaded with the appropriate model weights.
        split_size (int, optional): The number of sub-batches to split the input batch into for encoding.
            If set to more than 1, the input media items are processed in smaller batches according to
            this value. Defaults to 1, which processes all items in a single batch.
        vae_per_channel_normalize (bool, optional): Whether to normalize per channel. Defaults to False.
        tile_size (int, optional): Size of tiles for tiled processing. Defaults to 1024.
        tile_stride (int, optional): Stride between tiles. Defaults to 512.
        batch_size (int, optional): Batch size for processing tiles. Defaults to 1.

    Returns:
        Tensor: A torch Tensor of the encoded latent representations. The shape of the tensor is adjusted
            to match the input shape, scaled by the model's configuration.
    """
    is_video_shaped = media_items.dim() == 5
    _, channels = media_items.shape[0:2]

    if channels != 3:
        raise ValueError(f"Expects tensors with 3 channels, got {channels}.")

    if is_video_shaped and not isinstance(
        vae, (VideoAutoencoder, CausalVideoAutoencoder)
    ):
        media_items = rearrange(media_items, "b c n h w -> (b n) c h w")

    # Process in tiles
    if tile_size > 64:  # Only tile if tile size is significant
        latents = process_in_tiles(
            media_items,
            vae.encode,
            tile_size,
            tile_stride,
            batch_size,
            lambda x: x.latent_dist.sample(),
        )
    else:
        if split_size > 1:
            if len(media_items) % split_size != 0:
                raise ValueError(
                    "Error: The batch size must be divisible by split_size"
                )
            encode_bs = len(media_items) // split_size
            latents = []
            if media_items.device.type == "xla":
                xm.mark_step()
            for image_batch in media_items.split(encode_bs):
                latents.append(vae.encode(image_batch).latent_dist.sample())
                if media_items.device.type == "xla":
                    xm.mark_step()
            latents = torch.cat(latents, dim=0)
        else:
            latents = vae.encode(media_items).latent_dist.sample()

    latents = normalize_latents(latents, vae, vae_per_channel_normalize)
    if is_video_shaped and not isinstance(
        vae, (VideoAutoencoder, CausalVideoAutoencoder)
    ):
        original_batch_size = media_items.shape[0] // media_items.shape[2]
        latents = rearrange(latents, "(b n) c h w -> b c n h w", b=original_batch_size)
    return latents


def vae_decode(
    latents: Tensor,
    vae: AutoencoderKL,
    is_video: bool = True,
    split_size: int = 1,
    vae_per_channel_normalize=False,
    timestep=None,
    tile_size: int = 256,
    tile_stride: int = 128,
    batch_size: int = 1,
) -> Tensor:
    """
    Decodes latent representations back to media items using a specified VAE model.
    
    Args:
        latents (Tensor): Latent representations to decode
        vae (AutoencoderKL): VAE model to use for decoding
        is_video (bool): Whether the input represents video data
        split_size (int): Number of splits for batch processing
        vae_per_channel_normalize (bool): Whether to use per-channel normalization
        timestep (Optional[Tensor]): Timestep for conditional decoding
        tile_size (int): Size of tiles for tiled processing
        tile_stride (int): Stride between tiles
        batch_size (int): Batch size for processing tiles
        
    Returns:
        Tensor: Decoded media items
    """
    is_video_shaped = latents.dim() == 5
    original_batch_size = latents.shape[0]

    if is_video_shaped and not isinstance(
        vae, (VideoAutoencoder, CausalVideoAutoencoder)
    ):
        latents = rearrange(latents, "b c n h w -> (b n) c h w")

    # Process in tiles
    if tile_size > 64:  # Only tile if tile size is significant
        decode_func = lambda x: _run_decoder(
            x, vae, is_video, vae_per_channel_normalize, timestep
        )
        images = process_in_tiles(latents, decode_func, tile_size, tile_stride, batch_size)
    else:
        if split_size > 1:
            if len(latents) % split_size != 0:
                raise ValueError(
                    "Error: The batch size must be divisible by split_size"
                )
            encode_bs = len(latents) // split_size
            image_batch = [
                _run_decoder(
                    latent_batch, vae, is_video, vae_per_channel_normalize, timestep
                )
                for latent_batch in latents.split(encode_bs)
            ]
            images = torch.cat(image_batch, dim=0)
        else:
            images = _run_decoder(
                latents, vae, is_video, vae_per_channel_normalize, timestep
            )

    if is_video_shaped and not isinstance(
        vae, (VideoAutoencoder, CausalVideoAutoencoder)
    ):
        images = rearrange(images, "(b n) c h w -> b c n h w", b=original_batch_size)
    return images


def process_in_tiles(
    x: torch.Tensor,
    process_fn,
    tile_size: int,
    tile_stride: int,
    batch_size: int,
    post_process_fn=None,
) -> torch.Tensor:
    """
    Process a tensor in tiles to handle large inputs efficiently.
    
    Args:
        x (Tensor): Input tensor to process
        process_fn (callable): Function to process each tile
        tile_size (int): Size of each tile
        tile_stride (int): Stride between tiles
        batch_size (int): Number of tiles to process at once
        post_process_fn (callable, optional): Function to apply after processing
        
    Returns:
        Tensor: Processed output
    """
    B, C, *spatial_dims = x.shape
    device = x.device
    dtype = x.dtype

    # Calculate output shape
    if len(spatial_dims) == 2:
        H, W = spatial_dims
        D = 1
    else:
        D, H, W = spatial_dims

    # Initialize output tensor
    result = torch.zeros_like(x)
    count = torch.zeros_like(x)

    # Process tiles
    for h_start in range(0, H, tile_stride):
        for w_start in range(0, W, tile_stride):
            h_end = min(h_start + tile_size, H)
            w_end = min(w_start + tile_size, W)

            # Extract tile
            if len(spatial_dims) == 2:
                tile = x[:, :, h_start:h_end, w_start:w_end]
            else:
                tile = x[:, :, :, h_start:h_end, w_start:w_end]

            # Process tile
            processed = process_fn(tile)
            if post_process_fn is not None:
                processed = post_process_fn(processed)

            # Add processed tile back
            if len(spatial_dims) == 2:
                result[:, :, h_start:h_end, w_start:w_end] += processed
                count[:, :, h_start:h_end, w_start:w_end] += 1
            else:
                result[:, :, :, h_start:h_end, w_start:w_end] += processed
                count[:, :, :, h_start:h_end, w_start:w_end] += 1

    # Average overlapping regions
    result = result / count.clamp(min=1)
    return result


def _run_decoder(
    latents: Tensor,
    vae: AutoencoderKL,
    is_video: bool,
    vae_per_channel_normalize=False,
    timestep=None,
) -> Tensor:
    if isinstance(vae, (VideoAutoencoder, CausalVideoAutoencoder)):
        *_, fl, hl, wl = latents.shape
        temporal_scale, spatial_scale, _ = get_vae_size_scale_factor(vae)
        latents = latents.to(vae.dtype)
        vae_decode_kwargs = {}
        if timestep is not None:
            vae_decode_kwargs["timestep"] = timestep
        image = vae.decode(
            un_normalize_latents(latents, vae, vae_per_channel_normalize),
            return_dict=False,
            target_shape=(
                1,
                3,
                fl * temporal_scale if is_video else 1,
                hl * spatial_scale,
                wl * spatial_scale,
            ),
            **vae_decode_kwargs,
        )[0]
    else:
        image = vae.decode(
            un_normalize_latents(latents, vae, vae_per_channel_normalize),
            return_dict=False,
        )[0]
    return image


def get_vae_size_scale_factor(vae: AutoencoderKL) -> float:
    if isinstance(vae, CausalVideoAutoencoder):
        spatial = vae.spatial_downscale_factor
        temporal = vae.temporal_downscale_factor
    else:
        down_blocks = len(
            [
                block
                for block in vae.encoder.down_blocks
                if isinstance(block.downsample, Downsample3D)
            ]
        )
        spatial = vae.config.patch_size * 2**down_blocks
        temporal = (
            vae.config.patch_size_t * 2**down_blocks
            if isinstance(vae, VideoAutoencoder)
            else 1
        )

    return (temporal, spatial, spatial)


def normalize_latents(
    latents: Tensor, vae: AutoencoderKL, vae_per_channel_normalize: bool = False
) -> Tensor:
    return (
        (latents - vae.mean_of_means.to(latents.dtype).view(1, -1, 1, 1, 1))
        / vae.std_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
        if vae_per_channel_normalize
        else latents * vae.config.scaling_factor
    )


def un_normalize_latents(
    latents: Tensor, vae: AutoencoderKL, vae_per_channel_normalize: bool = False
) -> Tensor:
    return (
        latents * vae.std_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
        + vae.mean_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
        if vae_per_channel_normalize
        else latents / vae.config.scaling_factor
    )