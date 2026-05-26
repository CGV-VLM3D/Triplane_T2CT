from .triplane_ae import TriplaneAE
from .triplane_ae_compress3d import TriplaneAECompress3D
from .triplane_decoder import TriplaneDecoder
from .triplane_decoder_compress3d import TriplaneDecoderCompress3D
from .triplane_decoder_d import TriplaneDecoderD
from .triplane_encoder import TriplaneEncoder
from .triplane_encoder_compress3d import TriplaneEncoderCompress3D
from .deprecated.trivae_d3t import TriVAE_D3T
from .trivae_conv import TriVQAEConv

MODEL_REGISTRY = {
    "TriplaneAE": TriplaneAE,
    "TriplaneAECompress3D": TriplaneAECompress3D,
    "TriVAE_D3T": TriVAE_D3T,
    "TriVQAEConv": TriVQAEConv,
}

__all__ = [
    "TriplaneEncoder",
    "TriplaneEncoderCompress3D",
    "TriplaneDecoder",
    "TriplaneDecoderD",
    "TriplaneDecoderCompress3D",
    "TriplaneAE",
    "TriplaneAECompress3D",
    "TriVAE_D3T",
    "TriVQAEConv",
    "MODEL_REGISTRY",
]
