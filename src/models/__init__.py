from .triplane_ae import TriplaneAE
from .triplane_decoder import TriplaneDecoder
from .triplane_encoder import TriplaneEncoder
from .trivae_d3t import TriVAE_D3T
from .trivae_conv import TriVQAEConv

MODEL_REGISTRY = {
    "TriplaneAE": TriplaneAE,
    "TriVAE_D3T": TriVAE_D3T,
    "TriVQAEConv": TriVQAEConv,
}

__all__ = [
    "TriplaneEncoder",
    "TriplaneDecoder",
    "TriplaneAE",
    "TriVAE_D3T",
    "TriVQAEConv",
    "MODEL_REGISTRY",
]
