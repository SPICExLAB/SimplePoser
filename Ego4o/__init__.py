from Ego4o.vqvae import VQLimbHML
from Ego4o.encoder import IMUTransformerEncoder

MODEL_REGISTRY = {
    'vqvae_limb_hml': VQLimbHML,
    'imu_encoder': IMUTransformerEncoder,
}
