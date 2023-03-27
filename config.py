import yaml
from typing import List, Optional
from pydantic import BaseModel, StrictStr

class DataLoaderConfig(BaseModel):
    data_dir: StrictStr
    batch_size: int
    image_size: int
    num_workers: int
    rgb: bool
    seq_len: int

class ModelConfig(BaseModel):
    schedule_sampler: StrictStr
    lr: float
    weight_decay: float
    lr_anneal_steps: int
    microbatch: float
    ema_rate: StrictStr
    log_interval: int
    save_interval: int
    resume_checkpoint: StrictStr
    use_fp16: bool
    fp16_scale_growth: float
    clip: int
    seed: int
    anneal_type: StrictStr
    steps_drop: float
    drop: float
    decay: float
    max_num_mask_frames: int
    mask_range: str
    uncondition_rate: float
    exclude_conditional: bool
    INITIAL_LOG_LOSS_SCALE: float

class DiffusionModelConfig(BaseModel):
    num_channels: int
    num_res_blocks: int
    num_heads: int
    num_heads_upsample: int
    attention_resolutions: StrictStr
    dropout: float
    learn_sigma: bool
    sigma_small: bool
    class_cond: bool
    diffusion_steps: int
    noise_schedule: StrictStr
    timestep_respacing: StrictStr
    use_kl: bool
    predict_xstart: bool
    rescale_timesteps: bool
    rescale_learned_sigmas: bool
    use_checkpoint: bool
    use_scale_shift_norm: bool
    scale_time_dim: int
def read_yaml(file_path: str) -> dict:
    with open(file_path, 'r') as stream:
        config = yaml.safe_load(stream)

    return DataLoaderConfig(**config), ModelConfig(**config), DiffusionModelConfig(**config)
# config = read_yaml('model_config.yml')
# print(config)