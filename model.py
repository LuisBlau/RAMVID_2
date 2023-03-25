import datetime
import copy
import functools
import os
import numpy as np
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from utils.nn import update_ema
from utils.resample import LossAwareSampler, UniformSampler
from utils import dist_util, logger
from utils.resample import create_named_schedule_sampler
from utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from utils.fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
INITIAL_LOG_LOSS_SCALE = 20.0

class RAMVID:
        def __init__(self, data, loader_value):

                self.data = data

                # Data Loading parameters
                self.data_dir = loader_value.get('data_loader').get('data_dir')
                self.batch_size = loader_value.get('data_loader').get('batch_size')
                self.image_size = loader_value.get('data_loader').get('image_size')
                self.class_cond = loader_value.get('data_loader').get('class_cond')
                self.deterministic = loader_value.get('data_loader').get('deterministic')
                self.rgb = loader_value.get('data_loader').get('rgb')
                self.seq_len = loader_value.get('data_loader').get('seq_len')

                # Model Parameters
                self.schedule_sampler = loader_value.get('Model').get('schedule_sampler')
                self.lr = float(loader_value.get('Model').get('lr'))
                self.current_lr = self.lr
                self.weight_decay = loader_value.get('Model').get('weight_decay')
                self.lr_anneal_steps = loader_value.get('Model').get('lr_anneal_steps')
                self.microbatch = loader_value.get('Model').get('microbatch')
                self.accumulation_steps = self.batch_size / self.microbatch if self.microbatch > 0 else 1
                self.microbatch = self.microbatch if self.microbatch > 0 else self.batch_size
                self.ema_rate = loader_value.get('Model').get('ema_rate')
                self.ema_rate = (
                        [self.ema_rate]
                        if isinstance(self.ema_rate, float)
                        else [float(x) for x in self.ema_rate.split(",")]
                )
                self.log_interval = loader_value.get('Model').get('log_interval')
                self.save_interval = loader_value.get('Model').get('save_interval')
                self.resume_checkpoint = loader_value.get('Model').get('resume_checkpoint')
                self.use_fp16 = loader_value.get('Model').get('use_fp16')
                self.fp16_scale_growth = loader_value.get('Model').get('fp16_scale_growth')
                self.clip = loader_value.get('Model').get('clip')
                self.seed = loader_value.get('Model').get('seed')
                self.anneal_type = loader_value.get('Model').get('anneal_type')
                self.steps_drop = loader_value.get('Model').get('steps_drop')
                self.drop = loader_value.get('Model').get('drop')
                self.decay = loader_value.get('Model').get('decay')
                self.max_num_mask_frames = loader_value.get('Model').get('max_num_mask_frames')
                self.mask_range = loader_value.get('Model').get('mask_range')
                self.uncondition_rate = loader_value.get('Model').get('uncondition_rate')
                self.exclude_conditional = loader_value.get('Model').get('exclude_conditional')

                self.mask_range = [0, self.seq_len]

                 # if mask_range is None:
                #     mask_range = [0, seq_len]
                # else:
                 #     mask_range = [int(i) for i in mask_range if i != ","]
                #print(self.anneal_type)
                if self.anneal_type == 'linear':
                        assert self.lr_anneal_steps != 0
                        self.lr_anneal_steps = self.lr_anneal_steps
                if self.anneal_type == 'step':
                        assert self.steps_drop != 0
                        assert self.drop != 0
                        self.steps_drop = self.steps_drop
                        self.drop = self.drop
                if self.anneal_type == 'time_based':
                        assert self.decay != 0
                        self.decay = self.decay

                # Diffusion model parameters
                self.num_channels = loader_value.get('Diffusion_model').get('num_channels')
                self.num_res_blocks = loader_value.get('Diffusion_model').get('num_res_blocks')
                self.num_heads = loader_value.get('Diffusion_model').get('num_heads')
                self.num_heads_upsample = loader_value.get('Diffusion_model').get('num_heads_upsample')
                self.attention_resolutions = loader_value.get('Diffusion_model').get('attention_resolutions')
                self.dropout = loader_value.get('Diffusion_model').get('dropout')
                self.learn_sigma = loader_value.get('Diffusion_model').get('learn_sigma')
                self.sigma_small = loader_value.get('Diffusion_model').get('sigma_small')
                self.class_cond = loader_value.get('Diffusion_model').get('class_cond')
                self.diffusion_steps = loader_value.get('Diffusion_model').get('diffusion_steps')
                self.noise_schedule = loader_value.get('Diffusion_model').get('noise_schedule')
                self.timestep_respacing = loader_value.get('Diffusion_model').get('timestep_respacing')
                self.use_kl = loader_value.get('Diffusion_model').get('use_kl')
                self.predict_xstart = loader_value.get('Diffusion_model').get('predict_xstart')
                self.rescale_timesteps = loader_value.get('Diffusion_model').get('rescale_timesteps')
                self.rescale_learned_sigmas = loader_value.get('Diffusion_model').get('rescale_learned_sigmas')
                self.use_checkpoint = loader_value.get('Diffusion_model').get('use_checkpoint')
                self.use_scale_shift_norm = loader_value.get('Diffusion_model').get('use_scale_shift_norm')
                self.scale_time_dim = loader_value.get('Diffusion_model').get('scale_time_dim')


                logger.log("creating model and diffusion...")
                self.model, self.diffusion = create_model_and_diffusion(
                        self.image_size, self.class_cond, self.learn_sigma, self.sigma_small, self.num_channels,
                        self.num_res_blocks,self.scale_time_dim, self.num_heads, self.num_heads_upsample,
                        self.attention_resolutions, self.dropout,self.diffusion_steps, self.noise_schedule,
                        self.timestep_respacing, self.use_kl, self.predict_xstart,self.rescale_timesteps,
                        self.rescale_learned_sigmas, self.use_checkpoint, self.use_scale_shift_norm,self.rgb)


                print(dist_util.dev())
                # print(th.cuda.device_count())
                self.model.to(dist_util.dev())
                self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler, self.diffusion)
                self.schedule_sampler = self.schedule_sampler or UniformSampler(self.diffusion)

                self.step = 0
                self.resume_step = 0
                self.global_batch = self.batch_size# * dist.get_world_size()
                logger.log(f"global batch size = {self.global_batch}")

                self.model_params = list(self.model.parameters())
                self.master_params = self.model_params
                self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
                self.sync_cuda = th.cuda.is_available()

                self._load_and_sync_parameters()
                if self.use_fp16:
                        self._setup_fp16()

                self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
                if self.resume_step:
                        self._load_optimizer_state()
                        # Model was resumed, either due to a restart or a checkpoint
                        # being specified at the command line.
                        self.ema_params = [
                                self._load_ema_parameters(rate) for rate in self.ema_rate
                        ]
                else:
                        self.ema_params = [
                                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
                        ]

                if th.cuda.is_available():
                        logger.log(f"world_size: {dist.get_world_size()}")
                        self.use_ddp = True
                        self.ddp_model = DDP(
                                self.model,
                                device_ids=[dist_util.dev()],
                                output_device=dist_util.dev(),
                                broadcast_buffers=False,
                                bucket_cap_mb=128,
                                find_unused_parameters=False,
                        )
                else:
                        if dist.get_world_size() > 1:
                                logger.warn(
                                        "Distributed training requires CUDA. "
                                        "Gradients will not be synchronized properly!"
                                )
                        self.use_ddp = False
                        self.ddp_model = self.model

        def _load_and_sync_parameters(self):
                self.resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

                if self.resume_checkpoint:
                        self.resume_step = parse_resume_step_from_filename(self.resume_checkpoint)

                        # if dist.get_rank() == 0:
                        logger.log(f"loading model from checkpoint: {self.resume_checkpoint}...")
                        self.model.load_state_dict(
                                dist_util.load_state_dict(
                                        self.resume_checkpoint, map_location=dist_util.dev()
                                )
                        )

                dist_util.sync_params(self.model.parameters())

        def _load_ema_parameters(self, rate):
                ema_params = copy.deepcopy(self.master_params)

                main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
                ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
                if ema_checkpoint:
                        logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                        state_dict = dist_util.load_state_dict(
                                ema_checkpoint, map_location=dist_util.dev()
                        )
                        ema_params = self._state_dict_to_master_params(state_dict)

                dist_util.sync_params(ema_params)
                return ema_params

        def _load_optimizer_state(self):
                main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
                opt_checkpoint = bf.join(
                        bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
                )
                if bf.exists(opt_checkpoint):
                        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
                        state_dict = dist_util.load_opt_state_dict(
                                opt_checkpoint, map_location=dist_util.dev()
                        )

                        self.opt.load_state_dict(state_dict)

        def _setup_fp16(self):
                self.master_params = make_master_params(self.model_params)
                self.model.convert_to_fp16()

        def run(self):
                while (
                        self.current_lr
                ):
                        ####### One
                        batch = next(self.data)
                        #print(batch.shape)
                        self.run_step(batch)
                        if self.step % self.log_interval == 0:
                                logger.dumpkvs()
                        if self.step % self.save_interval == 0:
                                self.save()
                                # Run for a finite amount of time in integration tests. Does access an environment variable
                                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                                        return
                        self.step += 1
                # Save the last checkpoint if it wasn't already saved.
                if (self.step - 1) % self.save_interval != 0:
                        self.save()

        def run_step(self, batch):
                self.forward_backward(batch)
                if self.clip:
                        th.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), self.clip)
                if self.use_fp16:
                        self.optimize_fp16()
                else:
                        self.optimize_normal()
                self.log_step()

        def forward_backward(self, batch):
                zero_grad(self.model_params)
                ########## TIME (Per Batch) and Loss ###########
                ss = datetime.datetime.now()
                for i in range(0, batch.shape[0], self.microbatch):
                        micro = batch[i: i + self.microbatch].to(dist_util.dev())
                        last_batch = (i + self.microbatch) >= batch.shape[0]
                        t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
                        compute_losses = functools.partial(
                                self.diffusion.training_losses,
                                self.ddp_model,
                                micro,
                                t,
                                max_num_mask_frames=self.max_num_mask_frames,
                                mask_range=self.mask_range,
                                uncondition_rate=self.uncondition_rate,
                                exclude_conditional=self.exclude_conditional,
                        )
                        if last_batch or not self.use_ddp:
                                losses = compute_losses()
                        else:
                                with self.ddp_model.no_sync():
                                        losses = compute_losses()
                        if isinstance(self.schedule_sampler, LossAwareSampler):
                                self.schedule_sampler.update_with_local_losses(
                                        t, losses["loss"].detach()
                                )
                        loss = (losses["loss"] * weights).mean()
                        log_loss_dict(
                                self.diffusion, t, {k: v * weights for k, v in losses.items()}
                        )
                        loss = loss / self.accumulation_steps
                        if self.use_fp16:
                                loss_scale = 2 ** self.lg_loss_scale
                                (loss * loss_scale).backward()
                        else:
                                loss.backward()
                ee = datetime.datetime.now()
                print(f"===Per Batch Time===={ee-ss}=======Loss======{loss}")

        def optimize_fp16(self):
                if any(not th.isfinite(p.grad).all() for p in self.model_params):
                        self.lg_loss_scale -= 1
                        logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
                        return

                model_grads_to_master_grads(self.model_params, self.master_params)
                self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
                self._log_grad_norm()
                self._anneal_lr()
                self.opt.step()
                for rate, params in zip(self.ema_rate, self.ema_params):
                        update_ema(params, self.master_params, rate=rate)
                master_params_to_model_params(self.model_params, self.master_params)
                self.lg_loss_scale += self.fp16_scale_growth

        def optimize_normal(self):
                self._log_grad_norm()
                self._anneal_lr()
                self.opt.step()
                for rate, params in zip(self.ema_rate, self.ema_params):
                        update_ema(params, self.master_params, rate=rate)

        def _log_grad_norm(self):
                sqsum = 0.0
                for p in self.master_params:
                        sqsum += (p.grad ** 2).sum().item()
                logger.logkv_mean("grad_norm", np.sqrt(sqsum))

        def _anneal_lr(self):
                if self.anneal_type is None:
                        return
                if self.anneal_type == "linear":
                        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
                        self.lr = self.lr * (1 - frac_done)
                elif self.anneal_type == "step":
                        self.lr = self.lr * self.drop ** (np.floor((self.step + self.resume_step) / self.steps_drop))
                elif self.anneal_type == "time_based":
                        self.lr = self.lr / (1 + self.decay * (self.step + self.resume_step))
                else:
                        raise ValueError(f"unsupported anneal type: {self.anneal_type}")
                self.lr = self.lr
                self.current_lr = self.lr

        def log_step(self):
                logger.logkv("step", self.step + self.resume_step)
                logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
                if self.use_fp16:
                        logger.logkv("lg_loss_scale", self.lg_loss_scale)

        def save(self):
                def save_checkpoint(rate, params):
                        state_dict = self._master_params_to_state_dict(params)
                        if dist.get_rank() == 0:
                                logger.log(f"saving model {rate}...")
                                if not rate:
                                        filename = f"model{(self.step + self.resume_step):06d}.pt"
                                else:
                                        filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
                                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                                        th.save(state_dict, f)

                save_checkpoint(0, self.master_params)
                for rate, params in zip(self.ema_rate, self.ema_params):
                        save_checkpoint(rate, params)

                if dist.get_rank() == 0:
                        with bf.BlobFile(
                                bf.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"),
                                "wb",
                        ) as f:
                                th.save(self.opt.state_dict(), f)

                dist.barrier()

        def _master_params_to_state_dict(self, master_params):
                if self.use_fp16:
                        master_params = unflatten_master_params(
                                list(self.model.parameters()), master_params
                        )
                state_dict = self.model.state_dict()
                for i, (name, _value) in enumerate(self.model.named_parameters()):
                        assert name in state_dict
                        state_dict[name] = master_params[i]
                return state_dict

        def _state_dict_to_master_params(self, state_dict):
                params = [state_dict[name] for name, _ in self.model.named_parameters()]
                if self.use_fp16:
                        return make_master_params(params)
                else:
                        return params

def parse_resume_step_from_filename(filename):
        """
        Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
        checkpoint's number of steps.
        """
        split = filename.split("model")
        if len(split) < 2:
                return 0
        split1 = split[-1].split(".")[0]
        try:
                return int(split1)
        except ValueError:
                return 0

def get_blob_logdir():
        return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())

def find_resume_checkpoint():
        # On your infrastructure, you may want to override this to automatically
        # discover the latest checkpoint on your blob storage, etc.
        return None

def find_ema_checkpoint(main_checkpoint, step, rate):
        if main_checkpoint is None:
                return None
        filename = f"ema_{rate}_{(step):06d}.pt"
        path = bf.join(bf.dirname(main_checkpoint), filename)
        if bf.exists(path):
                return path
        return None

def log_loss_dict(diffusion, ts, losses):
        for key, values in losses.items():
                logger.logkv_mean(key, values.mean().item())
                # Log the quantiles (four quartiles, in particular).
                for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                        quartile = int(4 * sub_t / diffusion.num_timesteps)
                        logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


