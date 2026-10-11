import math

import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
import transformers

from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from vocos.feature_extractors import FeatureExtractor
from vocos.heads import FourierHead
from vocos.helpers import plot_spectrogram_to_numpy
from vocos.loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss, MelSpecReconstructionLoss
from vocos.models import Backbone
from vocos.modules import safe_log


class VocosExp(pl.LightningModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
        sample_rate: int,
        initial_learning_rate: float,
        num_warmup_steps: int = 0,
        mel_loss_coeff: float = 45,
        mrd_loss_coeff: float = 1.0,
        pretrain_mel_steps: int = 0,
        decay_mel_coeff: bool = False,
        evaluate_utmos: bool = False,
        evaluate_pesq: bool = False,
        evaluate_mrstft: bool = False,
        evaluate_periodicty: bool = False,
        validation_metric_batches: int = 0,
        validation_metrics_affect_val_loss: bool = True,
        generator_learning_rate: float | None = None,
        discriminator_learning_rate: float | None = None,
    ):
        """
        Args:
            feature_extractor (FeatureExtractor): An instance of FeatureExtractor to extract features from audio signals.
            backbone (Backbone): An instance of Backbone model.
            head (FourierHead):  An instance of Fourier head to generate spectral coefficients and reconstruct a waveform.
            sample_rate (int): Sampling rate of the audio signals.
            initial_learning_rate (float): Initial learning rate for the optimizer.
            num_warmup_steps (int): Number of steps for the warmup phase of learning rate scheduler. Default is 0.
            mel_loss_coeff (float, optional): Coefficient for Mel-spectrogram loss in the loss function. Default is 45.
            mrd_loss_coeff (float, optional): Coefficient for Multi Resolution Discriminator loss. Default is 1.0.
            pretrain_mel_steps (int, optional): Number of steps to pre-train the model without the GAN objective. Default is 0.
            decay_mel_coeff (bool, optional): If True, the Mel-spectrogram loss coefficient is decayed during training. Default is False.
            evaluate_utmos (bool, optional): If True, UTMOS scores are computed for each validation run.
            evaluate_pesq (bool, optional): If True, PESQ scores are computed for each validation run.
            evaluate_mrstft (bool, optional): If True, multi-resolution STFT loss is computed for validation.
            evaluate_periodicty (bool, optional): If True, periodicity scores are computed for each validation run.
            validation_metric_batches (int, optional): Number of deterministic validation batches used for
                expensive metrics. Zero evaluates all validation batches.
            validation_metrics_affect_val_loss (bool, optional): Retain the legacy composite validation
                objective when true. Set false for new runs so model selection remains based on mel error.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["feature_extractor", "backbone", "head"])

        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head

        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiresddisc = MultiResolutionDiscriminator()

        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(sample_rate=sample_rate)
        self.mrstft_loss = None

        if validation_metric_batches < 0:
            raise ValueError("validation_metric_batches must be non-negative")

        self.train_discriminator = False
        self.base_mel_coeff = self.mel_loss_coeff = mel_loss_coeff

    def _log_audio(self, key: str, audio: torch.Tensor) -> None:
        """Log audio through either TensorBoard or Weights & Biases."""
        audio_array = audio.detach().cpu().numpy()
        experiment = self.logger.experiment
        if hasattr(experiment, "add_audio"):
            experiment.add_audio(key, audio_array, self.global_step, self.hparams.sample_rate)
            return

        import wandb

        experiment.log({key: wandb.Audio(audio_array, sample_rate=self.hparams.sample_rate)}, step=self.global_step)

    def _log_image(self, key: str, image: np.ndarray) -> None:
        """Log an HWC image through either TensorBoard or Weights & Biases."""
        experiment = self.logger.experiment
        if hasattr(experiment, "add_image"):
            experiment.add_image(key, image, self.global_step, dataformats="HWC")
            return

        import wandb

        experiment.log({key: wandb.Image(image)}, step=self.global_step)

    def configure_optimizers(self):
        disc_params = [
            {"params": self.multiperioddisc.parameters()},
            {"params": self.multiresddisc.parameters()},
        ]
        gen_params = [
            {"params": self.feature_extractor.parameters()},
            {"params": self.backbone.parameters()},
            {"params": self.head.parameters()},
        ]

        generator_learning_rate = (
            self.hparams.initial_learning_rate
            if self.hparams.generator_learning_rate is None
            else self.hparams.generator_learning_rate
        )
        discriminator_learning_rate = (
            self.hparams.initial_learning_rate
            if self.hparams.discriminator_learning_rate is None
            else self.hparams.discriminator_learning_rate
        )
        opt_disc = torch.optim.AdamW(disc_params, lr=discriminator_learning_rate, betas=(0.8, 0.9))
        opt_gen = torch.optim.AdamW(gen_params, lr=generator_learning_rate, betas=(0.8, 0.9))

        max_steps = self.trainer.max_steps // 2  # Max steps per optimizer
        scheduler_disc = transformers.get_cosine_schedule_with_warmup(
            opt_disc, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        )
        scheduler_gen = transformers.get_cosine_schedule_with_warmup(
            opt_gen, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        )

        return (
            [opt_disc, opt_gen],
            [{"scheduler": scheduler_disc, "interval": "step"}, {"scheduler": scheduler_gen, "interval": "step"}],
        )

    @staticmethod
    def _split_batch(batch):
        if isinstance(batch, dict):
            return batch["audio"], batch.get("features"), batch.get("speaker_embedding"), batch.get("speaker_matrix")
        return batch, None, None, None

    def forward(self, audio_input, precomputed_features=None, speaker_embedding=None, speaker_matrix=None, **kwargs):
        feature_input = audio_input if precomputed_features is None else precomputed_features
        features = self.feature_extractor(
            feature_input,
            sample_rate=self.hparams.sample_rate,
            speaker_matrix=speaker_matrix,
            **kwargs,
        )
        x = self.backbone(
            features, speaker_embedding=speaker_embedding, speaker_matrix=speaker_matrix, **kwargs
        )
        audio_output = self.head(x)
        return audio_output

    @torch.inference_mode()
    def decode_stream(
        self,
        features_input: torch.Tensor,
        state: dict | None = None,
        speaker_embedding: torch.Tensor | None = None,
        speaker_matrix: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """Decode one chronological feature chunk while retaining causal state.

        ``features_input`` has the same shape and meaning as
        ``precomputed_features`` passed to :meth:`forward`.  Reuse the returned
        state for the next chunk of the *same* utterance/stream.  Concatenating
        all returned audio chunks is equivalent to decoding the complete
        feature sequence at once, subject only to floating-point tolerance.
        """
        if not hasattr(self.backbone, "forward_stream") or not hasattr(self.head, "forward_stream"):
            raise TypeError("decode_stream requires CausalVocosBackbone and CausalISTFTHead.")
        state = {} if state is None else state
        features = self.feature_extractor(
            features_input,
            sample_rate=self.hparams.sample_rate,
            speaker_matrix=speaker_matrix,
            **kwargs,
        )
        x, backbone_state = self.backbone.forward_stream(
            features,
            state.get("backbone"),
            speaker_embedding=speaker_embedding,
            speaker_matrix=speaker_matrix,
            **kwargs,
        )
        audio_output, head_state = self.head.forward_stream(x, state.get("head"))
        return audio_output, {"backbone": backbone_state, "head": head_state}

    def training_step(self, batch, batch_idx, optimizer_idx, **kwargs):
        audio_input, precomputed_features, speaker_embedding, speaker_matrix = self._split_batch(batch)

        # train discriminator
        if optimizer_idx == 0 and self.train_discriminator:
            with torch.no_grad():
                audio_hat = self(
                    audio_input, precomputed_features=precomputed_features, speaker_embedding=speaker_embedding,
                    speaker_matrix=speaker_matrix, **kwargs
                )

            real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(y=audio_input, y_hat=audio_hat, **kwargs,)
            real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(y=audio_input, y_hat=audio_hat, **kwargs,)
            loss_mp, loss_mp_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp
            )
            loss_mrd, loss_mrd_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
            )
            loss_mp /= len(loss_mp_real)
            loss_mrd /= len(loss_mrd_real)
            loss = loss_mp + self.hparams.mrd_loss_coeff * loss_mrd

            self.log("discriminator/total", loss, prog_bar=True)
            self.log("discriminator/multi_period_loss", loss_mp)
            self.log("discriminator/multi_res_loss", loss_mrd)
            return loss

        # train generator
        if optimizer_idx == 1:
            audio_hat = self(
                audio_input, precomputed_features=precomputed_features, speaker_embedding=speaker_embedding,
                speaker_matrix=speaker_matrix, **kwargs
            )
            if self.train_discriminator:
                _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(
                    y=audio_input, y_hat=audio_hat, **kwargs,
                )
                _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(
                    y=audio_input, y_hat=audio_hat, **kwargs,
                )
                loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
                loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
                loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
                loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
                loss_fm_mp = self.feat_matching_loss(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp) / len(fmap_rs_mp)
                loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)

                self.log("generator/multi_period_loss", loss_gen_mp)
                self.log("generator/multi_res_loss", loss_gen_mrd)
                self.log("generator/feature_matching_mp", loss_fm_mp)
                self.log("generator/feature_matching_mrd", loss_fm_mrd)
            else:
                loss_gen_mp = loss_gen_mrd = loss_fm_mp = loss_fm_mrd = 0

            mel_loss = self.melspec_loss(audio_hat, audio_input)
            loss = (
                loss_gen_mp
                + self.hparams.mrd_loss_coeff * loss_gen_mrd
                + loss_fm_mp
                + self.hparams.mrd_loss_coeff * loss_fm_mrd
                + self.mel_loss_coeff * mel_loss
            )

            self.log("generator/total_loss", loss, prog_bar=True)
            self.log("mel_loss_coeff", self.mel_loss_coeff)
            self.log("generator/mel_loss", mel_loss)

            if self.global_step % 1000 == 0 and self.global_rank == 0:
                self._log_audio("train/audio_in", audio_input[0])
                self._log_audio("train/audio_pred", audio_hat[0])
                with torch.no_grad():
                    mel = safe_log(self.melspec_loss.mel_spec(audio_input[0]))
                    mel_hat = safe_log(self.melspec_loss.mel_spec(audio_hat[0]))
                self._log_image("train/mel_target", plot_spectrogram_to_numpy(mel.data.cpu().numpy()))
                self._log_image("train/mel_pred", plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()))

            return loss

    def on_validation_epoch_start(self):
        if self.hparams.evaluate_utmos:
            from metrics.UTMOS import UTMOSScore

            if not hasattr(self, "utmos_model"):
                self.utmos_model = UTMOSScore(device=self.device)
        if self.hparams.evaluate_mrstft and self.mrstft_loss is None:
            import auraloss

            self.mrstft_loss = auraloss.freq.MultiResolutionSTFTLoss().to(self.device)

    def _evaluate_expensive_validation_metrics(self, batch_idx: int) -> bool:
        limit = self.hparams.validation_metric_batches
        return limit == 0 or batch_idx < limit

    @staticmethod
    def _average_available(outputs, key: str):
        values = [output[key] for output in outputs if output[key] is not None]
        return None if not values else torch.stack(values).mean()

    def validation_step(self, batch, batch_idx, **kwargs):
        audio_input, precomputed_features, speaker_embedding, speaker_matrix = self._split_batch(batch)
        audio_hat = self(
            audio_input, precomputed_features=precomputed_features, speaker_embedding=speaker_embedding,
            speaker_matrix=speaker_matrix, **kwargs
        )

        audio_16_khz = torchaudio.functional.resample(audio_input, orig_freq=self.hparams.sample_rate, new_freq=16000)
        audio_hat_16khz = torchaudio.functional.resample(audio_hat, orig_freq=self.hparams.sample_rate, new_freq=16000)

        evaluate_metrics = self._evaluate_expensive_validation_metrics(batch_idx)

        if evaluate_metrics and self.hparams.evaluate_periodicty:
            from metrics.periodicity import calculate_periodicity_metrics

            periodicity_loss, pitch_loss, f1_score = calculate_periodicity_metrics(audio_16_khz, audio_hat_16khz)
        else:
            periodicity_loss = pitch_loss = f1_score = 0

        if evaluate_metrics and self.hparams.evaluate_utmos:
            utmos_score = self.utmos_model.score(audio_hat_16khz.unsqueeze(1)).mean()
        else:
            utmos_score = None

        if evaluate_metrics and self.hparams.evaluate_pesq:
            from pesq import pesq

            pesq_score = 0
            for ref, deg in zip(audio_16_khz.cpu().numpy(), audio_hat_16khz.cpu().numpy()):
                pesq_score += pesq(16000, ref, deg, "wb", on_error=1)
            pesq_score /= len(audio_16_khz)
            pesq_score = torch.tensor(pesq_score, device=self.device)
        else:
            pesq_score = None

        if evaluate_metrics and self.hparams.evaluate_mrstft:
            assert self.mrstft_loss is not None
            mrstft_loss = self.mrstft_loss(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))
        else:
            mrstft_loss = None

        mel_loss = self.melspec_loss(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))
        if self.hparams.validation_metrics_affect_val_loss:
            utmos_for_loss = utmos_score if evaluate_metrics and self.hparams.evaluate_utmos else 0
            pesq_for_loss = pesq_score if pesq_score is not None else 0
            total_loss = mel_loss + (5 - utmos_for_loss) + (5 - pesq_for_loss)
        else:
            total_loss = mel_loss

        return {
            "val_loss": total_loss,
            "mel_loss": mel_loss,
            "utmos_score": utmos_score,
            "pesq_score": pesq_score,
            "mrstft_loss": mrstft_loss,
            "periodicity_loss": periodicity_loss,
            "pitch_loss": pitch_loss,
            "f1_score": f1_score,
            "audio_input": audio_input[0],
            "audio_pred": audio_hat[0],
        }

    def validation_epoch_end(self, outputs):
        # A periodic checkpoint may be taken between the train and validation
        # loops. Older Lightning releases then resume into an empty validation
        # epoch; there are no metrics or media to aggregate in that case.
        if not outputs:
            return
        if self.global_rank == 0:
            *_, audio_in, audio_pred = outputs[0].values()
            self._log_audio("val_in", audio_in)
            self._log_audio("val_pred", audio_pred)
            mel_target = safe_log(self.melspec_loss.mel_spec(audio_in))
            mel_hat = safe_log(self.melspec_loss.mel_spec(audio_pred))
            self._log_image("val_mel_target", plot_spectrogram_to_numpy(mel_target.data.cpu().numpy()))
            self._log_image("val_mel_hat", plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()))
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mel_loss = torch.stack([x["mel_loss"] for x in outputs]).mean()
        utmos_score = self._average_available(outputs, "utmos_score")
        pesq_score = self._average_available(outputs, "pesq_score")
        mrstft_loss = self._average_available(outputs, "mrstft_loss")
        periodicity_loss = np.array([x["periodicity_loss"] for x in outputs]).mean()
        pitch_loss = np.array([x["pitch_loss"] for x in outputs]).mean()
        f1_score = np.array([x["f1_score"] for x in outputs]).mean()

        self.log("val_loss", avg_loss, sync_dist=True)
        self.log("val/mel_loss", mel_loss, sync_dist=True)
        self.log("val/mel_spec_error", mel_loss, sync_dist=True)
        if utmos_score is not None:
            self.log("val/utmos_score", utmos_score, sync_dist=True)
        if pesq_score is not None:
            self.log("val/pesq_score", pesq_score, sync_dist=True)
            self.log("val/pesq", pesq_score, sync_dist=True)
        if mrstft_loss is not None:
            self.log("val/mrstft", mrstft_loss, sync_dist=True)
        self.log("val/periodicity_loss", periodicity_loss, sync_dist=True)
        self.log("val/pitch_loss", pitch_loss, sync_dist=True)
        self.log("val/f1_score", f1_score, sync_dist=True)

    @property
    def global_step(self):
        """
        Override global_step so that it returns the total number of batches processed
        """
        return self.trainer.fit_loop.epoch_loop.total_batch_idx

    def on_train_batch_start(self, *args):
        if self.global_step >= self.hparams.pretrain_mel_steps:
            self.train_discriminator = True
        else:
            self.train_discriminator = False

    def on_train_batch_end(self, *args):
        def mel_loss_coeff_decay(current_step, num_cycles=0.5):
            max_steps = self.trainer.max_steps // 2
            if current_step < self.hparams.num_warmup_steps:
                return 1.0
            progress = float(current_step - self.hparams.num_warmup_steps) / float(
                max(1, max_steps - self.hparams.num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        if self.hparams.decay_mel_coeff:
            self.mel_loss_coeff = self.base_mel_coeff * mel_loss_coeff_decay(self.global_step + 1)


class VocosEncodecExp(VocosExp):
    """
    VocosEncodecExp is a subclass of VocosExp that overrides the parent experiment to function as a conditional GAN.
    It manages an additional `bandwidth_id` attribute, which denotes a learnable embedding corresponding to
    a specific bandwidth value of EnCodec. During training, a random bandwidth_id is generated for each step,
    while during validation, a fixed bandwidth_id is used.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
        sample_rate: int,
        initial_learning_rate: float,
        num_warmup_steps: int,
        mel_loss_coeff: float = 45,
        mrd_loss_coeff: float = 1.0,
        pretrain_mel_steps: int = 0,
        decay_mel_coeff: bool = False,
        evaluate_utmos: bool = False,
        evaluate_pesq: bool = False,
        evaluate_periodicty: bool = False,
    ):
        super().__init__(
            feature_extractor,
            backbone,
            head,
            sample_rate,
            initial_learning_rate,
            num_warmup_steps,
            mel_loss_coeff,
            mrd_loss_coeff,
            pretrain_mel_steps,
            decay_mel_coeff,
            evaluate_utmos,
            evaluate_pesq,
            evaluate_periodicty,
        )
        # Override with conditional discriminators
        self.multiperioddisc = MultiPeriodDiscriminator(num_embeddings=len(self.feature_extractor.bandwidths))
        self.multiresddisc = MultiResolutionDiscriminator(num_embeddings=len(self.feature_extractor.bandwidths))

    def training_step(self, *args):
        bandwidth_id = torch.randint(low=0, high=len(self.feature_extractor.bandwidths), size=(1,), device=self.device,)
        output = super().training_step(*args, bandwidth_id=bandwidth_id)
        return output

    def validation_step(self, *args):
        bandwidth_id = torch.tensor([0], device=self.device)
        output = super().validation_step(*args, bandwidth_id=bandwidth_id)
        return output

    def validation_epoch_end(self, outputs):
        if self.global_rank == 0:
            *_, audio_in, _ = outputs[0].values()
            # Resynthesis with encodec for reference
            self.feature_extractor.encodec.set_target_bandwidth(self.feature_extractor.bandwidths[0])
            encodec_audio = self.feature_extractor.encodec(audio_in[None, None, :])
            self._log_audio("encodec", encodec_audio[0, 0])

        super().validation_epoch_end(outputs)
