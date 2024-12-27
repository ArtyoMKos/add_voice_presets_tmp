import os
import random
from time import time

from accelerate.commands.config.default import description
from tqdm import tqdm

from typing import Optional, Union
from bson import ObjectId
import numpy as np
import pandas as pd
import torch
import torchaudio
import string

from audiokit.text.phonemizer import Phonemizer
from audiokit.text.tokenizer import Tokenizer
from audiokit.utils.asr import ASRWhisper
from audiokit.utils.audio import Resampler
from audiokit.utils.vocoder import VocoderModel
from castle_flow import utils
from castle_flow.connect_to_vc import VC
from castle_flow.autoencoder.config import AutoEncoderTrainerConfig
from castle_flow.autoencoder.model import MelSpecAutoEncoderForQRec
from castle_flow.flow_decoder.config import FlowDecoderTrainerConfig
from castle_flow.flow_decoder.model import FlowDecoderModelForCFM
from castle_flow.speech_token_predictor.config import STPTrainerConfig
from castle_flow.speech_token_predictor.inference_config import InferenceConfig, InferenceHParams
from castle_flow.speech_token_predictor.model import (
    SpeechTokenizer, SpeechTokenPredictor, SpeechTokenPredictorWrapper,
    SpeechTokenPredictorForClassification, SpeechTokenPredictorSampleInputs)
from configs.config_wrapper import ConfigWrapper
from database import AudioDataRepository


class Inference:
    def __init__(
        self,
        inference_config_path: str | dict,
    ):
        if isinstance(inference_config_path, dict):
            self.inference_config = InferenceConfig.load_from_dict(inference_config_path)
        elif isinstance(inference_config_path, str):
            if not os.path.exists(inference_config_path):
                raise ValueError(f"{inference_config_path} does not exist")
            self.inference_config = InferenceConfig.load_from_yaml(inference_config_path)
        else:
            raise ValueError(
                f"inference_config_path must be either a path to a yaml file or a dictionary, got {type(inference_config_path)}"
            )
        
        self._fix_seed()
        
        self.stp_config: STPTrainerConfig = STPTrainerConfig.load_from_yaml(
            self.inference_config.stp_config_path
        )
        self.flow_decoder_config: FlowDecoderTrainerConfig = FlowDecoderTrainerConfig.load_from_yaml(
            self.inference_config.flow_decoder_config_path
        )
        
        self.phonemizer = Phonemizer(self.inference_config.phonemizer_service_url)
        self.text_tokenizer = self._load_text_tokenizer()
        self.speech_tokenizer = self._load_speech_tokenizer()
        self.flow_decoder = self._load_flow_decoder_torchscript()
        self.speech_token_predictor = self._load_speech_token_predictor()
        
        self.vocoder = self._load_vocoder()
        self.asr = self._load_asr()
        
        self.resampler = Resampler()

        self.vc = VC()
    
    def _fix_seed(self):
        if self.inference_config.seed is None:
            return
        torch.manual_seed(self.inference_config.seed)
        torch.cuda.manual_seed(self.inference_config.seed)
        np.random.seed(self.inference_config.seed)
        random.seed(self.inference_config.seed)
    
    def _load_text_tokenizer(
        self,
    ) -> Tokenizer:
        tokenizer = Tokenizer(
            vocab_path=self.stp_config.data.vocab_path,
            token_separator=self.stp_config.data.token_separator,
            add_blank=self.stp_config.data.add_blank
        )
        
        return tokenizer
        
    def _load_speech_tokenizer(
        self,
    ) -> SpeechTokenizer:
        assert (
            self.flow_decoder_config.autoencoder_config_path == self.stp_config.autoencoder_config_path == \
            self.inference_config.autoencoder_config_path   
        ), \
            "autoencoder config path in flow_decoder, stp and inference config must match"

        assert (
            self.flow_decoder_config.autoencoder_ckpt_path == self.stp_config.autoencoder_ckpt_path == \
            self.inference_config.autoencoder_checkpoint_path
        ), \
            "autoencoder ckpt path in flow_decoder, stp and inference config must match"
        
        autoencoder_config: AutoEncoderTrainerConfig = AutoEncoderTrainerConfig.load_from_yaml(
            self.inference_config.autoencoder_config_path
        )
        autoencoder_for_qrec = MelSpecAutoEncoderForQRec(autoencoder_config)
        autoencoder_for_qrec.load_state_dict(
            torch.load(
                self.inference_config.autoencoder_checkpoint_path, map_location='cpu', weights_only=True
            )["model"], strict=True
        )
        autoencoder = autoencoder_for_qrec.autoencoder
        autoencoder.eval()
        
        speech_tokenizer = SpeechTokenizer(
            autoencoder=autoencoder,
            add_bos_token=self.stp_config.data.add_bos_token,
            add_eos_token=self.stp_config.data.add_eos_token
            # add_eos_token=False
        )
        
        return speech_tokenizer.to(self.inference_config.device)
    
    def _load_flow_decoder_torchscript(
        self,
    ) -> FlowDecoderModelForCFM:
        flow_decoder = FlowDecoderModelForCFM(self.flow_decoder_config)
        # flow_decoder.load_state_dict(
        #     torch.load(
        #         self.inference_config.flow_decoder_checkpoint_path,
        #         map_location='cpu',
        #         weights_only=True
        #     )["model"], strict=True
        # )
        
        flow_decoder.model = torch.jit.load(self.inference_config.flow_decoder_checkpoint_path)
        flow_decoder.eval()
        
        return flow_decoder.to(self.inference_config.device)

    def _load_flow_decoder(
        self,
    ) -> FlowDecoderModelForCFM:
        flow_decoder = FlowDecoderModelForCFM(self.flow_decoder_config)
        flow_decoder.load_state_dict(
            torch.load(
                self.inference_config.flow_decoder_checkpoint_path,
                map_location='cpu',
                weights_only=True
            )["model"], strict=True
        )
        
        flow_decoder.model = torch.jit.load("/mnt/shared_data/users/artyom/projects/castle-ai-model-convert-factory/converted_models/mel_spec_flow_decoder/v2/mel_spec_flow_decoder_v2_1.pt")
        flow_decoder.eval()
        
        return flow_decoder.to(self.inference_config.device)
    
    def _load_speech_token_predictor(
        self,
    ) -> Union[SpeechTokenPredictor, SpeechTokenPredictorWrapper]:
        speech_token_predictor_wrapper = SpeechTokenPredictorForClassification(
            self.stp_config
        )
        
        if "+ts" in self.inference_config.stp_checkpoint_path:
            speech_token_predictor_wrapper.model = SpeechTokenPredictorWrapper(
                self.stp_config.model,
                torch.jit.load(self.inference_config.stp_checkpoint_path)
            )
        else:
            speech_token_predictor_wrapper.load_state_dict(
                torch.load(
                    self.inference_config.stp_checkpoint_path,
                    map_location='cpu',
                    weights_only=True
                )["model"], strict=True
            )
        speech_token_predictor = speech_token_predictor_wrapper.model
        speech_token_predictor.eval()
        
        return speech_token_predictor.to(self.inference_config.device)
    
    def _load_vocoder(self) -> Optional[VocoderModel]:
        vocoder = VocoderModel(
            model_path=self.inference_config.vocoder_path,
            hop_length=self.stp_config.data.hop_length,
            device=self.inference_config.device
        )
        print(f'Loaded Vocoder {self.inference_config.vocoder_path}')
        return vocoder
    
    def _load_asr(self) -> Optional[ASRWhisper]:
        asr = ASRWhisper(
            model_id=self.inference_config.asr_model_id,
            device=self.inference_config.device,
            return_timestamps=False,
            return_language=False,
        ) if self.inference_config.asr_model_id is not None else None
        if asr is not None:
            print(f'Loaded ASR {self.inference_config.asr_model_id}')
        
        return asr
    
    def transcribe(
        self,
        audio: str | torch.Tensor,
        sampling_rate: Optional[int] = None,
        language: Optional[str] = None
    ) -> str:
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        output = self.asr(audio, sampling_rate=sampling_rate, language=language)
        return output['text'].strip()
    
    def get_mel(
        self,
        audio: torch.Tensor,
        downsample_factor: int,
    ) -> torch.Tensor:
        mel = utils.get_mel(
            audio=audio.unsqueeze(0),
            config=self.flow_decoder_config.data
        )
        padding = (
            downsample_factor - mel.shape[-1] % downsample_factor
        ) % downsample_factor
        mel = torch.nn.functional.pad(mel, (0, padding), value=np.log(1e-5))
        return mel
    
    def prepare_prompt(
        self,
        audio_path: str,
        inference_hparams: InferenceHParams,
        text: Optional[str] = None,
    ) -> dict[str, torch.Tensor]:
        audio, sr = utils.load_audio(
            audio_path=audio_path,
            chunk_size=inference_hparams.chunk_size,
            cut_left_chunk=inference_hparams.cut_left_chunk,
            cut_right_chunk=inference_hparams.cut_right_chunk,
            cut_random_chunk=inference_hparams.cut_random_chunk,
            device=self.inference_config.device,
        )
        audio = self.resampler.resample(
            audio=audio,
            sr=sr,
            target_sr=self.flow_decoder_config.data.sampling_rate,
            device=self.inference_config.device,
        )
        transcribe = any([
            text is None,
            inference_hparams.cut_random_chunk,
            inference_hparams.cut_left_chunk,
            inference_hparams.cut_right_chunk
        ])
        if transcribe:
            text = self.transcribe(
                audio,
                self.flow_decoder_config.data.sampling_rate,
                language=self.inference_config.language
            )
        
        phonemes = self.phonemizer(text)
        
        # check if phonemes end with punctuation
        if not phonemes[-1] in string.punctuation:
            phonemes = phonemes + "."
        phonemes = phonemes + " "
        
        text_tokens = torch.LongTensor(self.text_tokenizer(phonemes))
        
        mel = self.get_mel(
            audio, self.flow_decoder_config.model.downsample_factor
        )
        # mel = (mel - self.flow_decoder_config.data.mel_mean) / self.flow_decoder_config.data.mel_std
        
        speech_tokens, _ = self.speech_tokenizer.tokenize(
            mel_spec=mel,
            mel_spec_lengths=torch.tensor([mel.shape[-1]]).to(self.inference_config.device)
        )
        # Todo: commented because I'm removing that token in tts service
        # if self.speech_tokenizer.add_eos_token:
        #     speech_tokens = speech_tokens[:, :-1]
        mel = (mel - self.flow_decoder_config.data.mel_mean) / self.flow_decoder_config.data.mel_std
        if self.vocoder is not None:
            audio_vocoded = self.vocoder(
                mel * self.flow_decoder_config.data.mel_std + self.flow_decoder_config.data.mel_mean
            ).squeeze(0).cpu()
        else:
            audio_vocoded = None
        
        return {
            "mel": mel,
            "text_tokens": text_tokens,
            "speech_tokens": speech_tokens,
            "text": text,
            "phonemes": phonemes,
            "vocoded": audio_vocoded,
            "audio": audio
        }
    
    def prepare_target(
        self,
        text: str,
    ) -> torch.LongTensor:
        if text[-1] not in string.punctuation:
            text = text + "."
        phonemes = self.phonemizer(text)
        text_tokens = torch.LongTensor(self.text_tokenizer(phonemes))
        
        return text_tokens
    
    def prepare_model_inputs(
        self,
        prompt: dict[str, torch.Tensor] | None,
        target_text_tokens: torch.LongTensor,
        inference_hparams: InferenceHParams,
    ) -> SpeechTokenPredictorSampleInputs:
        if prompt is None:
            prompt_speech_tokens = torch.ones((1, 1)).long() * self.speech_tokenizer.bos_token
            prompt_speech_tokens = prompt_speech_tokens.to(self.inference_config.device)
            text_tokens = target_text_tokens
        else:
            text_tokens = torch.cat([prompt["text_tokens"], target_text_tokens], dim=-1)
            prompt_speech_tokens = prompt["speech_tokens"]
        
        model_input = SpeechTokenPredictorSampleInputs(
            text_tokens=text_tokens.to(self.inference_config.device),
            speech_tokens=prompt_speech_tokens.to(self.inference_config.device),
            text_lengths=torch.LongTensor(
                [text_tokens.shape[-1]]
            ).to(self.inference_config.device),
            speech_token_lengths=torch.LongTensor(
                [prompt_speech_tokens.shape[-1]]
            ).to(self.inference_config.device),
            max_ar_steps=inference_hparams.max_ar_steps,
            eos_token=self.speech_tokenizer.eos_token,
            top_k=inference_hparams.top_k,
            temperature=inference_hparams.temperature,
            penalty=inference_hparams.penalty,
            top_p=inference_hparams.top_p,
            use_cache=inference_hparams.use_cache,
            min_prob_eos=inference_hparams.min_prob_eos
        )
        
        return model_input
    
    @torch.inference_mode()
    @torch.jit.optimized_execution(False)
    def run_flow_decoder(
        self,
        model_inputs: SpeechTokenPredictorSampleInputs,
        speech_tokens_pred: torch.LongTensor,
        inference_hparams: InferenceHParams,
        prompt: Optional[dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        prompt_tokens_wo_sos = model_inputs.speech_tokens[:, 1:]
        tokens_concat = torch.cat([prompt_tokens_wo_sos, speech_tokens_pred], dim=-1)
        tokens_concat_len = torch.LongTensor([tokens_concat.shape[-1]]).to(
            self.inference_config.device
        ) * self.flow_decoder_config.model.downsample_factor
        
        spec = torch.zeros((
            tokens_concat.shape[0], self.flow_decoder_config.data.n_mels, tokens_concat_len
        )).to(self.inference_config.device)
        ctx_mask = torch.zeros_like(spec[:, 0:1, :]).to(self.inference_config.device)
        if prompt is not None:
            spec[:, :, :prompt["mel"].shape[-1]] = prompt["mel"]
            ctx_mask[:, :, :prompt["mel"].shape[-1]] = 1.
        
        output_mel = self.flow_decoder.inference(
            tokens=tokens_concat,
            speech=spec,
            speech_lengths=tokens_concat_len,
            context_mask=ctx_mask,
            alpha=inference_hparams.alpha,
            n_steps=inference_hparams.n_steps,
            sigma=inference_hparams.sigma
        )
        output_mel = output_mel * self.flow_decoder_config.data.mel_std + self.flow_decoder_config.data.mel_mean
        if prompt is not None:
            output_mel = output_mel[..., prompt["mel"].shape[-1]: ]
        
        return output_mel
    
    @torch.inference_mode()
    def run(
        self,
        prompt_audio_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        inference_hparams: Optional[InferenceHParams] = None,
    ) -> dict:
        
        if inference_hparams is None:
            inference_hparams = self.inference_config.inference_hparams
        
        dtype = torch.float16 if self.inference_config.enable_fp16 else torch.float32
        with torch.autocast(device_type=self.inference_config.device, dtype=dtype):
            prompt = self.prepare_prompt(
                audio_path=prompt_audio_path,
                inference_hparams=inference_hparams,
                text=prompt_text
            ) if prompt_audio_path is not None else None

            audio = prompt["audio"]
            prompt_tokens = prompt["text_tokens"]
            speaker_embs = self.vc.get_speaker_embs(
                audio=audio.tolist(),
                tokens=prompt_tokens.tolist(),
                sample_rate=self.flow_decoder_config.data.sampling_rate
            )

        return {
            'speaker_embs': speaker_embs,
            'spec_norm': prompt["mel"][0].cpu().numpy(),
            'speech_tokens': prompt['speech_tokens'][0].cpu().numpy(),
            'prompt_tokens': prompt_tokens.cpu().numpy()
        }


if __name__ == "__main__":
    CONFIGS = ConfigWrapper().config

    configs = CONFIGS["DEFAULT"]

    db = AudioDataRepository(
        configs["database_name"],
        host=f"mongodb+srv://"
             f"{os.environ['DB_USERNAME']}:"
             f"{os.environ['DB_PASSWORD']}@"
             f"{configs['database_url']}",
    )

    infer = Inference("configs/inference_speech_token_predictor.yaml")
    voices_metadata = pd.read_csv('/mnt/datasets/processed/bunny/valid_prompts.csv')
    pbar = tqdm(
        total=voices_metadata.shape[0]
    )

    for _, row in voices_metadata.iterrows():
        voice_id = row['voice_id']
        prompt_audio_path = row['prompt_audio_path']
        prompt_text = row['prompt_text']

        output = infer.run(
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text
        )

        audio_id = db.save_audio_data(
            output['speaker_embs'],
            output['spec_norm'],
            output['speech_tokens'],
            output['prompt_tokens'],
            metadata="Sample audio metadata",
            _id=voice_id
        )
        print(f"Audio Data saved with ID: {audio_id}")

        # Retrieve from database and get all data as one base64 string
        audio_data_base64 = db.get_audio_data_as_base64(audio_id)
        print("Saved audio data.")
        pbar.update(1)
