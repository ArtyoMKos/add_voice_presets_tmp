import os
from torch.cuda.amp import autocast

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json

import torch
import torchaudio as ta
import torchdiffeq
import numpy as np

from commons import utils
from commons.utils import commons, helpers
from models.voicebox_2.v1.models import (
    VQMelSpecAutoEncoder,
    MelSpecFlowDecoder,
    SpeechTokenizer,
)
from models.stp.v1.models import SpeechTokenPredictor, StpConverted

from torch.nn import functional as F
from commons.text.tokenizer import Tokenizer
from commons.text.phonemizer import Phonemizer
from commons.mel_processing import mel_spectrogram_torch

import random

from models.vocoder.v2.bigvgan_v2_22khz_80band_256x import bigvgan

torch.manual_seed(19)
np.random.seed(19)
random.seed(19)

MAX_WAV_VALUE = 32767


def get_new_vocoder_raw(*args, **kwargs):
    device = 'cuda'

    model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_22khz_80band_256x', use_cuda_kernel=False)

    # for resblock in model.resblocks:
    #     resblock.activations_1 = resblock.activations[::2]
    #     resblock.activations_2 = resblock.activations[1::2]

    # remove weight norm in the model and set to eval mode
    model.remove_weight_norm()
    model = model.eval().to(device)
    return model


def get_bigvgan_vocoder(checkpoint_path, device):
    from models.vocoder.v1.bigvgan.env import AttrDict
    from models.vocoder.v1.bigvgan.models import BigVGAN as Generator

    global MAX_WAV_VALUE

    def load_checkpoint(filepath, device):
        print(filepath)
        assert os.path.isfile(filepath)
        print("Loading '{}'".format(filepath))
        checkpoint_dict = torch.load(filepath, map_location=device)
        print("Complete.")
        return checkpoint_dict

    config_path = os.path.join(os.path.split(checkpoint_path)[0], "config.json")
    with open(config_path, "r") as f:
        data = f.read()
    hps = AttrDict(json.loads(data))

    generator = Generator(hps).to(device)

    state_dict_g = load_checkpoint(checkpoint_path, device)
    generator.load_state_dict(state_dict_g["generator"])
    generator.eval()
    generator.remove_weight_norm()

    return generator


def vocode(x, generator):
    with torch.no_grad():
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        y_g_hat = generator(x)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype("int16")
    return audio


def load_speech_tokenizer(chkpt_path, hps, device="cpu"):
    model = VQMelSpecAutoEncoder(
        n_mel_channels=hps.data.n_mel_channels,
        # use_context_masking=use_context_masking,
        **hps.model,
    ).to(device)

    model.enc.enc.enc.fill_skip_conn_with_dummy_layers()  # Todo: Artko: Testing adaptation.

    print(f"[INFO] Loading last checkpoint {chkpt_path}")
    model, _, _, epoch, steps, _ = utils.load_checkpoint(chkpt_path, model)
    model.eval()
    print("Ready")

    return model, steps


def load_model(chkpt_path, hps, device="cpu"):
    # model = MelSpecFlowDecoder(
    #     n_mel_channels=hps.data.n_mel_channels,
    #     **hps.model
    # )

    # print(f"[INFO] Loading last checkpoint {chkpt_path}")
    # model.to(device)
    # model, _, _, epoch, steps, _ = utils.load_checkpoint(chkpt_path, model)
    # model.dec.enc.fill_skip_conn_with_dummy_layers()  # Todo: Artko: Testing adaptation.

    # model.eval()
    # print("Ready")

    # return model, steps
    model = torch.load(
        "converted_models/mel_spec_flow_decoder/v1/mel_spec_flow_decoder_v1.pt"
    )
    model.to(device)
    return model, 1000000


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
base = "/storage_drive_2/maintanance/214/aiteam_gevorg/castle-ai-speech-flow"
vocoder_checkpoint_path = f"./runs/logs/g_05000000"

base = "."
hps = utils.get_hparams_from_file(
    f"{base}/runs/logs/flow_decoder_v0.0.9+downsample4/config.yaml"
)
st_hps = utils.get_hparams_from_file(f"{base}/{hps.model.speech_tokenizer_config}")

# vocoder = get_bigvgan_vocoder(vocoder_checkpoint_path, DEVICE)
vocoder = get_new_vocoder_raw()
# vocoder = torch.load("converted_models/vocoder_model.pt").to(DEVICE)

speech_tokenizer, model_steps = load_speech_tokenizer(
    chkpt_path=f"{base}/{hps.model.speech_tokenizer_chkpt_path}",
    hps=st_hps,
    device=DEVICE,
)


# checkpoint_path = utils.get_latest_checkpoint_path(f'{hps.run_dir}', 'G*')
model, model_steps = load_model(
    chkpt_path="runs/logs/flow_decoder_v0.0.9+downsample4/G_1000000.pth",
    # chkpt_path=checkpoint_path,
    hps=hps,
    device=DEVICE,
    # use_context_masking=False
)

stp_hps = utils.get_hparams_from_file(
    f"runs/logs/speech_token_predictor_v0.0.13+downsample/config.yaml"
)


def load_speech_token_predictor(chkpt_path, hps, device="cpu"):
    model = SpeechTokenPredictor(
        n_vocab=tokenizer.n_symbols,
        n_vocab_speech=speech_tokenizer_.n_vocab_speech,
        speech_pad_token=speech_tokenizer_.pad_token,
        n_mel_channels=hps.data.n_mel_channels,
        **hps.model
    ).to(device)

    print(f"Loading last checkpoint {chkpt_path}")
    model, _, _, epoch, steps, _ = utils.load_checkpoint(chkpt_path, model)
    model = model.eval()
    model = model.half()

    return model, steps

    # model = StpConverted(torch.jit.script(model))
    # model.model.to(device)
    # print("Ready")
    # return torch.jit.optimize_for_inference(model), 420000

    # model = StpConverted(frozen=False, optimize=False)
    # model.model.to(device)
    # return model, 420000


phonemizer = Phonemizer(stp_hps.data)
stp_hps.run_dir

stp_checkpoint_path = utils.get_latest_checkpoint_path(stp_hps.run_dir, "G*")
tokenizer = Tokenizer(stp_hps.data)
speech_tokenizer_ = SpeechTokenizer(
    quantizer_model=speech_tokenizer,
    add_bos_token=stp_hps.data.add_bos_token,
    add_eos_token=stp_hps.data.add_eos_token,
)

stp_model, model_steps = load_speech_token_predictor(
    chkpt_path=stp_checkpoint_path,
    hps=stp_hps,
    device=DEVICE,
    # use_context_masking=False
)


def get_mel_from_audio(audio: torch.FloatTensor):
    return mel_spectrogram_torch(
        y=audio,
        n_fft=hps.data.filter_length,
        num_mels=hps.data.n_mel_channels,
        sampling_rate=hps.data.sampling_rate,
        hop_size=hps.data.hop_length,
        win_size=hps.data.filter_length,
        fmin=hps.data.mel_fmin,
        fmax=hps.data.mel_fmax,
        center=False,
        padding=True,
    )


audio_path = "test_prompts/dmitriy.wav"
# audio_path = 'test_prompts/artko_prompt.wav'
# audio_path = 'test_prompts/artko_3_1.wav'

audio, sr = ta.load(audio_path)
audio = audio.mean(dim=0, keepdim=True)

resampler = commons.Resampler()
audio = resampler.resample(audio, sr, hps.data.sampling_rate, device="cpu")
audio_prompt = audio
spec = get_mel_from_audio(audio_prompt)
spec_len = spec.shape[-1]
padding = (
    hps.model.downsample - spec_len % hps.model.downsample
) % hps.model.downsample
spec = F.pad(spec, (0, padding), value=np.log(1e-5))

spec_len = torch.tensor([spec.shape[-1]]).to(DEVICE)
spec = spec.to(DEVICE)

spec_norm = (spec - hps.data.spec_mean) / hps.data.spec_std
speech_tokens, speech_tokens_lengths = speech_tokenizer_.tokenize(spec, spec_len)


text_ipa = "there were some curtains on the window and some rugs on the floor. "
# text_ipa = "Hey, my name is Artyom. How are you? I heard that something is weird with your latest experience in hiking. Am I right?"
# text_ipa = "You know that I'm working on this project? "
text_tokens = tokenizer(phonemizer(text_ipa))
text_tokens_prompt = torch.LongTensor([text_tokens]).to(DEVICE)

text_ipa = phonemizer(
    "I was, like, talking to my friend, and she’s all, um, excited about her, uh, trip to Europe, and I’m just, like, so jealous, right?"
)
text_tokens = tokenizer(text_ipa)
text_tokens = torch.LongTensor([text_tokens]).to(DEVICE)
text_tokens = torch.cat([text_tokens_prompt, text_tokens], dim=-1)
text_tokens_lengths = torch.LongTensor([text_tokens.size(-1)]).to(DEVICE)

with torch.no_grad():
    # with autocast(enabled=True):
    for _ in range(1):
        speech_tokens_pred, _, accumulated_outputs = stp_model.sample(
            y=text_tokens.clone(),
            y_lengths=text_tokens_lengths.clone(),
            l=speech_tokens[:, :-1].clone(),
            l_lengths=(speech_tokens_lengths - 1).clone(),
            max_ar_steps=1200,
            eos_token=speech_tokenizer_.eos_token,
            top_k=20,
            penalty=1.5,
        )

# Artko: saving outputs
print("accumulated_outputs: ", len(accumulated_outputs))
# for i, output in enumerate(accumulated_outputs):
#     helpers.save_output_tensors(output, f"experiments/output_tensors/output_tensor_step_nt_{i}.pt")
# print("Saved tensors: ", len(accumulated_outputs))

tokens_with_pred = torch.cat([speech_tokens[:, 1:-1], speech_tokens_pred], dim=-1)
spec_norm_with_zeros = torch.cat(
    [
        spec_norm,
        torch.zeros(
            spec_norm.size(0),
            spec_norm.size(1),
            speech_tokens_pred.size(-1) * st_hps.model.downsample,
        ).to(DEVICE),
    ],
    dim=-1,
)
m = torch.zeros_like(spec_norm_with_zeros[:, 0:1, :]).to(DEVICE)
m[:, :, : spec_norm.size(-1)] = 1
spec_len_with_zeros = torch.tensor([spec_norm_with_zeros.shape[-1]]).to(DEVICE)

alpha = 0.7
with torch.no_grad():
    mel_hat = torchdiffeq.odeint(
        lambda t, w: model.decode_step(
            t,
            w,
            tokens_with_pred,
            spec_norm_with_zeros,
            spec_len_with_zeros,
            m,
            alpha=alpha,
        ),
        torch.randn(
            spec_norm_with_zeros.shape[0], 80, spec_norm_with_zeros.shape[-1]
        ).to(DEVICE),
        torch.linspace(0, 1, 2).to(DEVICE),
        method="midpoint",
        options=dict(step_size=0.0625),
    )
    mel_hat = mel_hat[-1].squeeze(0)
    mel_hat = mel_hat * hps.data.spec_std + hps.data.spec_mean

    audio = vocode(mel_hat, vocoder)
    audio = audio.astype(np.float32, order="C") / MAX_WAV_VALUE

print("orig audio")
# ipd.display(ipd.Audio(audio_prompt, rate=22050))
ta.save("experiments/orig_audio.wav", audio_prompt, sample_rate=22050)

print("decoded only tokens")
# ipd.display(ipd.Audio(audio[spec_norm.shape[-1]*256:], rate=hps.data.sampling_rate))
ta.save(
    "experiments/decoded_only_tokens.wav",
    torch.from_numpy(
        audio[spec_norm.shape[-1] * 256 :].reshape(
            1, audio[spec_norm.shape[-1] * 256 :].shape[-1]
        )
    ),
    sample_rate=hps.data.sampling_rate,
)

print("full decoded")
# ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate))
ta.save(
    "experiments/full_decoded.wav",
    torch.from_numpy(audio.reshape(1, audio.shape[-1])),
    sample_rate=hps.data.sampling_rate,
)

# scripted = torch.jit.script(stp_model)
# torch.jit.save(scripted, "converted_models/stp/v1/stp_v1_backup.pt")
