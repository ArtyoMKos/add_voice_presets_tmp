import io

import requests
import numpy as np
import torchaudio


class VC:
    def __init__(
            self,
            endpoint: str = "http://localhost:8001/voice_style_cloning/editing"
    ):
        self._endpoint = endpoint

    def _send_request(self, data: dict):
        """
        Sends a single POST request asynchronously to the given URL with the specified data.

        Args:
            client (httpx.AsyncClient): The HTTP client to use for sending the request.
            url (str): The URL to which the request will be sent.
            data (dict): The JSON data to send in the request body.
        """
        response = requests.post(self._endpoint, json=data)
        buffer = io.BytesIO(response.content)
        with np.load(buffer) as data:
            mel_spec = data["mel_spec"]
            durations = data["durations"]
            tokens = data["tokens"]
            sample_rate = int(data["sample_rate"])
            embs = data.get("embs")
        return {
            "mel_spec": mel_spec,
            "speaker_embs": embs
        }

    def get_speaker_embs(
            self,
            audio,
            tokens,
            sample_rate,

    ):
        response = self._send_request(
            {
                "audio": audio,
                "tokens": tokens,
                "sample_rate": sample_rate
            }
        )
        return response["speaker_embs"]

# waveform, sample_rate = torchaudio.load('experiments/dmitriy.wav')
# waveform = waveform.cpu().numpy()
# waveform = np.mean(waveform, axis=0, dtype=waveform.dtype)
