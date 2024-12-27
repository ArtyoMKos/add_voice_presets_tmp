import base64
import io
import os

import numpy as np

from configs.config_wrapper import ConfigWrapper
from database import AudioDataRepository

CONFIGS = ConfigWrapper().config

configs = CONFIGS["DEFAULT"]

db = AudioDataRepository(
    configs["database_name"],
    host=f"mongodb+srv://"
         f"{os.environ['DB_USERNAME']}:"
         f"{os.environ['DB_PASSWORD']}@"
         f"{configs['database_url']}",
)


def _get_prompt_from_db(voice_preset_id: str, _database) -> dict:
    """
    Retrieves the voice preset data from the database.

    Args:
        voice_preset_id (str): ID of the voice preset to retrieve.

    Returns:
        dict: Base64-encoded audio data for the preset.
        - 'data': Key contains base64-encoded voice preset data.
    """
    audio_data_base64 = _database.get_audio_data_as_base64(voice_preset_id)
    return audio_data_base64

def _decode_base64_preset(content):
    binary_content = base64.b64decode(
        content
    )
    buffer = io.BytesIO(binary_content)
    with np.load(buffer) as data:
        spec_norm = data["spec_norm"]
        speech_tokens = data["speech_tokens"]
        prompt_tokens = data["prompt_tokens"]
        speaker_embs = data["speaker_embs"]
    return {
        'spec_norm': spec_norm,
        'speech_tokens': speech_tokens,
        'prompt_tokens': prompt_tokens,
        'speaker_embs': speaker_embs
    }

voice_presets = _get_prompt_from_db(
    'cc36c623-ca34-492a-bae4-4f54c4036786',
    db
)

data = _decode_base64_preset(voice_presets['data'])

print(f"Voice presets count: {voice_presets['total_objects']}")
