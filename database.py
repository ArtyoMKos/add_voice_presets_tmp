import logging
from datetime import datetime
from typing import Optional, Dict, Unpack, Union
import io
import base64
import numpy as np
from bson import ObjectId
from mongoengine import (Document, BinaryField,
                         StringField, connect,
                         ListField, ObjectIdField,
                         DateTimeField
                         )


class AudioData(Document):
    # id = ObjectIdField(primary_key=True, unique=True)
    id = StringField(primary_key=True)
    data_binary: BinaryField = BinaryField(required=True)  # Storing the entire npz compressed file as binary
    metadata: Optional[str] = StringField(required=False)
    # created_at = DateTimeField(default=datetime.utcnow)
    description: str = StringField(
        default="Data includes speaker embeddings, spectrogram, and speech tokens"
    )
    keys: ListField = ListField(
        StringField(),
        default=["speaker_embs", "spec_norm", "speech_tokens", "prompt_tokens"],
    )

    meta = {"collection": "tts_voice_presets"}


# Helper class that adheres to SOLID principles
class AudioDataRepository:
    def __init__(
        self,
        db_name: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
    ) -> None:
        """
        Initialize the repository and connect to the MongoDB database.

        :param db_name: Name of the database.
        :param username: Username for MongoDB authentication.
        :param password: Password for MongoDB authentication.
        :param host: MongoDB host url. If it is None - default is
            `mongodb+srv://{username}:{password}@cluster0.witcg.mongodb.net`.
        """
        self._connect_to_db(db_name, username, password, host)

    def _connect_to_db(
        self, db_name: str, username: str, password: str, host: Optional[str] = None
    ) -> None:
        """
        Establishes a connection to the MongoDB database.
        """
        if host is None:
            _host = (
                f"mongodb+srv://{username}:{password}@cluster0-pri.witcg.mongodb.net"
            )
        else:
            _host = host
        connect(
            db=db_name,
            host=_host,
        )

    def save_audio_data(
        self,
        speaker_embs: np.ndarray,
        spec_norm: np.ndarray,
        speech_tokens: np.ndarray,
        prompt_tokens: np.ndarray,
        metadata: Optional[str] = None,
        _id: Optional[str] = None,
        force_insert: bool = True
    ) -> str:
        """Save the entire numpy object in a single compressed binary file."""
        compressed_data = self._compress_to_binary(
            speaker_embs=speaker_embs,
            spec_norm=spec_norm,
            speech_tokens=speech_tokens,
            prompt_tokens=prompt_tokens,
        )

        audio_data = AudioData(
            id=_id,
            data_binary=compressed_data,
            metadata=metadata
        )

        if not force_insert:
            logging.warning("Force insert disabled, voice would be updated")
        #     result = AudioData.objects(id=_id).update_one(
        #         set__data_binary=compressed_data,
        #         set__metadata=metadata,
        #         upsert=True,  # Create a new document if it doesn't exist
        #     )
        #     if not result:
        #         logging.warning("No document matched for update.")
        #         raise ValueError(f"{_id}")
        # else:
        audio_data.save(force_insert=force_insert)
        return str(audio_data.id)

    def get_audio_data_as_base64(
        self, audio_id: str
    ) -> Optional[Dict[str, Optional[str]]]:
        """
        Retrieve the single compressed binary file from the database and return it as base64.

        :param audio_id: The ID of the AudioData document to retrieve.
        :return: A dictionary with the base64-encoded data and metadata, or None if not found.
        """
        audio_data = AudioData.objects(id=audio_id).first()
        total_count_of_objects = AudioData.objects.count()
        if audio_data:
            return {
                "data": self._to_base64(audio_data.data_binary),
                "metadata": audio_data.metadata,
                "total_objects": total_count_of_objects
            }
        return None

    @staticmethod
    def _compress_to_binary(**np_data: Unpack[Union[np.ndarray, int]]) -> bytes:
        """Helper function to compress all numpy arrays into a single binary buffer."""
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **np_data)  # Saving all arrays at once
        buffer.seek(0)
        return buffer.getvalue()

    @staticmethod
    def _to_base64(binary_data: bytes) -> str:
        """Helper function to convert binary data to base64."""
        return base64.b64encode(binary_data).decode("utf-8")
