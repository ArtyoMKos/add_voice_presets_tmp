import os
import uuid
import argparse
import pandas as pd


def create_csv(audio_dir, output_csv="output.csv"):
    """
    Generate a CSV file containing metadata for audio files in a given directory.

    Args:
        audio_dir (str): Path to the directory containing audio files.
        output_csv (str): Path to save the output CSV file. Defaults to "output.csv".
    """
    # Define column headers
    columns = [
        "voice_id",
        "speaker_id",
        "task_id",
        "gender",
        "prompt_audio_path",
        "prompt_text",
        "prompt_audio_name",
    ]

    # Collect data for each audio file
    data = []
    for file_name in os.listdir(audio_dir):
        file_path = os.path.join(audio_dir, file_name)

        if os.path.isfile(file_path):
            file_name_without_ext, file_ext = os.path.splitext(file_name)

            if file_ext.lower() in [".wav", ".mp3", ".flac"]:  # Add supported audio file extensions
                data.append([
                    str(uuid.uuid4()),  # voice_id
                    None,               # speaker_id
                    None,               # task_id
                    None,               # gender
                    os.path.abspath(file_path),  # prompt_audio_path
                    None,               # prompt_text
                    file_name_without_ext,  # prompt_audio_name
                ])

    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)

    print(f"CSV file generated successfully: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Generate CSV metadata for audio files in a directory.")
    parser.add_argument("audio_dir", help="Path to the directory containing audio files.")
    parser.add_argument("--output", default="output.csv", help="Path to save the generated CSV file.")
    args = parser.parse_args()

    create_csv(args.audio_dir, args.output)


if __name__ == "__main__":
    main()
