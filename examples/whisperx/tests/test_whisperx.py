import tempfile
from pathlib import Path

import ffmpeg
from nos.client import Client
from nos.test.utils import NOS_TEST_AUDIO, get_benchmark_audio  # noqa: F401
from rich import print


def trim_audio(audio_path: Path, duration_s: int = 600) -> Path:
    with tempfile.NamedTemporaryFile(suffix=Path(audio_path).suffix, delete=False) as tmp:
        audio_trimmed = ffmpeg.input(str(audio_path)).audio.filter("atrim", duration=duration_s)
        audio_output = ffmpeg.output(audio_trimmed, tmp.name)
        ffmpeg.run(audio_output, overwrite_output=True)
        return Path(tmp.name)


def test_whisperx_transcribe_audio_file():
    from rich.pretty import pretty_repr

    # Create a client
    client = Client("[::]:50051")
    assert client.WaitForServer()

    # Load the custom whisperx model
    model = client.Module("m-bain/whisperx-large-v2", shm=True)

    # Upload local audio path to server and execute inference
    # on the remote path. Note that the audio file is deleted
    # from the server after the inference is complete via the
    # context manager.
    audio_path = get_benchmark_audio()  # NOS_TEST_AUDIO

    # Read the first 600s of the audio file and write it to
    # a temporary file with the same file extension as the
    # original audio file.
    print("\n[green]Trimming audio ...[/green]")
    audio_path = trim_audio(audio_path, duration_s=600)
    print(f"[green]Uploading {audio_path} to server...[/green]")
    with client.UploadFile(audio_path) as remote_path:
        response = model.transcribe(path=remote_path, batch_size=96)
    audio_path.unlink()
    assert isinstance(response, dict)
    print(pretty_repr(response["segments"]))
