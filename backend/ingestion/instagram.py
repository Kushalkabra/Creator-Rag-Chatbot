import os
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import instaloader
import whisper
import yt_dlp

from ingestion.errors import InvalidInstagramURLError, InstagramIngestionError

# Instagram does not offer a public transcript API (unlike YouTube captions).
# We download audio with yt-dlp, then run OpenAI Whisper locally to get text
# and word-level timestamps — same role as youtube-transcript-api on YouTube.

AUDIO_PATH = Path(tempfile.gettempdir()) / "reel_audio.mp3"
HASHTAG_PATTERN = re.compile(r"#[\w]+")
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*m")

_whisper_model = None


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_PATTERN.sub("", text)


def _find_ffmpeg_dir() -> str | None:
    """Locate ffmpeg bin directory even if the server started before PATH was updated."""
    env_path = os.getenv("FFMPEG_PATH")
    if env_path:
        candidate = Path(env_path)
        if (candidate / "ffmpeg.exe").exists() or (candidate / "ffmpeg").exists():
            return str(candidate)
        if candidate.name in ("ffmpeg", "ffmpeg.exe") and candidate.exists():
            return str(candidate.parent)

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        return str(Path(ffmpeg_bin).parent)

    if os.name == "nt":
        search_roots = [
            Path.home() / "AppData/Local/Microsoft/WinGet/Packages",
            Path("C:/ffmpeg/bin"),
            Path("C:/Program Files/ffmpeg/bin"),
        ]
        for root in search_roots:
            if not root.exists():
                continue
            for ffmpeg_exe in root.rglob("ffmpeg.exe"):
                ffprobe_exe = ffmpeg_exe.with_name("ffprobe.exe")
                if ffprobe_exe.exists():
                    return str(ffmpeg_exe.parent)

    return None


def _ensure_ffmpeg_available() -> str:
    ffmpeg_dir = _find_ffmpeg_dir()
    if ffmpeg_dir:
        _add_ffmpeg_to_path(ffmpeg_dir)
        return ffmpeg_dir
    raise InstagramIngestionError(
        "ffmpeg is not installed or not on PATH. "
        "Install from https://ffmpeg.org/download.html, then restart your terminal and backend."
    )


def _add_ffmpeg_to_path(ffmpeg_dir: str) -> None:
    """Whisper invokes ffmpeg via subprocess and expects it on PATH."""
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if ffmpeg_dir not in path_entries:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _ensure_ffmpeg_available()
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def _extract_shortcode(url: str) -> str:
    """Pull Instagram post/reel shortcode from common URL formats."""
    parsed = urlparse(url.strip())
    path_parts = [part for part in parsed.path.split("/") if part]

    for index, part in enumerate(path_parts):
        if part in ("reel", "reels", "p", "tv") and index + 1 < len(path_parts):
            return path_parts[index + 1]

    match = re.search(r"instagram\.com/(?:reels?|p|tv)/([^/?#]+)", url)
    if match:
        return match.group(1)

    raise InvalidInstagramURLError(f"Invalid Instagram URL: {url}")


def _download_audio(url: str) -> Path:
    """Download reel audio only to a temp mp3 file via yt-dlp."""
    ffmpeg_dir = _ensure_ffmpeg_available()
    AUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    if AUDIO_PATH.exists():
        AUDIO_PATH.unlink()

    ydl_opts = {
        "format": "bestaudio",
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": ffmpeg_dir,
        "outtmpl": str(AUDIO_PATH.with_suffix("")),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        message = _strip_ansi(str(exc)).lower()
        if "ffmpeg" in message or "ffprobe" in message:
            raise InstagramIngestionError(
                "ffmpeg is not installed or not on PATH. "
                "Install from https://ffmpeg.org/download.html, then restart your terminal and backend."
            ) from exc
        raise

    if not AUDIO_PATH.exists():
        candidates = list(AUDIO_PATH.parent.glob(f"{AUDIO_PATH.stem}.*"))
        if candidates:
            candidates[0].rename(AUDIO_PATH)

    if not AUDIO_PATH.exists():
        raise InstagramIngestionError("Could not download Instagram reel audio")

    return AUDIO_PATH


def _transcribe_audio(audio_path: Path) -> tuple[str, list[dict]]:
    """Transcribe audio with Whisper; return full text and word-level segments."""
    model = _get_whisper_model()
    result = model.transcribe(str(audio_path), word_timestamps=True)

    transcript = (result.get("text") or "").strip()
    segments: list[dict] = []

    for segment in result.get("segments", []):
        words = segment.get("words")
        if words:
            for word in words:
                start = float(word.get("start", 0))
                end = float(word.get("end", start))
                segments.append(
                    {
                        "text": word.get("word", "").strip(),
                        "start": start,
                        "duration": max(end - start, 0),
                    }
                )
        else:
            start = float(segment.get("start", 0))
            end = float(segment.get("end", start))
            segments.append(
                {
                    "text": segment.get("text", "").strip(),
                    "start": start,
                    "duration": max(end - start, 0),
                }
            )

    return transcript, segments


def _fetch_post_metadata(shortcode: str) -> dict:
    """Fetch reel/post stats and creator info with instaloader."""
    loader = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        quiet=True,
    )

    try:
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
    except instaloader.exceptions.InstaloaderException as exc:
        raise InvalidInstagramURLError("Invalid Instagram URL") from exc
    caption = post.caption or ""

    try:
        comments = int(post.comments)
    except Exception:
        comments = 0

    views = int(post.video_view_count or 0)
    likes = int(post.likes)

    owner_username = post.owner_username
    followers = 0
    try:
        profile = post.owner_profile
        followers = int(profile.followers)
    except Exception:
        pass

    upload_date = ""
    if post.date_utc:
        upload_date = post.date_utc.isoformat()

    duration = float(post.video_duration or 0)

    return {
        "likes": likes,
        "views": views,
        "comments": comments,
        "caption": caption,
        "creator": owner_username,
        "creator_followers": followers,
        "upload_date": upload_date,
        "duration": duration,
    }


def _compute_engagement_rate(views: int, likes: int, comments: int) -> float:
    if views <= 0:
        return 0.0
    return round((likes + comments) / views * 100, 2)


def get_instagram_data(url: str, video_label: str) -> dict:
    """
    Fetch transcript (via audio + Whisper) and metadata for an Instagram reel.

    video_label: typically "A" or "B" for comparison workflows.
    """
    try:
        shortcode = _extract_shortcode(url)
        metadata = _fetch_post_metadata(shortcode)

        audio_path = _download_audio(url)
        try:
            transcript, transcript_segments = _transcribe_audio(audio_path)
        finally:
            if audio_path.exists():
                audio_path.unlink()
    except InvalidInstagramURLError:
        raise
    except Exception as exc:
        message = str(exc).lower()
        if "instagram" in message or "shortcode" in message or "url" in message:
            raise InvalidInstagramURLError("Invalid Instagram URL") from exc
        raise InstagramIngestionError(
            f"Instagram ingestion failed: {exc}"
        ) from exc

    caption = metadata["caption"]
    hashtags = HASHTAG_PATTERN.findall(caption)
    engagement_rate = _compute_engagement_rate(
        metadata["views"],
        metadata["likes"],
        metadata["comments"],
    )

    return {
        "video_label": video_label,
        "url": url,
        "transcript": transcript,
        "transcript_segments": transcript_segments,
        "views": metadata["views"],
        "likes": metadata["likes"],
        "comments": metadata["comments"],
        "engagement_rate": engagement_rate,
        "creator": metadata["creator"],
        "hashtags": hashtags,
        "upload_date": metadata["upload_date"],
        "duration": metadata["duration"],
    }
