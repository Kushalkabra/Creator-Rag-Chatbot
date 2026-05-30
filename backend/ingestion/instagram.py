import re
from pathlib import Path
from urllib.parse import urlparse

import instaloader
import whisper
import yt_dlp

# Instagram does not offer a public transcript API (unlike YouTube captions).
# We download audio with yt-dlp, then run OpenAI Whisper locally to get text
# and word-level timestamps — same role as youtube-transcript-api on YouTube.

AUDIO_PATH = Path("/tmp/reel_audio.mp3")
HASHTAG_PATTERN = re.compile(r"#[\w]+")

_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def _extract_shortcode(url: str) -> str:
    """Pull Instagram post/reel shortcode from common URL formats."""
    parsed = urlparse(url.strip())
    path_parts = [part for part in parsed.path.split("/") if part]

    for index, part in enumerate(path_parts):
        if part in ("reel", "p", "tv") and index + 1 < len(path_parts):
            return path_parts[index + 1]

    match = re.search(r"instagram\.com/(?:reel|p|tv)/([^/?#]+)", url)
    if match:
        return match.group(1)

    raise ValueError(f"Could not extract Instagram shortcode from URL: {url}")


def _download_audio(url: str) -> Path:
    """Download reel audio only to /tmp/reel_audio.mp3 via yt-dlp."""
    AUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    if AUDIO_PATH.exists():
        AUDIO_PATH.unlink()

    ydl_opts = {
        "format": "bestaudio",
        "quiet": True,
        "no_warnings": True,
        "outtmpl": str(AUDIO_PATH.with_suffix("")),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not AUDIO_PATH.exists():
        candidates = list(AUDIO_PATH.parent.glob(f"{AUDIO_PATH.stem}.*"))
        if candidates:
            candidates[0].rename(AUDIO_PATH)

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(f"Audio file not created at {AUDIO_PATH}")

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

    post = instaloader.Post.from_shortcode(loader.context, shortcode)
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
    shortcode = _extract_shortcode(url)
    metadata = _fetch_post_metadata(shortcode)

    audio_path = _download_audio(url)
    try:
        transcript, transcript_segments = _transcribe_audio(audio_path)
    finally:
        if audio_path.exists():
            audio_path.unlink()

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
