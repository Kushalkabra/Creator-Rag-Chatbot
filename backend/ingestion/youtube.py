import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

HASHTAG_PATTERN = re.compile(r"#[\w]+")
ISO8601_DURATION_PATTERN = re.compile(
    r"^PT(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?$"
)


def _extract_video_id(url: str) -> str:
    """Pull the 11-character video ID from common YouTube URL shapes."""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower().replace("www.", "")

    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]
        if video_id:
            return video_id

    if host in ("youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
            if video_id:
                return video_id

        for prefix in ("/shorts/", "/embed/", "/v/"):
            if parsed.path.startswith(prefix):
                video_id = parsed.path[len(prefix) :].split("/")[0]
                if video_id:
                    return video_id

    match = re.search(r"(?:v=|/)([\w-]{11})(?:\?|&|/|$)", url)
    if match:
        return match.group(1)

    raise ValueError(f"Could not extract video ID from URL: {url}")


def _parse_duration_seconds(iso_duration: str) -> int:
    """Convert YouTube ISO 8601 duration (e.g. PT4M13S) to total seconds."""
    match = ISO8601_DURATION_PATTERN.match(iso_duration or "")
    if not match:
        return 0

    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return hours * 3600 + minutes * 60 + seconds


def _fetch_transcript(video_id: str) -> tuple[str, list[dict]]:
    """Return full transcript text and timestamped segments, or empty on failure."""
    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join(segment["text"].strip() for segment in segments)
        return text, segments
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return "", []
    except Exception:
        return "", []


def _fetch_video_metadata(video_id: str) -> dict:
    """Call YouTube Data API v3 for stats and snippet fields."""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY is not set in environment")

    youtube = build("youtube", "v3", developerKey=api_key)
    response = (
        youtube.videos()
        .list(part="snippet,statistics,contentDetails", id=video_id)
        .execute()
    )

    items = response.get("items", [])
    if not items:
        raise ValueError(f"No video found for ID: {video_id}")

    item = items[0]
    snippet = item.get("snippet", {})
    statistics = item.get("statistics", {})
    content_details = item.get("contentDetails", {})

    views = int(statistics.get("viewCount", 0))
    likes = int(statistics.get("likeCount", 0))
    comment_count = int(statistics.get("commentCount", 0))

    return {
        "views": views,
        "likes": likes,
        "comment_count": comment_count,
        "channel_title": snippet.get("channelTitle", ""),
        "published_at": snippet.get("publishedAt", ""),
        "description": snippet.get("description", ""),
        "duration_seconds": _parse_duration_seconds(
            content_details.get("duration", "")
        ),
    }


def _compute_engagement_rate(views: int, likes: int, comment_count: int) -> float:
    if views <= 0:
        return 0.0
    return round((likes + comment_count) / views * 100, 2)


def get_youtube_data(url: str, video_label: str) -> dict:
    """
    Fetch transcript and metadata for a YouTube video.

    video_label: typically "A" or "B" for comparison workflows.
    """
    video_id = _extract_video_id(url)
    transcript, transcript_segments = _fetch_transcript(video_id)
    metadata = _fetch_video_metadata(video_id)

    description = metadata["description"]
    hashtags = HASHTAG_PATTERN.findall(description)
    engagement_rate = _compute_engagement_rate(
        metadata["views"],
        metadata["likes"],
        metadata["comment_count"],
    )

    return {
        "video_label": video_label,
        "url": url,
        "video_id": video_id,
        "transcript": transcript,
        "transcript_segments": transcript_segments,
        "views": metadata["views"],
        "likes": metadata["likes"],
        "comment_count": metadata["comment_count"],
        "channel_title": metadata["channel_title"],
        "published_at": metadata["published_at"],
        "description": description,
        "duration_seconds": metadata["duration_seconds"],
        "hashtags": hashtags,
        "engagement_rate": engagement_rate,
    }
