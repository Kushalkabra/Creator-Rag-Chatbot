"""Shared ingestion errors with user-facing messages."""


class IngestionError(Exception):
    """Base error for video ingestion failures."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class YouTubeAPIError(IngestionError):
    pass


class YouTubeTranscriptError(IngestionError):
    pass


class InvalidYouTubeURLError(IngestionError):
    pass


class InvalidInstagramURLError(IngestionError):
    pass


class InstagramIngestionError(IngestionError):
    pass


class InstagramTimeoutError(IngestionError):
    pass
