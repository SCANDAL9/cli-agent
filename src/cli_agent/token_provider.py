"""
OAuth2 Access Token Provider for Google Cloud Vertex AI.
Provides thread-safe in-memory caching and non-blocking asynchronous token refresh.
"""

import asyncio
import os
import time


class TokenProvider:
    """
    Manages Google Cloud OAuth2 access tokens with in-memory TTL caching.
    Prevents event-loop blocking by using asynchronous subprocess execution.
    """

    def __init__(self, ttl_buffer_seconds: int = 300, default_ttl_seconds: int = 3600):
        self._cached_token: str | None = None
        self._expires_at: float = 0.0
        self._ttl_buffer_seconds = ttl_buffer_seconds
        self._default_ttl_seconds = default_ttl_seconds
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        """
        Retrieves a valid OAuth2 access token.
        Returns cached token if still valid, otherwise fetches and caches a new one.
        """
        # Allow environment variable override (e.g. in CI or containerized environments)
        env_token = os.getenv("GCLOUD_ACCESS_TOKEN") or os.getenv("VERTEX_TOKEN")
        if env_token:
            return env_token.strip()

        async with self._lock:
            now = time.time()
            if self._cached_token and now < (self._expires_at - self._ttl_buffer_seconds):
                return self._cached_token

            token = await self._fetch_token()
            self._cached_token = token
            self._expires_at = now + self._default_ttl_seconds
            return token

    async def _fetch_token(self) -> str:
        """
        Executes 'gcloud auth print-access-token' asynchronously without blocking the event loop.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "gcloud",
                "auth",
                "print-access-token",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
        except FileNotFoundError as err:
            raise RuntimeError(
                "The 'gcloud' CLI is not found in PATH. "
                "Ensure Google Cloud SDK is installed or set GCLOUD_ACCESS_TOKEN."
            ) from err
        except Exception as e:
            raise RuntimeError(f"Unexpected error running gcloud: {e}") from e

        if proc.returncode != 0:
            err_msg = stderr.decode().strip() if stderr else f"Exit code {proc.returncode}"
            raise RuntimeError(f"Failed to obtain gcloud token: {err_msg}")

        token = stdout.decode().strip()
        if not token:
            raise RuntimeError("gcloud returned an empty access token.")

        return token

    def clear_cache(self) -> None:
        """Flushes the cached token, forcing a refresh on the next call."""
        self._cached_token = None
        self._expires_at = 0.0
