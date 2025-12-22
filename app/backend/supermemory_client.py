from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


class SupermemoryError(Exception):
    """
    Raised when the Supermemory API responds with an error or an unexpected payload.
    """

    def __init__(self, message: str, status_code: Optional[int] = None, response_text: Optional[str] = None) -> None:
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(message)


class SupermemoryClient:
    """
    Synchronous client for the Supermemory API.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.supermemory.ai", timeout_s: float = 10) -> None:
        self.api_key = api_key or os.getenv("SUPERMEMORY_API_KEY")
        if not self.api_key:
            raise ValueError("Supermemory API key is required. Set SUPERMEMORY_API_KEY or pass api_key.")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _drop_none(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in payload.items() if value is not None}

    def _sanitize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if metadata is None:
            return None
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a flat dictionary.")
        clean: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (dict, list, tuple, set)):
                raise ValueError("metadata must be flat (no nested structures).")
            if value is not None:
                clean[str(key)] = value
        return clean

    def _request(self, method: str, path: str, json_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not path.startswith("/"):
            path = f"/{path}"
        url = f"{self.base_url}{path}"
        try:
            response = httpx.request(method=method, url=url, headers=self._headers(), json=json_body, timeout=self.timeout_s)
        except httpx.RequestError as exc:
            raise SupermemoryError(f"Request error for {method.upper()} {path}: {exc}") from exc

        if response.status_code < 200 or response.status_code >= 300:
            raise SupermemoryError(
                f"{method.upper()} {path} failed with status {response.status_code}: {response.text}",
                status_code=response.status_code,
                response_text=response.text,
            )

        try:
            return response.json()
        except ValueError as exc:
            raise SupermemoryError(
                f"Failed to parse JSON response for {method.upper()} {path}: {response.text}",
                status_code=response.status_code,
                response_text=response.text,
            ) from exc

    def add_document(self, content: str, container_tag: Optional[str] = None, custom_id: Optional[str] = None,metadata: Optional[Dict[str, Any]] = None,) -> Dict[str, Any]:
        if not content:
            raise ValueError("content is required for add_document.")
        payload = self._drop_none(
            {
                "content": content,
                "containerTag": container_tag,
                "customId": custom_id,
                "metadata": self._sanitize_metadata(metadata),
            }
        )
        return self._request("POST", "/v3/documents", payload)

    def search_memories(self, q: str, container_tag: Optional[str] = None,
        threshold: float = 0.6,
        limit: int = 5,
        rerank: bool = False,
        search_mode: str = "memories",
        filters: Optional[Dict[str, Any]] = None,
        include: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not q:
            raise ValueError("q is required for search_memories.")
        if not 0 <= float(threshold) <= 1:
            raise ValueError("threshold must be between 0 and 1.")
        if not 1 <= int(limit) <= 100:
            raise ValueError("limit must be between 1 and 100.")
        payload = self._drop_none(
            {
                "q": q,
                "containerTag": container_tag,
                "threshold": float(threshold),
                "limit": int(limit),
                "rerank": bool(rerank),
                "searchMode": search_mode,
                "filters": filters,
                "include": include,
            }
        )
        return self._request("POST", "/v4/search", payload)

    def update_document(
        self,
        document_id: str,
        *,
        content: Optional[str] = None,
        container_tag: Optional[str] = None,
        custom_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not document_id:
            raise ValueError("document_id is required for update_document.")
        if all(value is None for value in (content, container_tag, custom_id, metadata)):
            raise ValueError("At least one field must be provided to update_document.")
        payload = self._drop_none(
            {
                "content": content,
                "containerTag": container_tag,
                "customId": custom_id,
                "metadata": self._sanitize_metadata(metadata),
            }
        )
        return self._request("PATCH", f"/v3/documents/{document_id}", payload)

    def update_memory(
        self,
        container_tag: str,
        *,
        memory_id: Optional[str] = None,
        content_exact: Optional[str] = None,
        new_content: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not container_tag:
            raise ValueError("container_tag is required for update_memory.")
        if not memory_id and not content_exact:
            raise ValueError("Either memory_id or content_exact must be provided for update_memory.")
        if not new_content:
            raise ValueError("new_content is required for update_memory.")
        payload = self._drop_none(
            {
                "containerTag": container_tag,
                "id": memory_id,
                "content": content_exact,
                "newContent": new_content,
                "metadata": self._sanitize_metadata(metadata),
            }
        )
        return self._request("PATCH", "/v4/memories", payload)

    def forget_memory(
        self,
        container_tag: str,
        *,
        memory_id: Optional[str] = None,
        content_exact: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not container_tag:
            raise ValueError("container_tag is required for forget_memory.")
        if not memory_id and not content_exact:
            raise ValueError("Either memory_id or content_exact must be provided for forget_memory.")
        payload = self._drop_none(
            {
                "containerTag": container_tag,
                "id": memory_id,
                "content": content_exact,
                "reason": reason,
            }
        )
        return self._request("DELETE", "/v4/memories", payload)
