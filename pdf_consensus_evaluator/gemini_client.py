"""Gemini multimodal client supporting thinking budget controls and direct API execution."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("pdf_consensus_evaluator.client")


class GeminiClient:
    """Multimodal Gemini API client with thinking budget configuration and retry resilience."""

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ):
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("CLOUD_ML_API_KEY")
        )
        self.oauth_token = (
            os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN")
            or os.environ.get("ACCESS_TOKEN")
            or self._resolve_gcp_access_token()
        )
        self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("VERTEX_PROJECT")
        self.location = os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("VERTEX_LOCATION", "us-central1")
        self.base_url = base_url or os.environ.get("GEMINI_API_BASE_URL") or self.DEFAULT_BASE_URL
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def _resolve_gcp_access_token(self) -> Optional[str]:
        """Attempts to dynamically obtain an OAuth access token from the environment."""
        try:
            import google.auth
            import google.auth.transport.requests
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            auth_req = google.auth.transport.requests.Request()
            credentials.refresh(auth_req)
            return credentials.token
        except Exception:
            pass

        # Fallback to gcloud if available in GCP environment
        try:
            import subprocess
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass

        return None

    def encode_pdf(self, pdf_path: str) -> Tuple[str, str]:
        """Encodes PDF file to base64 string and returns (base64_data, mime_type)."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")
        with open(pdf_path, "rb") as f:
            data = f.read()
        b64_str = base64.b64encode(data).decode("utf-8")
        return b64_str, "application/pdf"

    def _build_endpoint_and_headers(self, model: str) -> Tuple[str, Dict[str, str]]:
        """Constructs the appropriate API endpoint URL and authentication headers."""
        headers = {"Content-Type": "application/json"}

        # Vertex AI endpoint if project_id is configured
        if self.project_id:
            endpoint = (
                f"https://{self.location}-aiplatform.googleapis.com/v1/projects/"
                f"{self.project_id}/locations/{self.location}/publishers/google/models/{model}:generateContent"
            )
            if self.oauth_token:
                headers["Authorization"] = f"Bearer {self.oauth_token}"
            elif self.api_key:
                headers["x-goog-api-key"] = self.api_key
            return endpoint, headers

        # Standard Developer / AI Studio endpoint
        if self.api_key:
            endpoint = f"{self.base_url}/{model}:generateContent?key={self.api_key}"
            headers["x-goog-api-key"] = self.api_key
            return endpoint, headers

        if self.oauth_token:
            endpoint = f"{self.base_url}/{model}:generateContent"
            headers["Authorization"] = f"Bearer {self.oauth_token}"
            return endpoint, headers

        raise ValueError(
            "Authentication required: Please set the GEMINI_API_KEY environment variable, "
            "provide an OAuth access token, or log in via Google Cloud Application Default Credentials."
        )

    def generate_content(
        self,
        model: str,
        system_instruction: Optional[str],
        prompt: str,
        pdf_path: Optional[str] = None,
        pdf_b64: Optional[str] = None,
        thinking_budget: int = 0,
        temperature: float = 0.1,
        response_json: bool = True,
    ) -> Dict[str, Any]:
        """Synchronous generation call with retries and structured payload handling."""
        endpoint, headers = self._build_endpoint_and_headers(model)

        parts: List[Dict[str, Any]] = []

        if pdf_path:
            b64_data, mime_type = self.encode_pdf(pdf_path)
            parts.append({
                "inlineData": {
                    "mimeType": mime_type,
                    "data": b64_data,
                }
            })
        elif pdf_b64:
            parts.append({
                "inlineData": {
                    "mimeType": "application/pdf",
                    "data": pdf_b64,
                }
            })

        parts.append({"text": prompt})

        generation_config: Dict[str, Any] = {
            "temperature": temperature,
        }

        if response_json:
            generation_config["responseMimeType"] = "application/json"

        # Thinking configuration
        if thinking_budget is not None:
            generation_config["thinkingConfig"] = {
                "thinkingBudget": thinking_budget
            }

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": generation_config,
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        raise ValueError("No candidates returned from Gemini API response.")
                    content_parts = candidates[0].get("content", {}).get("parts", [])
                    text = "".join([p.get("text", "") for p in content_parts])
                    return self._clean_and_parse_json(text)
                
                if response.status_code in (429, 500, 502, 503, 504):
                    logger.warning(
                        "Gemini API returned status %d (attempt %d/%d). Retrying...",
                        response.status_code,
                        attempt,
                        self.max_retries,
                    )
                    time.sleep(self.backoff_factor ** attempt)
                    continue
                
                response.raise_for_status()

            except (requests.RequestException, ValueError) as err:
                last_error = err
                logger.warning(
                    "Network/Parsing error on attempt %d/%d: %s",
                    attempt,
                    self.max_retries,
                    str(err),
                )
                if attempt < self.max_retries:
                    time.sleep(self.backoff_factor ** attempt)

        raise RuntimeError(
            f"Failed to generate content after {self.max_retries} attempts: {last_error}"
        )

    async def generate_content_async(
        self,
        model: str,
        system_instruction: Optional[str],
        prompt: str,
        pdf_path: Optional[str] = None,
        pdf_b64: Optional[str] = None,
        thinking_budget: int = 0,
        temperature: float = 0.1,
        response_json: bool = True,
    ) -> Dict[str, Any]:
        """Asynchronous wrapper for generate_content running in worker thread."""
        return await asyncio.to_thread(
            self.generate_content,
            model=model,
            system_instruction=system_instruction,
            prompt=prompt,
            pdf_path=pdf_path,
            pdf_b64=pdf_b64,
            thinking_budget=thinking_budget,
            temperature=temperature,
            response_json=response_json,
        )

    def _clean_and_parse_json(self, raw_text: str) -> Dict[str, Any]:
        """Strips markdown code blocks if present and parses JSON safely."""
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
            text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
            text = text.strip()
        return json.loads(text)
