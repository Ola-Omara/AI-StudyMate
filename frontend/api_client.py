import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 60


class BackendUnavailableError(Exception):
    pass


class BackendRequestError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Backend returned {status_code}: {detail}")


def get_api_base_url() -> str:
    return API_BASE_URL


def check_health() -> dict:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException as exc:
        raise BackendUnavailableError(str(exc)) from exc
    if response.status_code != 200:
        raise BackendRequestError(response.status_code, response.text)
    return response.json()


def ask_question(question: str) -> dict:
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as exc:
        raise BackendUnavailableError(str(exc)) from exc

    if response.status_code == 422:
        raise BackendRequestError(422, "Question was empty or invalid.")
    if response.status_code != 200:
        raise BackendRequestError(response.status_code, response.text)

    return response.json()
