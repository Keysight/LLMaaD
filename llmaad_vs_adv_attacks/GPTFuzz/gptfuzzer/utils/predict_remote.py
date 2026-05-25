"""Remote RoBERTa predictor — hits roberta_server.py endpoint instead of loading model."""

import json
import logging
import urllib.request
from gptfuzzer.utils.predict import Predictor


class RemoteRoBERTaPredictor(Predictor):
    def __init__(self, url="http://localhost:5100"):
        super().__init__(path=url)
        self.url = url.rstrip("/")

    def predict(self, sequences):
        if not sequences:
            return []
        data = json.dumps({"sequences": sequences}).encode()
        req = urllib.request.Request(
            f"{self.url}/predict",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                return result["predictions"]
        except Exception as e:
            logging.error(f"Remote RoBERTa predict failed: {e}")
            return [0] * len(sequences)
