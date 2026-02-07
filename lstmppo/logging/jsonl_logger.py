import json
from pathlib import Path


class JSONLLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(self.path, "a", encoding="utf-8")

    def log(self, record: dict):
        self.fp.write(json.dumps(record) + "\n")
        self.fp.flush()

    def close(self):
        self.fp.close()
