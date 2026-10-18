from pathlib import Path


class AuthoredOCR:
    def __init__(self, text: str = "Hold for 12 seconds. Stop if pain develops.") -> None:
        self.identity = {"engine": "authored-ocr", "version": "1", "dpi": 400, "psm": 7}
        self.text = text
        self.crops: list[bytes] = []

    def recognize(self, image: Path) -> str:
        self.crops.append(image.read_bytes())
        return self.text
