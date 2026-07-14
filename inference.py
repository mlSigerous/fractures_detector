"""Загрузка YOLO-модели и инференс без зависимости от интерфейса Streamlit."""

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter

import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    """Один найденный моделью регион возможного перелома."""

    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class InferenceResult:
    """Результат инференса, готовый для отображения в приложении."""

    annotated_image: Image.Image
    detections: tuple[Detection, ...]
    inference_time_ms: float

    @property
    def fracture_detected(self) -> bool:
        return bool(self.detections)

    @property
    def max_confidence(self) -> float:
        return max((item.confidence for item in self.detections), default=0.0)


class FractureDetector:
    """Потокобезопасная обёртка над Ultralytics YOLO."""

    def __init__(self, model_path: str | Path, image_size: int = 768) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Файл весов не найден: {model_path}")

        self.model_path = model_path
        self.image_size = image_size
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self._model = YOLO(str(model_path))
        self._lock = Lock()

    def predict(
        self,
        image: Image.Image,
        confidence: float = 0.25,
        iou: float = 0.70,
    ) -> InferenceResult:
        """Возвращает координаты и снимок с отмеченными областями."""
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence должен находиться между 0 и 1")

        image = image.convert("RGB")
        started_at = perf_counter()

        # Кешированная модель общая для сессий Streamlit. Lock не даёт двум
        # пользователям одновременно менять внутреннее состояние predictor.
        with self._lock:
            prediction = self._model.predict(
                source=image,
                imgsz=self.image_size,
                conf=confidence,
                iou=iou,
                device=self.device,
                verbose=False,
            )[0]

        elapsed_ms = (perf_counter() - started_at) * 1000
        detections = self._extract_detections(prediction)
        annotated_image = self._draw_detections(image, detections)

        return InferenceResult(
            annotated_image=annotated_image,
            detections=detections,
            inference_time_ms=elapsed_ms,
        )

    @staticmethod
    def _extract_detections(prediction) -> tuple[Detection, ...]:
        if prediction.boxes is None or len(prediction.boxes) == 0:
            return ()

        coordinates = prediction.boxes.xyxy.detach().cpu().tolist()
        confidences = prediction.boxes.conf.detach().cpu().tolist()

        detections = [
            Detection(
                confidence=float(confidence),
                x1=max(0, round(box[0])),
                y1=max(0, round(box[1])),
                x2=max(0, round(box[2])),
                y2=max(0, round(box[3])),
            )
            for box, confidence in zip(coordinates, confidences)
        ]
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return tuple(detections)

    @staticmethod
    def _draw_detections(
        image: Image.Image,
        detections: tuple[Detection, ...],
    ) -> Image.Image:
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        font = ImageFont.load_default(size=max(12, min(image.size) // 35))
        line_width = max(3, min(image.size) // 180)

        for index, detection in enumerate(detections, start=1):
            box = (detection.x1, detection.y1, detection.x2, detection.y2)
            label = f"Possible fracture {index}: {detection.confidence:.0%}"
            label_box = draw.textbbox((detection.x1, detection.y1), label, font=font)
            text_height = label_box[3] - label_box[1]
            text_width = label_box[2] - label_box[0]
            text_y = max(0, detection.y1 - text_height - 8)

            draw.rectangle(box, outline="#ff3b30", width=line_width)
            draw.rectangle(
                (
                    detection.x1,
                    text_y,
                    detection.x1 + text_width + 10,
                    text_y + text_height + 8,
                ),
                fill="#ff3b30",
            )
            draw.text(
                (detection.x1 + 5, text_y + 3),
                label,
                fill="white",
                font=font,
            )

        return annotated
