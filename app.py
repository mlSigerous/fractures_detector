"""Streamlit-приложение для предварительного поиска переломов на рентгене."""

from io import BytesIO
from pathlib import Path
import hashlib
import os

import requests
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from inference import FractureDetector, InferenceResult


APP_ROOT = Path(__file__).resolve().parent
MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_MODEL_SIZE = 1024 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 50_000_000


def get_setting(name: str) -> str | None:
    """Читает настройку из окружения или Streamlit Secrets."""
    if value := os.getenv(name):
        return value
    try:
        value = st.secrets.get(name)
    except FileNotFoundError:
        return None
    return str(value) if value else None


def find_local_model() -> Path | None:
    """Ищет веса в местах, используемых training notebook и приложением."""
    candidates = []
    if configured_path := get_setting("MODEL_PATH"):
        candidates.append(Path(configured_path).expanduser())

    candidates.extend(
        [
            APP_ROOT / "models" / "fracture_detector_best.pt",
            APP_ROOT / "fracture_detector" / "fracture_detector_best.pt",
            APP_ROOT / "fracture_detector_best.pt",
        ]
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


@st.cache_resource(show_spinner="Загружаем веса модели…")
def download_model(model_url: str) -> Path:
    """Один раз скачивает публичный .pt-файл во временное хранилище."""
    cache_dir = Path("/tmp/fracture_detector")
    cache_dir.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.sha256(model_url.encode()).hexdigest()[:12]
    download_path = cache_dir / f"model-{url_hash}.pt"
    temporary_path = download_path.with_suffix(".part")

    if download_path.is_file() and download_path.stat().st_size > 1_000_000:
        return download_path

    downloaded_size = 0
    try:
        with requests.get(model_url, stream=True, timeout=(10, 180)) as response:
            response.raise_for_status()
            declared_size = int(response.headers.get("content-length", 0))
            if declared_size > MAX_MODEL_SIZE:
                raise ValueError("Размер файла модели превышает 1 ГБ")

            with temporary_path.open("wb") as model_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    downloaded_size += len(chunk)
                    if downloaded_size > MAX_MODEL_SIZE:
                        raise ValueError("Размер файла модели превышает 1 ГБ")
                    model_file.write(chunk)
    except requests.RequestException as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("Не удалось скачать веса по MODEL_URL") from error
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    if downloaded_size < 1_000_000:
        temporary_path.unlink(missing_ok=True)
        raise ValueError("Скачанный файл слишком мал и не похож на веса модели")

    temporary_path.replace(download_path)
    return download_path


def resolve_model_path() -> Path:
    if local_model := find_local_model():
        return local_model
    if model_url := get_setting("MODEL_URL"):
        return download_model(model_url)
    raise FileNotFoundError(
        "Не найдены веса fracture_detector_best.pt и не задан MODEL_URL."
    )


@st.cache_resource(show_spinner="Инициализируем детектор…")
def load_detector(model_path: str) -> FractureDetector:
    return FractureDetector(model_path=model_path, image_size=768)


def read_uploaded_image(file_bytes: bytes) -> Image.Image:
    if not file_bytes:
        raise ValueError("Загружен пустой файл")
    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError("Размер изображения превышает 20 МБ")

    try:
        image = Image.open(BytesIO(file_bytes))
        image.verify()
        image = Image.open(BytesIO(file_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("Файл не является корректным изображением") from error

    if min(image.size) < 128:
        raise ValueError("Изображение слишком маленькое: минимальная сторона — 128 px")
    return image


def render_result(result: InferenceResult) -> None:
    if result.fracture_detected:
        st.warning(
            "Модель обнаружила области с возможными признаками перелома. "
            "Необходима оценка врача-рентгенолога."
        )
    else:
        st.success(
            "При выбранном пороге модель не обнаружила подозрительных областей. "
            "Это не исключает наличие перелома."
        )

    image_column, details_column = st.columns([3, 2], gap="large")
    with image_column:
        st.image(
            result.annotated_image,
            caption="Результат детекции",
            use_container_width=True,
        )
    with details_column:
        st.metric("Найдено областей", len(result.detections))
        st.metric("Максимальная уверенность", f"{result.max_confidence:.1%}")
        st.caption(f"Время инференса: {result.inference_time_ms / 1000:.2f} с")

        if result.detections:
            st.markdown("#### Найденные области")
            for index, detection in enumerate(result.detections, start=1):
                st.write(
                    f"{index}. Уверенность **{detection.confidence:.1%}**, "
                    f"координаты `({detection.x1}, {detection.y1}) — "
                    f"({detection.x2}, {detection.y2})`"
                )


st.set_page_config(
    page_title="Детектор переломов",
    page_icon="🩻",
    layout="wide",
)

st.title("Детектор переломов по рентгеновскому снимку")
st.write(
    "Загрузите рентген в формате JPG, PNG или WEBP. "
    "Модель отметит области с возможными признаками перелома."
)
st.caption("Загруженные снимки не сохраняются приложением на диск.")

with st.expander("Важная информация", expanded=False):
    st.markdown(
        "Приложение предназначено для учебных и исследовательских целей. "
        "Результат модели не является медицинским диагнозом и не заменяет "
        "заключение квалифицированного специалиста."
    )

try:
    model_path = resolve_model_path()
except (FileNotFoundError, RuntimeError, ValueError) as error:
    st.error(f"Модель пока недоступна: {error}")
    st.info(
        "Положите веса в `models/fracture_detector_best.pt` либо задайте "
        "публичную прямую ссылку `MODEL_URL` в Streamlit Secrets."
    )
    st.stop()

confidence = st.slider(
    "Минимальная уверенность модели",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05,
    help="Чем ниже значение, тем больше подозрительных областей покажет модель.",
)
uploaded_file = st.file_uploader(
    "Рентгеновский снимок",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    try:
        uploaded_image = read_uploaded_image(file_bytes)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    st.image(uploaded_image, caption="Загруженный снимок", width=520)

    if st.button("Проанализировать снимок", type="primary", use_container_width=True):
        try:
            detector = load_detector(str(model_path))
            with st.spinner("Модель анализирует снимок…"):
                result = detector.predict(uploaded_image, confidence=confidence)
        except Exception as error:
            st.error(f"Не удалось выполнить предсказание: {error}")
        else:
            st.session_state["prediction"] = result
            st.session_state["prediction_key"] = (file_hash, confidence)

    if st.session_state.get("prediction_key") == (file_hash, confidence):
        render_result(st.session_state["prediction"])
    elif "prediction" in st.session_state:
        st.info("Изображение или порог изменены — запустите анализ повторно.")
