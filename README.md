# Fracture detector

Streamlit-приложение для предварительного поиска областей возможного перелома
на рентгеновском снимке. Результат модели не является медицинским диагнозом.

## Локальный запуск

1. Установите зависимости:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Поместите обученные веса в один из поддерживаемых путей:

   ```text
   models/fracture_detector_best.pt
   fracture_detector/fracture_detector_best.pt
   fracture_detector_best.pt
   ```

3. Запустите приложение из корня проекта:

   ```bash
   streamlit run app.py
   ```

## Streamlit Community Cloud

1. Загрузите проект в GitHub.
2. Создайте приложение на Streamlit Community Cloud с entrypoint `app.py`.
3. Выберите Python 3.12.
4. Передайте веса одним из способов:

   - добавьте `models/fracture_detector_best.pt` в репозиторий;
   - либо загрузите веса в публичное файловое хранилище и добавьте в Secrets:

     ```toml
     MODEL_URL = "https://example.com/fracture_detector_best.pt"
     ```

   Ссылка должна вести непосредственно на файл, без HTML-страницы
   предпросмотра. Для локального нестандартного расположения можно использовать
   переменную окружения `MODEL_PATH`.

Итоговые веса из `train_model.py` сохраняются как
`fracture_detector/fracture_detector_best.pt`, поэтому после завершения обучения
их можно использовать без переименования.
