Есть два варианта запуска:
--dataset small-set - использует данные из data/data-set-classification
--dataset big-set - использует данные из data/data-set-objectDetection
Для small-set:
Использует данные напрямую из data/data-set-classification
Создает маппинг классов на основе существующей структуры
Не требует конвертации
Для big-set:
Проверяет наличие конвертированных данных в data/data_classification
Если данных нет, автоматически конвертирует из YOLO формата
Если данные уже есть, использует существующие