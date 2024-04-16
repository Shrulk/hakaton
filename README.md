# Hackathon. Постановка задачи
## Задача
Создание прогнозной модели для детектирования угла
расхождения строп при погрузочных работах.

## Желаемый результат
MVP с функционалом:
- Принимает изображение (фото погрузки)
- Оценивает угол между строп
- Выдает результат измерения

## Критерии оценки решения
1. Понимание проблематики: Заказчика и Исполнителя (проблема бизнеса
и проблема тех. реализации)
2. Техническое решение: качество, глубина проработки и пр.
3. Креатив и инновационность
4. Качество и полнота презентации решения
5. Ответы на доп. вопросы

# Hackathon. Проект
## Структура проекта
Для решения поставленной задачи было принято решение о разбиении проекта на несколько основных структурных блоков, в рамках каждого из которых будут выделены подзадачи для программной и аналитической реализации.
### 1. Подготовка данных
Основой любого проекта, связанного с данными, является их подготовка. При решении комплексной задачи машинного обучения данные необходимо отобрать, разметить и разделить, что и легло в основу ключевых подзадач.
#### 1.1. Отбор кадров
На этом этапе в исходном видео-материале обнаруживаются фрагменты, содержащие целевые объекты. После чего видео материал разбивается на отдельные кадры, которые отправляются на дальнейшую обработку.
#### 1.2. Разметка данных
Отобранные данные размечаются для обработки нейронной сетью и переводятся в удобный формат. В качестве целевых объектов выделяются отдельные стропы, а также контур контейнера.
#### 1.3. Увеличение кадровой выборки
Ограниченное количество кадров в выборке для повышения эффективности дальнейшего обучения модели увеличивается методами вариации контраста, яркости и размытия. Разметка сохраняется путем применения к ней геометрических преобразований, аналогичных преобразованиям изображений.
#### 1.4. Подготовка наборов
Отобранные данные разбиваются на наборы для обучения, тестирования и валидации.
### 2. Выделение целевых объектов
В данном модуле проекта отобранные сырые данные обретают математические очертания.
#### 2.2. Нейро-модуль
![](images/model.png){: width="100"}
В основе проекта лежит нейросетевой модуль, подобный Fast RCCN. Модель будет обучаться на обучающей выборке, а затем применяться к тестовым данным для поиска целевых объектов – строп, на которых подвешен контейнер. Выделенные прямоугольные сегменты со стропами должны поступать на вход следующему блоку.
#### 2.3. CV-модуль
Модуль компьютерного зрения на основе пакетов open-cv применяется для поиска на выделенных участках изображения со тропой точной линии, по которой проходит стропа. Методы компьютерного зрения, в частности пространственная фильтрация, способны справиться с этой задачей достаточно эффективно и быстро, сохранив при этом вычислительные ресурсы.
### 3. Расчетный блок
Заключительный алгебраическо-геометрический блок по определенным линиям находит точки пересечения и углы между стропами. В итоге, именно этот блок выносит вердикт по проделанной работе.
## Прогресс
### Общие задачи
 - Произведен обзор методов
 - Отобраны наиболее релевантные методы
 - Осуществлена аналитическая работа и оценка
### Подготовка данных
 - Реализован отбор кадров для обработки из тестового фильма
 - Отобрана выборка из подходящих для представления модели
 - Осуществлено расширение выборки методом варьирования контраста и яркости
 - Осуществлена разметка
 - Сет разбит на Train, Test и Validation
### Выделение целевых объектов
 - Первая модель сети создана и первично обучена, сделаны выводы о необходимости усовершенствования обучающей выборки
![](images/plot_1.jpg)
![](images/plot_2.jpg)
 - Создана грубая модель CV блока
### Расчетный блок
 - Создан прототип вычислителя и определителя углов
## Задачи
 - Усовершенствование модели RCNN 
 - Доработка СV блока
 - Организация API между сегментами проекта
 - Первые этапы обучения показали, что некоторые модификации, примененные к данным для увеличения их количества, сказались негативно на качестве данных. В связи с этим необходимо отказаться от части сгенерированных образцов
