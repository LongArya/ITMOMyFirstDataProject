Реализация ITMO проекта MyFirstDataProject

# Инструкция по запуску приложения

```
git clone https://github.com/LongArya/ITMOMyFirstDataProject.git
cd ITMOMyFirstDataProject 
docker build -t itmo_mvp .
docker run -it --device=/dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --network host itmo_mvp bash
python3 ./mvp_entry_point.py
```

# Описание модулей приложения

Основная инфомация находится в файле report4.pdf

Ссылка на neptune проект с логами экспериментов:

https://app.neptune.ai/longarya/StaticGestureClassification/runs/details?viewId=98f483ca-a116-4e81-b3b7-5c1077f3bd4f&detailsTab=dashboard&dashboardId=98ec2178-7296-4c52-8235-3c905b10e27d&shortId=STAT-92
