# PIN_proj
Credit prediction ML model  
API взаимодействие с моделью

## Запуск сервиса titanic-service

```bash
# Сборка Docker-образа
docker build -t pin:latest .
```
```bash
# Запуск контейнера
docker run -d --name pin -p 5000:5000 pin:latest
```
