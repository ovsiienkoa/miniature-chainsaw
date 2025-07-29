FROM python:3.12-slim

WORKDIR /app

COPY requirements_gradio.txt .
RUN pip install --no-cache-dir -r requirements_gradio.txt

COPY . .

EXPOSE 7860

CMD ["python", "gradio_app.py"]
