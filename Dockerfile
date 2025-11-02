FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir  -r requirements.txt
EXPOSE $PORT
CMD gunicorn -w 4 -b 0.0.0.0:$PORT app:app