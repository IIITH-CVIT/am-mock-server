FROM python:3.13-slim

# libgl1/libglib2.0-0: opencv needs these at runtime even in "headless" mode.
# cmake/g++/make/libopenblas-dev: dlib compiles from source at pip-install time
# (needed for the 128-dim dlib embedding enrolled alongside mobilefacenet).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    cmake \
    g++ \
    make \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# dlib compiles from source here — expect ~10-15 min on the first build. Cached
# afterwards, so later rebuilds (code-only changes) are fast.
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# config.yaml, models/, and data/ are bind-mounted at runtime (see compose.yml)
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
