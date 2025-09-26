# -------------------------
# 1. Base image
# -------------------------
FROM python:3.11-bookworm

# -------------------------
# 2. Set working directory
# -------------------------
WORKDIR /opt

# -------------------------
# 3. Install dependencies
# -------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# 4. Copy project source
# -------------------------
COPY ./mlops ./mlops
COPY ./data-prepare ./data-prepare

# -------------------------
# 5. Expose FastAPI port
# -------------------------
EXPOSE 8000

# -------------------------
# 6. Run FastAPI server
#    (webapp.py 안에 app = FastAPI() 있어야 함)
# -------------------------
CMD ["uvicorn", "mlops.src.webapp:app", "--host", "0.0.0.0", "--port", "8000"]
