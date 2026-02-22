FROM python:3.10-slim

# set folder kerja
WORKDIR /app

# copy requirements dulu supaya cache docker optimal
COPY requirements.txt .

# install dependency
RUN pip install --no-cache-dir -r requirements.txt

# copy semua file project
COPY . .

# port untuk streamlit (yang tampil di Space)
EXPOSE 7860

# jalankan FastAPI + Streamlit bersamaan
CMD bash -c "uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.port 7860 --server.address 0.0.0.0"
