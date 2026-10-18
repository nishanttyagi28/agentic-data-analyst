FROM python:3.12-slim

# hf spaces run the container as uid 1000, so the app dir must be writable
# by that user (sqlite and chroma write under ./data)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

COPY --chown=user . .

# hf spaces expects the app on 7860
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
