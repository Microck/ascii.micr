# Backend API Setup

While I rebuild the frontend, you need to set up the Python backend to enable real image generation (otherwise the app will stay in "demo mode").

## 1. Prerequisites
- **Python 3.10+**
- **NVIDIA GPU** (Recommended) or CPU (slower)
- **CUDA Toolkit** (if using GPU)

## 2. Installation
Open a new terminal in `D:\GitHub2\ascii\api` and run:

```bash
cd api
pip install -r requirements-api.txt
```

## 3. Run the Server
Start the FastAPI server on port 8000:

```bash
# Using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 4. Verify
Open http://localhost:8000/docs in your browser. You should see the Swagger UI.

## Notes
- The frontend is pre-configured to look for the backend at `http://localhost:8000`.
- If you change the port, update the `NEXT_PUBLIC_API_URL` environment variable in the frontend.
