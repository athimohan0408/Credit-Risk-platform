from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import predict, eda, explainability, chat

app = FastAPI(title="Credit Risk Platform API", version="1.0.0")

# Allow CORS for local React development and production Docker
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:80", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes with sub-prefixes so paths match React API calls:
#   /api/predict            -> predict.router  /
#   /api/eda/summary        -> eda.router      /summary
#   /api/explainability/... -> explainability.router
#   /api/chat               -> chat.router     /
app.include_router(predict.router,        prefix="/api",              tags=["predict"])
app.include_router(eda.router,            prefix="/api/eda",          tags=["eda"])
app.include_router(explainability.router, prefix="/api/explainability", tags=["explainability"])
app.include_router(chat.router,           prefix="/api",              tags=["chat"])


@app.get("/health")
def health_check():
    return {"status": "healthy"}
