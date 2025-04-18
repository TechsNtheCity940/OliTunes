from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
from typing import List, Optional
import os

# 🔧 CONFIG
ROOT_DIR = "F:/newrepos/olitunes"

app = FastAPI(title="Local Codebase API")

# 📦 Data Models
class FileEntry(BaseModel):
    path: str
    content: str

class AnalyzeRequest(BaseModel):
    files: List[FileEntry]

class AnalysisResult(BaseModel):
    path: str
    summary: str
    unnecessary: Optional[bool] = False


# 📁 List all files recursively
@app.get("/files", response_model=List[str], tags=["Files"])
def list_files():
    all_files = []
    for root, _, files in os.walk(ROOT_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, ROOT_DIR)
            all_files.append(rel_path.replace("\\", "/"))
    return all_files


# 📄 Get contents of a file
@app.get("/files/{file_path:path}", response_model=str, tags=["Files"])
def get_file_content(file_path: str = Path(..., description="Relative file path from root")):
    abs_path = os.path.join(ROOT_DIR, file_path)
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="File not found")
    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# 🤖 Analyze codebase: summarize and identify unnecessary files
@app.post("/analyze", response_model=List[AnalysisResult], tags=["Analysis"])
def analyze_codebase(data: AnalyzeRequest):
    results = []
    for entry in data.files:
        summary = ""
        unnecessary = False

        # Naive heuristics for demo purposes
        if entry.path.endswith(".md"):
            summary = "This appears to be a markdown documentation file."
        elif "test" in entry.path.lower():
            summary = "This looks like a test file."
        elif "main" in entry.content or "app" in entry.content:
            summary = "This is likely a main application file."
        elif len(entry.content.strip()) == 0:
            summary = "Empty file — possibly unnecessary."
            unnecessary = True
        else:
            summary = "General source code file."

        results.append(AnalysisResult(path=entry.path, summary=summary, unnecessary=unnecessary))

    return results


# 🛠️ Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
