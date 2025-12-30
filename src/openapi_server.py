#!/usr/bin/env python3
"""
OpenAPI server wrapper for LibreOffice MCP tools.

Exposes MCP tool functions as HTTP endpoints for OpenAPI-based clients
like OpenWebUI tool integrations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import libremcp


class CreateDocumentRequest(BaseModel):
    path: str = Field(description="Full path where the document should be created")
    doc_type: str = Field(default="writer", description="writer, calc, impress, draw")
    content: str = Field(default="", description="Initial document content")


class ReadDocumentRequest(BaseModel):
    path: str = Field(description="Path to the document file")


class ConvertDocumentRequest(BaseModel):
    source_path: str = Field(description="Source document path")
    target_path: str = Field(description="Target document path")
    target_format: str = Field(description="Target format (pdf, docx, txt, etc.)")


class GetDocumentInfoRequest(BaseModel):
    path: str = Field(description="Path to the document file")


class ReadSpreadsheetRequest(BaseModel):
    path: str = Field(description="Path to the spreadsheet file")
    sheet_name: Optional[str] = Field(default=None, description="Specific sheet name")
    max_rows: int = Field(default=100, description="Maximum rows to read")


class InsertTextRequest(BaseModel):
    path: str = Field(description="Path to the document file")
    text: str = Field(description="Text to insert")
    position: str = Field(default="end", description="start, end, or replace")


class SearchDocumentsRequest(BaseModel):
    query: str = Field(description="Text to search for")
    search_path: Optional[str] = Field(default=None, description="Directory to search in")


class BatchConvertRequest(BaseModel):
    source_dir: str = Field(description="Directory containing source documents")
    target_dir: str = Field(description="Directory for converted documents")
    target_format: str = Field(description="Target format for conversion")
    source_extensions: Optional[List[str]] = Field(
        default=None,
        description="List of source file extensions to convert",
    )


class MergeDocumentsRequest(BaseModel):
    document_paths: List[str] = Field(description="List of document paths to merge")
    output_path: str = Field(description="Path for the merged document")
    separator: str = Field(default="\n\n---\n\n", description="Separator between docs")


class DocumentStatisticsRequest(BaseModel):
    path: str = Field(description="Path to the document file")


class OpenDocumentRequest(BaseModel):
    path: str = Field(description="Path to the document to open")
    readonly: bool = Field(default=False, description="Open in read-only mode")


class RefreshDocumentRequest(BaseModel):
    path: str = Field(description="Path to the document to refresh")

3000
class WatchDocumentRequest(BaseModel):
    path: str = Field(description="Path to the document to watch")
    duration_seconds: int = Field(default=30, description="Watch duration in seconds")


class LiveEditingSessionRequest(BaseModel):
    path: str = Field(description="Path to the document for live editing")
    auto_refresh: bool = Field(default=True, description="Enable auto refresh hints")


def _http_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


app = FastAPI(
    title="LibreOffice MCP OpenAPI",
    version="1.0.0",
    description="OpenAPI wrapper for LibreOffice MCP tool functions.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/tools/create_document", response_model=libremcp.DocumentInfo)
def create_document_endpoint(req: CreateDocumentRequest):
    try:
        return libremcp.create_document(req.path, req.doc_type, req.content)
    except Exception as exc:
        raise _http_exception(exc)


@app.post("/tools/read_document_text", response_model=libremcp.TextContent)
def read_document_text_endpoint(req: ReadDocumentRequest):
    try:
        return libremcp.read_document_text(req.path)
    except Exception as exc:
        raise _http_exception(exc)


@app.post("/tools/convert_document", response_model=libremcp.ConversionResult)
def convert_document_endpoint(req: ConvertDocumentRequest):
    try:
        return libremcp.convert_document(req.source_path, req.target_path, req.target_format)
    except Exception as exc:
        raise _http_exception(exc)


@app.post("/tools/get_document_info", response_model=libremcp.DocumentInfo)
def get_document_info_endpoint(req: GetDocumentInfoRequest):
    try:
        return libremcp.get_document_info(req.path)
    except Exception as exc:
        raise _http_exception(exc)


@app.post("/tools/read_spreadsheet_data", response_model=libremcp.SpreadsheetData)
def read_spreadsheet_data_endpoint(req: ReadSpreadsheetRequest):
    try:
        return libremcp.read_spreadsheet_data(req.path, req.sheet_name, req.max_rows)
    except Exception as exc:
        raise _http_exception(exc)


@app.post("/tools/insert_text_at_position", response_model=libremcp.DocumentInfo)
def insert_text_at_position_endpoint(req: InsertTextRequest):
    try:
        return libremcp.insert_text_at_position(req.path, req.text, req.position)
    except Exception as exc:
        raise _http_exception(exc)


@app.post("/tools/search_documents", response_model=List[Dict[str, Any]])
def search_documents_endpoint(req: SearchDocumentsRequest):
    try:
        return libremcp.search_documents(req.query, req.search_path)
    except Exception as exc:
        raise _http_exception(exc)


@app.post("/tools/batch_convert_documents", response_model=List[libremcp.ConversionResult])
def batch_convert_documents_endpoint(req: BatchConvertRequest):
    try:
        return libremcp.batch_convert_documents(
            req.source_dir,
            req.target_dir,
            req.target_format,
            req.source_extensions,
        )
    except Exception as exc:
        raise _http_exception(exc)


@app.post("/tools/merge_text_documents", response_model=libremcp.DocumentInfo)
def merge_text_documents_endpoint(req: MergeDocumentsRequest):
    try:
        return libremcp.merge_text_documents(req.document_paths, req.output_path, req.separator)
    except Exception as exc:
        raise _http_exception(exc)


@app.post("/tools/get_document_statistics", response_model=Dict[str, Any])
def get_document_statistics_endpoint(req: DocumentStatisticsRequest):
    try:
        return libremcp.get_document_statistics(req.path)
    except Exception as exc:
        raise _http_exception(exc)


@app.post("/tools/open_document_in_libreoffice", response_model=Dict[str, Any])
def open_document_in_libreoffice_endpoint(req: OpenDocumentRequest):
    try:
        return libremcp.open_document_in_libreoffice(req.path, req.readonly)
    except Exception as exc:
        raise _http_exception(exc)


@app.post("/tools/refresh_document_in_libreoffice", response_model=Dict[str, Any])
def refresh_document_in_libreoffice_endpoint(req: RefreshDocumentRequest):
    try:
        return libremcp.refresh_document_in_libreoffice(req.path)
    except Exception as exc:
        raise _http_exception(exc)


@app.post("/tools/watch_document_changes", response_model=Dict[str, Any])
def watch_document_changes_endpoint(req: WatchDocumentRequest):
    try:
        return libremcp.watch_document_changes(req.path, req.duration_seconds)
    except Exception as exc:
        raise _http_exception(exc)


@app.post("/tools/create_live_editing_session", response_model=Dict[str, Any])
def create_live_editing_session_endpoint(req: LiveEditingSessionRequest):
    try:
        return libremcp.create_live_editing_session(req.path, req.auto_refresh)
    except Exception as exc:
        raise _http_exception(exc)


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="LibreOffice MCP OpenAPI server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8001, help="Bind port")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
