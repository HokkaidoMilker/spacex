# Knowledge Base Documents

This directory stores knowledge base documents for the SpaceX stock analysis RAG assistant.

## Supported Formats
- PDF (`.pdf`)
- Text (`.txt`)
- Markdown (`.md`)

## Suggested Document Types

| Document Type | Suggested Sources |
|---------------|-------------------|
| SpaceX Launch Records | Wikipedia export / SpaceX official news |
| Starlink User & Revenue Reports | Analyst reports, Bloomberg news |
| SpaceX Valuation & Funding History | Crunchbase, PitchBook exports |
| Competitor Comparison Reports | Industry research PDFs |
| Government Contract Announcements | NASA.gov, DoD press releases |
| Musk/Management Public Speeches | Financial media articles |

## Usage
Place your documents here, then run `python ingest.py` to index them into the Chroma vector database.
