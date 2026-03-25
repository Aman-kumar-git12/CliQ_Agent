from pathlib import Path
from langchain_core.documents import Document

class CliQDocumentLoader:
    """Handles loading and parsing of CliQ platform documentation files."""
    
    def __init__(self, data_dir: str = None):
        self.base_dir = Path(__file__).resolve().parent
        self.data_dir = Path(data_dir) if data_dir else self.base_dir / "data"

    def load_documents(self):
        """Loads all .txt files from the configured data directory."""
        documents = []
        if not self.data_dir.exists():
            print(f"Warning: Data directory {self.data_dir} does not exist.")
            return documents

        for file_path in self.data_dir.glob("*.txt"):
            try:
                text = file_path.read_text(encoding="utf-8")
                # Extract section name (e.g. 'home' from 'home.txt')
                section = file_path.stem 
                
                documents.append(Document(
                    page_content=text, 
                    metadata={
                        "source": file_path.name,
                        "section": section
                    }
                ))
                print(f"Loaded: {file_path.name} (Section: {section})")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
        
        return documents