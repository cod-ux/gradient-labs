import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List


class FileIOUtil:
    """Utility functions for file input/output operations."""
    
    @staticmethod
    def read_json(file_path: Path) -> Any:
        """Read JSON file and return data."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def write_json(data: Any, file_path: Path, indent: int = 2) -> None:
        """Write data to JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
    
    @staticmethod
    def read_excel(file_path: Path, **kwargs) -> pd.DataFrame:
        """Read Excel file and return DataFrame."""
        return pd.read_excel(file_path, **kwargs)
    
    @staticmethod
    def write_excel(df: pd.DataFrame, file_path: Path, **kwargs) -> None:
        """Write DataFrame to Excel file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(file_path, **kwargs)
    
    @staticmethod
    def ensure_directory(directory: Path) -> None:
        """Ensure directory exists, create if it doesn't."""
        directory.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def list_files_with_pattern(directory: Path, pattern: str) -> List[Path]:
        """List all files in directory matching pattern."""
        return list(directory.glob(pattern))