# utilities.py  
# ─────────────────────────────────────────────
# Utility functions for Pantry App
# ─────────────────────────────────────────────

import os
import logging
from datetime import datetime
from PIL import Image


class utilities:
    """Centralized utility class for the PantryApp."""

    # ─────────────────────────────────────────
    # CONSTANTS
    # ─────────────────────────────────────────

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # ─────────────────────────────────────────
    # LOGGING SETUP
    # ─────────────────────────────────────────

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),                                # ← console
            logging.FileHandler("pantry_to_table.log", mode="a"),   # ← file  (append mode)
        ]
    )
    logger = logging.getLogger(__qualname__)

    # ─────────────────────────────────────────
    # CONFIG / ENV
    # ─────────────────────────────────────────

    @staticmethod
    def get_env(key: str, default=None) -> str:
        """Safely fetch an environment variable."""
        ut = utilities          # ← alias

        value = os.environ.get(key, default)
        if value is None:
            ut.logger.warning(f"Environment variable '{key}' not set.")
            
        return value

    # ─────────────────────────────────────────
    # FILE HELPERS
    # ─────────────────────────────────────────

    @staticmethod
    def ensure_dir(path: str) -> None:
        """Create directory if it doesn't exist."""
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def file_exists(path: str) -> bool:
        """Check if a file exists."""
        return os.path.isfile(path)
    
    @staticmethod
    def clear_directory(target_directory) -> str:
        """Clear/delete all the files in the specified directory"""
        status      = []
        summary     = None
        file_count  = 0
        error_count = 0

        for file in os.scandir(target_directory):
            try:
                os.remove(file.path)
                file_count += 1
                #status.append(f"Deleted: {file.name}")
            except Exception as e:
                #status.append(f"Error deleting {file.name}: {e}")
                error_count += 1

        summary = (
            f"{file_count} files successfully deleted.\n"
            f"{error_count} files not deleted due to error.\n"
        )

        if status:
            summary += "\n".join(status)

        return summary     


    # ─────────────────────────────────────────
    # DATA HELPERS
    # ─────────────────────────────────────────

    @staticmethod
    def chunk_list(lst: list, size: int) -> list:
        """Split a list into chunks of given size."""
        return [lst[i:i + size] for i in range(0, len(lst), size)]

    @staticmethod
    def flatten_list(nested: list) -> list:
        """Flatten one level of a nested list."""
        return [item for sublist in nested for item in sublist]

    @staticmethod
    def remove_duplicates(lst: list) -> list:
        """Remove duplicates while preserving order."""
        seen = set()
        return [x for x in lst if not (x in seen or seen.add(x))]

    # ─────────────────────────────────────────
    # STRING HELPERS
    # ─────────────────────────────────────────

    @staticmethod
    def safe_strip(value) -> str:
        """Strip whitespace safely, handles None."""
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    def truncate(text: str, max_len: int = 100) -> str:
        """Truncate a string with ellipsis."""
        return text if len(text) <= max_len else text[:max_len] + "..."

    # ─────────────────────────────────────────
    # DATE / TIME HELPERS
    # ─────────────────────────────────────────

    @staticmethod
    def now_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Return current datetime as formatted string."""
        return datetime.now().strftime(fmt)

    @staticmethod
    def today_str(fmt: str = "%Y-%m-%d") -> str:
        """Return today's date as formatted string."""
        return datetime.now().strftime(fmt)

    # ─────────────────────────────────────────
    # API HELPERS
    # ─────────────────────────────────────────

    @staticmethod
    def build_url(base: str, **params) -> str:
        """Build a URL with query parameters."""
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{base}?{query}"

    @staticmethod
    def handle_response(response)  -> tuple[dict | None, str]:
        """
        Standard API response handler.
        Returns (parsed JSON, status message) tuple.
        """
        ut = utilities          # ← alias

        if response.status_code == 200: 
            return response.json(), 'Successfully called API.'
        else:
            error_message = f"API Error {response.status_code}: {response.text}"
            ut.logger.error(error_message)
            return None, error_message

    @staticmethod    
    def scale_image(input_path, output_path, size=(300, 300)):

        try:
            with Image.open(input_path) as img:
                # 'Resampling.LANCZOS' provides the highest quality downscaling
                scaled_img = img.resize(size, Image.Resampling.LANCZOS)
                scaled_img.save(output_path)         
        except Exception as e:
            print(f'Error scaling {input_path}: {e}')  