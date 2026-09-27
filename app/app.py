"""
Streamlit Web Application entry-point redirect.
Runs the primary Stage 10 application defined in app/streamlit_app.py.
"""

from pathlib import Path
import runpy

if __name__ == "__main__":
    target = Path(__file__).parent / "streamlit_app.py"
    runpy.run_path(str(target), run_name="__main__")
