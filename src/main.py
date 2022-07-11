"""
Launch the dashboard with:

streamlit run /path/to/main.py

if using poetry, then execute

poetry run streamlit run path/to/main.py
"""

from k2_oai.dashboard import run

if __name__ == "__main__":
    run()
