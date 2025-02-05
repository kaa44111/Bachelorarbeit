import sys
import os

# Fügen Sie den Pfad zu Ihrem Projekt hinzu
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Abschlussarbeit_DL'))
if project_path not in sys.path:
    sys.path.append(project_path)

print(f"PYTHONPATH is set to: {sys.path}")