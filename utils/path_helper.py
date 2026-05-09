import os

def get_project_root():
    """Returns the absolute path to the project root directory."""
    # Current file is in project_root/utils/path_helper.py
    # So we go up 2 levels
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_absolute_path(relative_path):
    """
    Converts a relative path from the project root to an absolute path.
    Also handles converting old D:/Project_PlantDisease/ paths to relative ones.
    """
    # If it's an old Windows path, clean it up
    if "Project_PlantDisease" in relative_path:
        parts = relative_path.split("Project_PlantDisease")[-1]
        # Remove leading slashes/backslashes
        relative_path = parts.lstrip("\\/")

    return os.path.join(get_project_root(), relative_path)
