try:
    from .config import get_datasets_directory, get_log_directory
except ImportError:
    from config import get_datasets_directory, get_log_directory

DATASETS_DIRECTORY = get_datasets_directory()
LOG_DIRECTORY = get_log_directory()
