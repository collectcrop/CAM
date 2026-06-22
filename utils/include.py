try:
    from .config import get_datasets_directory
except ImportError:
    from config import get_datasets_directory

DATASETS_DIRECTORY = get_datasets_directory()
LOG_DIRECTORY = "/mnt/home/zwshi/learned-index/cost-model/visualize/data/log/"