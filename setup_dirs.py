import os

project_dirs = [
    "notebooks",
    "data/raw",
    "data/processed",
    "src/data_preprocessing/audio",
    "src/data_preprocessing/text",
    "src/data_preprocessing/fusion",
    "src/feature_extraction/audio",
    "src/feature_extraction/text",
    "src/models/audio",
    "src/models/text",
    "src/models/fusion",
    "src/train",
    "src/evaluation",
    "src/utils",
]

for d in project_dirs:
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "__init__.py"), "a").close()
