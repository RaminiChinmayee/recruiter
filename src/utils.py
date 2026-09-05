import json
import os
import logging


# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


logger = logging.getLogger(__name__)


# --------------------------------------------------
# Save JSON
# --------------------------------------------------

def save_json(data, path):

    folder = os.path.dirname(path)

    if folder:
        os.makedirs(
            folder,
            exist_ok=True
        )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


# --------------------------------------------------
# Load JSON
# --------------------------------------------------

def load_json(path):

    if not os.path.exists(path):
        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)