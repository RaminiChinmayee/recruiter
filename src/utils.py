import json
import os
import logging


logging.basicConfig(
    level=logging.INFO,
    format=
    "%(asctime)s - %(levelname)s - %(message)s"
)


def save_json(data,path):

    folder=os.path.dirname(path)

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
            indent=4
        )



def load_json(path):

    if not os.path.exists(path):
        return {}

    with open(
        path,
        encoding="utf-8"
    ) as f:

        return json.load(f)



def clean_text(text):

    if not text:
        return ""

    return (
        text
        .replace("\x00","")
        .strip()
    )