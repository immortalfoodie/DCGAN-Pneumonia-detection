"""Print instructions for downloading Kaggle chest X-ray datasets."""

from __future__ import annotations

from textwrap import dedent

from config import Config


def main() -> None:
    instructions = f"""
    ===============================================
    DATA DOWNLOAD INSTRUCTIONS
    ===============================================

    1) Install Kaggle CLI and configure API credentials:
       pip install kaggle
       Place kaggle.json in:
       - Linux/macOS: ~/.kaggle/kaggle.json
       - Windows: %USERPROFILE%/.kaggle/kaggle.json

    2) Download Kaggle chest X-ray pneumonia dataset:
       kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

    3) Extract the zip and place dataset in:
       {Config.DATA_DIR}
       Expected split directories:
       - train/NORMAL
       - train/PNEUMONIA
       - val/NORMAL
       - val/PNEUMONIA
       - test/NORMAL
       - test/PNEUMONIA

    Optional RSNA bounding box dataset:
    - Download challenge data from:
      https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data
    - Place CSV and related files under:
      {Config.PROJECT_ROOT / 'data' / 'rsna'}
    """
    print(dedent(instructions))


if __name__ == "__main__":
    main()
