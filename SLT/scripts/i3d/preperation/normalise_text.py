"""use this to normalise How2Sign text files (Removes punctuation and uppercase but keep "'" for words like "don't")
1.input csv files from dataset VIDEO_ID	VIDEO_NAME	SENTENCE_ID	SENTENCE_NAME	START_REALIGNED	END_REALIGNED	SENTENCE
2.output csv with extra row with normalised text VIDEO_ID	VIDEO_NAME	SENTENCE_ID	SENTENCE_NAME	START_REALIGNED	END_REALIGNED	SENTENCE SENTENCE_NORMALIZED
3. save as new csv file how2sign\how2sign_realigned_<split>_normalised.csv
4.use normalised text for training

example call: python normalise_text.py "G:\My Drive\AAI_project\manifests\SLT\how2sign\how2sign_realigned_train.csv"
"""

import pandas as pd
import re
import argparse
from pathlib import Path


def normalize_text(text):
    """
    Normalize text by:
    - Converting to lowercase
    - Removing punctuation except apostrophes (')
    - Removing extra whitespace
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove all punctuation except apostrophes
    # Keep apostrophes for contractions like "don't", "can't", etc.
    text = re.sub(r"[^\w\s']", "", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def process_csv(input_csv_path, output_csv_path=None):
    """
    Process the CSV file and add normalized text column.
    
    Args:
        input_csv_path: Path to input CSV file
        output_csv_path: Path to output CSV file (optional)
    """
    # Read the CSV file
    print(f"Reading CSV from: {input_csv_path}")
    input_path = Path(input_csv_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv_path}")
    if input_path.is_dir():
        raise IsADirectoryError(f"Input path is a directory, not a file: {input_csv_path}")

    try:
        df = pd.read_csv(input_csv_path, sep='\t')
    except OSError as exc:
        if getattr(exc, "errno", None) == 22:
            print("Primary read failed (Errno 22). Retrying with Python engine...")
            df = pd.read_csv(input_csv_path, sep='\t', engine='python')
        else:
            raise
    
    # Check if SENTENCE column exists
    if 'SENTENCE' not in df.columns:
        raise ValueError("CSV file must contain a 'SENTENCE' column")
    
    # Apply normalization to the SENTENCE column
    print("Normalizing text...")
    df['SENTENCE_NORMALIZED'] = df['SENTENCE'].apply(normalize_text)
    
    # Determine output path if not provided
    if output_csv_path is None:
        # Replace .csv with _normalised.csv
        output_csv_path = input_path.parent / f"{input_path.stem}_normalised{input_path.suffix}"
    
    # Save to new CSV file
    print(f"Saving normalized CSV to: {output_csv_path}")
    df.to_csv(output_csv_path, sep='\t', index=False)
    
    print(f"Done! Processed {len(df)} rows.")
    print(f"\nExample normalization:")
    for idx in range(min(3, len(df))):
        print(f"  Original: {df['SENTENCE'].iloc[idx]}")
        print(f"  Normalized: {df['SENTENCE_NORMALIZED'].iloc[idx]}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Normalize text in How2Sign CSV files"
    )
    parser.add_argument(
        "input_csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output CSV file (optional, defaults to input_normalised.csv)",
        default=None
    )
    
    args = parser.parse_args()
    
    process_csv(args.input_csv, args.output)


if __name__ == "__main__":
    main()



