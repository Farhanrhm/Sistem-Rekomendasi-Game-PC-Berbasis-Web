"""Module for training, feature extraction, and serializing TF-IDF recommendation artifacts.

This module ingests processed Steam game catalog metadata, performs text
preprocessing and platform noise reduction, extracts textual features using
Scikit-Learn's TfidfVectorizer, computes sparse representation matrices,
and serializes runtime artifacts (.pkl) into the models directory for in-memory
inference in the LevelFind Flask web application.
"""

from typing import Set, Tuple
import os
import re
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# ==============================================================================
# NOISE REDUCTION: FILTER PLATFORM & ACCESSIBILITY NON-CONTENT TAGS
# ==============================================================================
PLATFORM_FEATURE_TAGS: Set[str] = {
    # Platform & Licensing Tags
    "steam cloud", "steam achievements", "steam trading cards",
    "surround sound", "stereo sound", "cross-platform multiplayer",
    "steam workshop", "full controller support", "partial controller support",
    "remote play on phone", "remote play on tablet", "remote play on tv",
    "remote play together", "steam turn-notifications", "steam leaderboards",
    "captions available", "commentary available", "includes level editor",
    "hdr available", "tracked controller support", "family sharing",
    # Technical & Accessibility Non-Content Tags
    "custom volume controls", "playable without timed input", "save anytime",
    "mouse only option", "keyboard only option", "touch only option",
    "camera comfort", "adjustable difficulty", "adjustable text size",
    "subtitle options", "dualshock controller support", "dualsense controller support",
    "color alternatives", "stats", "shared/split screen", "vr only",
    "gamepad recommended", "in-app purchases", "shared/split screen pvp",
    "shared/split screen co-op", "#category_playable_at_your_own_pace"
}


def clean_platform_tags(tags_str: str) -> str:
    """Filter out non-content, platform, and hardware-specific tags from Steam metadata.

    Args:
        tags_str (str): Semicolon-delimited tag string directly extracted from Steam.

    Returns:
        str: Semicolon-delimited string containing only meaningful gameplay/genre tags.
    """
    if not tags_str or not isinstance(tags_str, str):
        return ""
    tags = [t.strip() for t in tags_str.split(';') if t.strip()]
    pure_tags = [t for t in tags if t.lower() not in PLATFORM_FEATURE_TAGS]
    return "; ".join(pure_tags)


def clean_html_and_urls(text: str) -> str:
    """Sanitize raw text by stripping URLs, HTML tags, and non-alphanumeric characters.

    Args:
        text (str): Raw textual description or synopsis.

    Returns:
        str: Normalized, lowercase alphanumeric text without formatting tags.
    """
    if not isinstance(text, str):
        return ""
    # Strip web hyperlinks (HTTP/HTTPS/WWW)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Strip HTML tags
    text = re.sub(r'<[^>]*>', ' ', text)
    # Retain standard alphanumeric words and whitespaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.lower().strip()


def combine_features(row: pd.Series) -> str:
    """Combine text features (descriptions, genres, and filtered tags) into a single corpus.

    Args:
        row (pd.Series): Single row of DataFrame representing a game entry.

    Returns:
        str: Concatenated textual representation ready for TF-IDF tokenization.
    """
    raw_desc = f"{row.get('short_description', '')} {row.get('detailed_description', '')}"
    desc = clean_html_and_urls(raw_desc)
    genres_val = str(row.get('genres', ''))
    tags_val = str(row.get('tags', ''))
    genres = genres_val.replace(';', ' ') if genres_val != 'nan' else ""
    tags = tags_val.replace(';', ' ') if tags_val != 'nan' else ""
    return f"{desc} {genres} {tags}".strip()


def check_is_sparse(row: pd.Series) -> bool:
    """Determine whether a game possesses an insufficiently descriptive text corpus.

    Used by the recommendation engine to apply soft penalties against sparse
    metadata, preventing short or blank descriptions from ranking artificially high.

    Args:
        row (pd.Series): Row of game metadata containing descriptions, genres, and tags.

    Returns:
        bool: True if descriptions are missing or the total token count is under 15 words.
    """
    short_d = str(row.get('short_description', '')).strip()
    detail_d = str(row.get('detailed_description', '')).strip()
    both_empty = (short_d in ['', 'nan']) and (detail_d in ['', 'nan'])

    desc_text = short_d if (short_d and short_d != 'nan') else detail_d
    if desc_text == 'nan':
        desc_text = ""
    genres = str(row.get('genres', '')).replace(';', ' ') if str(row.get('genres', '')).strip() != 'nan' else ""
    tags = str(row.get('tags', '')).replace(';', ' ') if str(row.get('tags', '')).strip() != 'nan' else ""
    full_text = f"{desc_text} {genres} {tags}".strip()
    word_cnt = len([w for w in full_text.split() if w])
    return bool(both_empty or word_cnt < 15)


def load_and_preprocess_dataset(data_path: str) -> pd.DataFrame:
    """Load raw dataset and apply structural normalization and tag filtering.

    Args:
        data_path (str): Relative or absolute filesystem path to the dataset CSV.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with synthesized feature columns.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File dataset {data_path} tidak ditemukan. Jalankan smart_scraper.py terlebih dahulu.")

    print(f"Membaca dataset dari: {data_path}...")
    df = pd.read_csv(data_path)

    # Ensure required string columns exist
    for col in ['clean_desc', 'detailed_description', 'genres', 'tags', 'short_description']:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].fillna('')

    # Ensure numerical metrics are sanitized and typed
    for col in ['positive_reviews', 'total_reviews', 'rating_score', 'price']:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Clean non-content platform tags
    df['tags'] = df['tags'].apply(clean_platform_tags)

    # Generate combined text corpus
    print("Menggabungkan fitur teks (deskripsi, genre, tags)...")
    df['combined_features'] = df.apply(combine_features, axis=1)

    # Mark sparse corpus flags for tie-breaking/penalty
    df['is_sparse_corpus'] = df.apply(check_is_sparse, axis=1)

    return df


def extract_tfidf_features(
    corpus: pd.Series,
    stop_words: str = 'english',
    min_df: int = 2,
    max_df: float = 0.50
) -> Tuple[csr_matrix, TfidfVectorizer]:
    """Fit TfidfVectorizer on corpus and return sparse matrix and vectorizer model.

    Args:
        corpus (pd.Series): Series containing concatenated game feature texts.
        stop_words (str, optional): Language stop words collection. Defaults to 'english'.
        min_df (int, optional): Minimum document frequency threshold. Defaults to 2.
        max_df (float, optional): Maximum document frequency threshold to cut universal terms. Defaults to 0.50.

    Returns:
        Tuple[csr_matrix, TfidfVectorizer]: Sparse CSR TF-IDF matrix and fitted vectorizer.
    """
    print("Melakukan ekstraksi fitur teks gabungan dengan TfidfVectorizer (Scikit-Learn)...")
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, min_df=min_df, max_df=max_df)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    return tfidf_matrix, tfidf_vectorizer


def save_model_artifacts(
    df: pd.DataFrame,
    tfidf_matrix: csr_matrix,
    tfidf_vectorizer: TfidfVectorizer,
    output_dir: str = 'models'
) -> None:
    """Serialize and export data models and index mappings as pickle artifacts.

    Args:
        df (pd.DataFrame): Processed DataFrame containing game metadata.
        tfidf_matrix (csr_matrix): Sparse CSR matrix representing TF-IDF weights.
        tfidf_vectorizer (TfidfVectorizer): Fitted Scikit-Learn TfidfVectorizer instance.
        output_dir (str, optional): Destination directory for .pkl files. Defaults to 'models'.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Menyimpan file model .pkl ke folder '{output_dir}'...")

    # 1. Save Sparse TF-IDF Matrix
    matrix_path = os.path.join(output_dir, 'tfidf_matrix.pkl')
    with open(matrix_path, 'wb') as f:
        pickle.dump(tfidf_matrix, f)

    # 2. Save TF-IDF Vectorizer
    vectorizer_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    # 3. Clean and Save Metadata DataFrame
    cols_to_keep = [
        'steam_appid', 'name', 'price', 'genres', 'header_image',
        'short_description', 'detailed_description', 'rating',
        'positive_reviews', 'total_reviews', 'rating_score', 'tags', 'is_sparse_corpus'
    ]
    for col in cols_to_keep:
        if col not in df.columns:
            if col in ['positive_reviews', 'total_reviews', 'rating_score', 'price']:
                df[col] = 0
            else:
                df[col] = ''

    df_clean = df[cols_to_keep].copy()
    game_data_path = os.path.join(output_dir, 'game_data.pkl')
    with open(game_data_path, 'wb') as f:
        pickle.dump(df_clean, f)

    # 4. Save Title-to-Index Lookup Series
    indices = pd.Series(df_clean.index, index=df_clean['name'].str.lower()).drop_duplicates()
    indices_path = os.path.join(output_dir, 'indices.pkl')
    with open(indices_path, 'wb') as f:
        pickle.dump(indices, f)

    print(f"Bentuk Matriks TF-IDF Sparse: {tfidf_matrix.shape}")
    print(f"Total Game Terindeks: {len(df_clean)}")
    print("[OK] --- PROSES PEMODELAN IN-MEMORY (.PKL) SELESAI ---")


def build_pipeline(
    data_path: str = 'dataset/processed/steam_new_and_fav_final_1.csv',
    output_dir: str = 'models'
) -> None:
    """Execute end-to-end training and artifact generation pipeline.

    Args:
        data_path (str, optional): Path to input dataset CSV.
        output_dir (str, optional): Target folder for serialized models.

    Returns:
        None
    """
    df_processed = load_and_preprocess_dataset(data_path)
    tfidf_matrix, tfidf_vectorizer = extract_tfidf_features(df_processed['combined_features'])
    save_model_artifacts(df_processed, tfidf_matrix, tfidf_vectorizer, output_dir=output_dir)


if __name__ == '__main__':
    build_pipeline()