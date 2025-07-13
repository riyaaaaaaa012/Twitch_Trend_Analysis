from snowflake.snowpark import Session
from snowflake.snowpark.functions import col
import pandas as pd
import re
from dotenv import load_dotenv
import os
from tqdm import tqdm

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import torch
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

from nltk.corpus import stopwords

load_dotenv()
tqdm.pandas()


def create_session():
    connection_parameters = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": "TARGET_TWITCH",
    }
    return Session.builder.configs(connection_parameters).create()


def safe_detect(text):
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def clean_text(text):
    if not text:
        return ""
    return re.sub(r"[^\w\s#]", "", text).strip().lower()


def get_stopwords(language):
    try:
        return stopwords.words(language)
    except:
        return []


def extract_keywords_keybert(text, kw_model, language, top_n=7):
    if not text:
        return []

    stop_words = get_stopwords(language)

    keywords = kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 3), stop_words=stop_words, top_n=top_n
    )
    return [kw[0] for kw in keywords] if keywords else [text]


def keybert_analysis():
    session = create_session()

    df = (
        session.table("target_stream")
        .select(col("STREAM_ID"), col("TITLE"), col("TAGS"), col("CATEGORY"))
        .to_pandas()
    )

    df.fillna("", inplace=True)
    df = df[(df["TITLE"] != "") & (df["CATEGORY"] != "") & (df["TAGS"] != "")]

    df["CATEGORY"] = df["CATEGORY"].apply(clean_text)
    df["CLEAN_TITLE"] = df["TITLE"].apply(clean_text)
    df["TAG_LIST"] = df["TAGS"].apply(
        lambda x: [t.strip() for t in x.split(",") if t.strip()]
    )
    df = df.explode("TAG_LIST").rename(columns={"TAG_LIST": "CLEAN_TAG"})
    df["CLEAN_TAG"] = df["CLEAN_TAG"].apply(clean_text)
    df = df[df["CLEAN_TAG"] != ""]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    sentence_model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2", device=device
    )
    kw_model = KeyBERT(model=sentence_model)

    tqdm.pandas(desc="Extracting keywords from tags")
    df["LANG_TAG"] = df["CLEAN_TAG"].progress_apply(safe_detect)
    df["KEYWORDS_TAGS"] = df.apply(
        lambda row: extract_keywords_keybert(
            row["CLEAN_TAG"], kw_model, row["LANG_TAG"], top_n=7
        ),
        axis=1,
    )

    tqdm.pandas(desc="Extracting keywords from titles")
    title_df = df[["CATEGORY", "CLEAN_TITLE"]].drop_duplicates()
    title_df["LANG_TITLE"] = title_df["CLEAN_TITLE"].progress_apply(safe_detect)
    title_df["KEYWORDS_TITLE"] = title_df.apply(
        lambda row: extract_keywords_keybert(
            row["CLEAN_TITLE"], kw_model, row["LANG_TITLE"], top_n=5
        ),
        axis=1,
    )

    tag_kw_expanded = df.explode("KEYWORDS_TAGS")
    tag_counts = (
        tag_kw_expanded.groupby(["CATEGORY", "KEYWORDS_TAGS"])
        .size()
        .reset_index(name="COUNT")
        .rename(columns={"KEYWORDS_TAGS": "KEYWORDS"})
    )
    tag_counts["SOURCE"] = "tags"

    title_kw_expanded = title_df.explode("KEYWORDS_TITLE")
    title_counts = (
        title_kw_expanded.groupby(["CATEGORY", "KEYWORDS_TITLE"])
        .size()
        .reset_index(name="COUNT")
        .rename(columns={"KEYWORDS_TITLE": "KEYWORDS"})
    )
    title_counts["SOURCE"] = "title"

    all_keywords = pd.concat([tag_counts, title_counts], ignore_index=True)

    top_categories = (
        df.groupby("CATEGORY")["STREAM_ID"].nunique().reset_index(name="CATEGORY_COUNT")
    )
    top_categories = top_categories.nlargest(500, "CATEGORY_COUNT")

    all_keywords = all_keywords[
        all_keywords["CATEGORY"].isin(top_categories["CATEGORY"])
    ]

    print("Selecting top 10 keywords per category per source...")
    top_keywords = (
        all_keywords.groupby(["CATEGORY", "SOURCE"])
        .apply(lambda x: x.nlargest(10, "COUNT"))
        .reset_index(drop=True)
    )

    session.write_pandas(
        top_categories.reset_index(drop=True),
        "TOP_CATEGORIES",
        auto_create_table=True,
        overwrite=True,
    )
    print("Saved top 500 categories to 'TOP_CATEGORIES'")

    session.write_pandas(
        top_keywords.reset_index(drop=True),
        "TOP_CATEGORY_KEYWORDS",
        auto_create_table=True,
        overwrite=True,
    )
    print("Saved top keywords per category to 'TOP_CATEGORY_KEYWORDS'")


keybert_analysis()
