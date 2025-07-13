from snowflake.snowpark import Session
from snowflake.snowpark.functions import col
from sqlalchemy import create_engine, inspect
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import hashlib
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
from tqdm import tqdm
import time

tqdm.pandas()

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.cuda.empty_cache()
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, torch_dtype=torch.float32
).to(device)


def get_dw_engine():
    return create_engine(
        "snowflake://{user}:{password}@{account}/{database}/{schema}?warehouse={warehouse}&role={role}".format(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema="TARGET_TWITCH",
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            role=os.getenv("SNOWFLAKE_ROLE"),
        )
    )


def create_session(schema: str):
    connection_parameters = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": schema,
    }
    return Session.builder.configs(connection_parameters).create()


stg_session = create_session("STG_TWITCH")
dw_engine = get_dw_engine()


def generate_surrogate_key(input_str: str):
    try:
        hash_digest = hashlib.sha256(input_str.encode("utf-8")).hexdigest()[:15]
        return int(hash_digest, 16)
    except Exception as e:
        print(f"Error generating surrogate key for '{input_str}': {e}")
        return None


def batch_sentiment(texts):
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=128
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = F.softmax(outputs.logits, dim=1).cpu().numpy()

    results = []
    for score in scores:
        neg = score[0]
        neutral = score[1]
        pos = score[2] + 0.5 * neutral
        compound = pos - neg
        results.append({"pos": pos, "neg": neg, "compound": compound})
    return results


def scd2_merge(new_df, existing_df, attributes, id):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    new_df = new_df.copy()
    new_df["start_date"] = now
    new_df["end_date"] = pd.NaT
    new_df["is_current"] = True

    inserts = []

    for _, new_row in new_df.iterrows():
        match = existing_df[existing_df[id] == new_row[id]]

        if match.empty:
            inserts.append(new_row)
        else:
            current_row = match[match["is_current"] == True]
            if not current_row.empty:
                changed = any(
                    (pd.isna(new_row[attr]) and not pd.isna(current_row.iloc[0][attr]))
                    or (
                        not pd.isna(new_row[attr])
                        and new_row[attr] != current_row.iloc[0][attr]
                    )
                    for attr in attributes
                )
                if changed:
                    existing_df.loc[current_row.index, ["is_current", "end_date"]] = [
                        False,
                        now,
                    ]
                    inserts.append(new_row)

    insert_dfs = [
        pd.DataFrame([row]).dropna(axis=1, how="all")
        for row in inserts
        if not pd.DataFrame([row]).dropna(axis=1, how="all").empty
    ]

    updated_df = pd.concat([existing_df] + insert_dfs, ignore_index=True)
    return updated_df


def table_exists(table_name):
    inspector = inspect(dw_engine)
    return table_name in inspector.get_table_names()


def load_existing_table(table_name):
    try:
        return pd.read_sql_table(table_name, dw_engine)
    except Exception:
        print(f"Existing table '{table_name}' not found. Returning empty DataFrame.")
        return pd.DataFrame()


def save_table(
    df, table_name, mode="replace", dedup_on=None, existing_df=None, batch_size=100000
):
    try:
        for col in df.select_dtypes(include=["datetimetz"]).columns:
            df[col] = df[col].dt.tz_localize(None)

        inserted = 0
        if mode == "replace":
            df.to_sql(table_name, dw_engine, index=False, if_exists="replace")
            inserted = len(df)
            print(f"Replaced table '{table_name}' with {inserted} rows.")
        elif mode == "append":
            if dedup_on:
                if existing_df is None:
                    existing_df = load_existing_table(table_name)
                if not existing_df.empty:
                    df = df[~df[dedup_on].isin(existing_df[dedup_on])]

            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i : i + batch_size]
                batch_df.to_sql(table_name, dw_engine, index=False, if_exists="append")
                inserted += len(batch_df)
                print(f"Appended {len(batch_df)} rows to '{table_name}'.")

        return inserted
    except Exception as e:
        print(f"Error saving table '{table_name}': {e}")
        return 0


def stage_streamer():
    stg_streamer_raw = (
        stg_session.table("stg_streamer_raw")
        .select(
            col('"login"'),
            col('"display_name"'),
        )
        .dropna(subset=['"login"'])
    )
    df = stg_streamer_raw.to_pandas()
    df = df.drop_duplicates(subset="login")
    df["streamer_id"] = df["login"].apply(
        lambda x: hashlib.sha256(x.encode()).hexdigest()
    )
    df = df[["login", "display_name", "streamer_id"]]
    df["streamer_key"] = df["streamer_id"].apply(generate_surrogate_key)

    return df


def stage_user():
    stg_user_raw = stg_session.table("stg_user_raw")
    df = stg_user_raw.to_pandas()
    df = df.drop_duplicates(subset="user_id")
    df["user_key"] = df["user_id"].apply(generate_surrogate_key)
    return df


def stage_dates():
    stg_dates_raw = stg_session.table("stg_dates_raw")
    df = stg_dates_raw.to_pandas()
    df = df.drop_duplicates(subset="timestamp")
    df["date_key"] = df["timestamp"].astype(str).apply(generate_surrogate_key)
    return df


def stage_stream(streamer):
    stg_stream_raw = stg_session.table("stg_stream_raw")
    df = stg_stream_raw.to_pandas()
    df = df.rename(columns={"id": "stream_id", "game_name": "category"})
    df = df.drop_duplicates(subset=["stream_id"])
    df.dropna(subset=["stream_id", "category", "title"], inplace=True)
    df = df.merge(streamer[["login", "streamer_key"]], on="login", how="inner")
    return df


def stage_message(user, dates, stream):
    stg_message_raw = stg_session.table("stg_message_raw")
    df = stg_message_raw.to_pandas()
    df = df.rename(columns={"message": "chat", "channel": "login"})
    df.dropna(subset=["chat"], inplace=True)
    df = df.drop_duplicates(subset=["message_id"])
    df = df.merge(user[["user_id", "user_key"]], on="user_id", how="inner")
    df = df.merge(stream[["login", "stream_id"]], on="login", how="inner")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    dates["timestamp"] = pd.to_datetime(dates["timestamp"], utc=True).dt.tz_localize(
        None
    )
    df = df.merge(dates[["timestamp", "date_key"]], on="timestamp", how="inner")
    df.drop(columns=["timestamp", "login"], inplace=True)
    df = df.drop_duplicates(subset=["message_id"])

    return df


def target_storage():

    user = stage_user()
    streamer = stage_streamer()

    dates = stage_dates()
    if not dates.empty:
        existing_dates = load_existing_table("target_dates")
        save_table(
            dates,
            "target_dates",
            mode="append",
            dedup_on="date_key",
            existing_df=existing_dates,
        )

    stream = stage_stream(streamer=streamer)
    if not stream.empty:
        existing_stream = load_existing_table("target_stream")
        save_table(
            stream,
            "target_stream",
            mode="append",
            dedup_on="stream_id",
            existing_df=existing_stream,
        )

    message = stage_message(dates=dates, user=user, stream=stream)
    new_message_count = 0
    if not message.empty:
        existing_messages = load_existing_table("target_message")
        existing_message_ids = (
            set(existing_messages["message_id"])
            if not existing_messages.empty
            else set()
        )
        print(f"Existing message IDs: {len(existing_message_ids)}")
        new_messages = message[~message["message_id"].isin(existing_message_ids)]
        print(f"New messages to analyze: {new_messages.shape[0]} rows")

        if not new_messages.empty:
            new_message_count = save_table(
                df=new_messages,
                table_name="target_message",
                dedup_on="message_id",
                mode="append",
                existing_df=existing_messages,
            )
        else:
            print("No new messages to save.")

    if not table_exists("target_streamer"):
        now = datetime.now(timezone.utc)
        streamer["start_date"] = now
        streamer["end_date"] = pd.NaT
        streamer["is_current"] = True
        save_table(
            streamer,
            "target_streamer",
            mode="append",
            dedup_on="streamer_key",
        )
    else:
        existing_streamer = load_existing_table("target_streamer")
        target_streamer = scd2_merge(
            streamer, existing_streamer, ["login", "display_name"], "streamer_id"
        )
        save_table(
            target_streamer,
            "target_streamer",
            mode="append",
            dedup_on="streamer_key",
        )

    if not table_exists("target_user"):
        now = datetime.now(timezone.utc)
        user["start_date"] = now
        user["end_date"] = pd.NaT
        user["is_current"] = True
        save_table(user, "target_user", mode="append", dedup_on="user_key")
    else:
        existing_stores = load_existing_table("target_user")
        target_user = scd2_merge(
            user,
            existing_stores,
            [
                "user_display_name",
                "user_is_mod",
                "user_is_vip",
                "user_is_broadcaster",
                "user_is_subscriber",
            ],
            "user_id",
        )
        save_table(target_user, "target_user", mode="append", dedup_on="user_key")

    if new_message_count != 0:
        sentiment_df = new_messages.copy()

        sentiment_df = sentiment_df.merge(
            stream[["stream_id", "streamer_key"]], on="stream_id", how="inner"
        ).drop_duplicates()

        sentiment_df = sentiment_df.merge(
            streamer[["streamer_key", "streamer_id"]],
            on="streamer_key",
            how="inner",
        ).drop_duplicates()

        sentiment_df = sentiment_df[["chat", "streamer_id"]]

        print(
            f"Unique streamers (streamer_id): {sentiment_df['streamer_id'].nunique()}"
        )

        print(f"Missing streamer_id: {sentiment_df['streamer_id'].isna().sum()}")

        if sentiment_df["streamer_id"].isna().sum() > 0:
            print(
                f"Rows with missing streamer_id:\n{sentiment_df[sentiment_df['streamer_id'].isna()]}"
            )

        print(f"Sentiment analysis starting on {len(sentiment_df)} messages...")
        start_time = time.time()

        batch_size = 128
        texts = sentiment_df["chat"].tolist()
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        all_results = []
        for batch in tqdm(batches, desc="Analyzing Sentiment"):
            all_results.extend(batch_sentiment(batch))

        elapsed_time = time.time() - start_time
        print(f"Sentiment analysis completed in {elapsed_time:.2f} seconds.")

        sentiments_df = pd.DataFrame(all_results)
        sentiment_df = pd.concat(
            [sentiment_df.reset_index(drop=True), sentiments_df], axis=1
        )

        agg_df = (
            sentiment_df.groupby(["streamer_id"])
            .agg(
                pos_sum=("pos", "sum"),
                neg_sum=("neg", "sum"),
                compound_sum=("compound", "sum"),
                chat_count=("chat", "count"),
            )
            .reset_index()
        )
        agg_df["sponsorable"] = agg_df["compound_sum"] > 0.5
        print(agg_df.head())
        print(len(agg_df))

        existing_sent_df = load_existing_table("streamer_chat_sentiment")
        if existing_sent_df.empty:
            combined_df = agg_df.copy()
        else:
            combined_df = existing_sent_df.merge(
                agg_df,
                on="streamer_id",
                how="inner",
                suffixes=("_old", "_new"),
            )

            missing_old_columns = [
                f"{column}_old"
                for column in ["pos_sum", "neg_sum", "compound_sum", "chat_count"]
                if f"{column}_old" not in combined_df.columns
            ]

            if missing_old_columns:
                print(
                    f"Warning: Missing columns in merged DataFrame: {missing_old_columns}"
                )
                for column in missing_old_columns:
                    combined_df[column] = 0

            for column in ["pos_sum", "neg_sum", "compound_sum", "chat_count"]:
                combined_df[f"{column}_old"] = combined_df[f"{column}_old"].fillna(0)
                combined_df[f"{column}_new"] = combined_df[f"{column}_new"].fillna(0)

            combined_df["pos_sum"] = (
                combined_df["pos_sum_old"] + combined_df["pos_sum_new"]
            )
            combined_df["neg_sum"] = (
                combined_df["neg_sum_old"] + combined_df["neg_sum_new"]
            )
            combined_df["compound_sum"] = (
                combined_df["compound_sum_old"] + combined_df["compound_sum_new"]
            )
            combined_df["chat_count"] = (
                combined_df["chat_count_old"] + combined_df["chat_count_new"]
            )

            combined_df["avg_compound"] = (
                combined_df["compound_sum"] / combined_df["chat_count"]
            )
            combined_df["sponsorable"] = combined_df["compound_sum"] > 0.5

            combined_df = combined_df[
                [
                    "streamer_id",
                    "pos_sum",
                    "neg_sum",
                    "compound_sum",
                    "chat_count",
                    "sponsorable",
                ]
            ]

        save_table(
            df=combined_df,
            table_name="streamer_chat_sentiment",
            dedup_on="streamer_id",
            mode="append",
            existing_df=existing_sent_df,
        )
    else:
        print("No new message found! Skipping Sentiment")

    for tbl in [
        "stg_streamer_raw",
        "stg_user_raw",
        "stg_dates_raw",
        "stg_stream_raw",
        "stg_message_raw",
    ]:
        stg_session.sql(f"TRUNCATE TABLE {tbl}").collect()
        print(f"Truncated: {tbl}")

    stg_session.close()


# target_storage()
