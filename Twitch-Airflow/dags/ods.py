from snowflake.snowpark import Session
from snowflake.snowpark.types import *
from extract import get_streamers, collect_chat_async
import asyncio
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()


connection_parameters = {
    "account": os.environ["SNOWFLAKE_ACCOUNT"],
    "user": os.environ["SNOWFLAKE_USER"],
    "password": os.environ["SNOWFLAKE_PASSWORD"],
    "role": os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    "database": os.getenv("SNOWFLAKE_DATABASE", "TWITCH_DB"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA", "ODS_TWITCH"),
}


session = Session.builder.configs(connection_parameters).create()


def add_business_date(df):
    df["business_date"] = datetime.now()
    return df


def process_chat_timestamp(chat_df):
    chat_df["timestamp"] = pd.to_datetime(
        chat_df["timestamp"], errors="raise", utc=True, format="mixed"
    )
    chat_df["timestamp"] = chat_df["timestamp"].dt.floor("min")
    chat_df["timestamp"] = chat_df["timestamp"].dt.tz_localize(None)
    return chat_df


def ods_streamer_raw(stream_df):
    return stream_df[["login", "display_name", "business_date"]]


def ods_stream_raw(stream_df):
    return stream_df[
        [
            "id",
            "login",
            "viewer_count",
            "title",
            "language",
            "game_name",
            "tags",
            "business_date",
        ]
    ]


def ods_message_raw(chat_df):
    return chat_df[
        [
            "message_id",
            "user_id",
            "channel",
            "message",
            "first_message",
            "returning_chatter",
            "timestamp",
            "business_date",
        ]
    ]


def ods_user_raw(chat_df):
    return chat_df[
        [
            "user_id",
            "user_name",
            "user_display_name",
            "user_is_mod",
            "user_is_vip",
            "user_is_broadcaster",
            "user_is_subscriber",
            "business_date",
        ]
    ]


def ods_dates_raw(chat_df):
    df = chat_df.copy()
    df["Full Date"] = df["timestamp"].dt.date
    df["Year"] = df["timestamp"].dt.year
    df["Quarter"] = df["timestamp"].dt.quarter
    df["Month"] = df["timestamp"].dt.month
    df["Week"] = df["timestamp"].dt.isocalendar().week
    df["Day"] = df["timestamp"].dt.day
    df["Weekday"] = df["timestamp"].dt.weekday
    df["Hour"] = df["timestamp"].dt.hour
    df["Minute"] = df["timestamp"].dt.minute
    return df[
        [
            "timestamp",
            "Full Date",
            "Year",
            "Quarter",
            "Month",
            "Week",
            "Day",
            "Weekday",
            "Hour",
            "Minute",
            "business_date",
        ]
    ]


def load_table(name: str, df: pd.DataFrame):
    try:
        snow_df = session.create_dataframe(df)
        snow_df.write.mode("append").save_as_table(name)
        print(f"Loaded {len(df)} rows into {name}")
    except Exception as e:
        print(f"Failed to load {name}: {e}")


def load_all():
    stream_df = add_business_date(get_streamers())
    chat_df = add_business_date(asyncio.run(collect_chat_async(stream_df)))
    chat_df = process_chat_timestamp(chat_df)

    tables = {
        "ods_streamer_raw": ods_streamer_raw(stream_df),
        "ods_stream_raw": ods_stream_raw(stream_df),
        "ods_message_raw": ods_message_raw(chat_df),
        "ods_user_raw": ods_user_raw(chat_df),
        "ods_dates_raw": ods_dates_raw(chat_df),
    }

    for name, df in tables.items():
        load_table(name, df)


# load_all()
