from snowflake.snowpark import Session
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

load_dotenv()


def get_snowpark_session(schema: str) -> Session:
    return Session.builder.configs(
        {
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "role": os.getenv("SNOWFLAKE_ROLE"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": schema,
        }
    ).create()


def stage_replace_incremental():
    ods_session = get_snowpark_session("ODS_TWITCH")
    stg_session = get_snowpark_session("STG_TWITCH")

    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=10)
    window_end = now + timedelta(minutes=10)

    start_str = window_start.strftime("%Y-%m-%d %H:%M:%S")
    end_str = window_end.strftime("%Y-%m-%d %H:%M:%S")

    tables = {
        "ods_streamer_raw": "stg_streamer_raw",
        "ods_stream_raw": "stg_stream_raw",
        "ods_message_raw": "stg_message_raw",
        "ods_user_raw": "stg_user_raw",
        "ods_dates_raw": "stg_dates_raw",
    }

    for ods_table, stg_table in tables.items():
        try:
            truncate_sql = f"TRUNCATE TABLE STG_TWITCH.{stg_table}"
            stg_session.sql(truncate_sql).collect()

            count_sql = f"""
                SELECT COUNT(*) AS cnt FROM ODS_TWITCH.{ods_table}
                WHERE "business_date" BETWEEN TO_TIMESTAMP('{start_str}') AND TO_TIMESTAMP('{end_str}')
            """
            count_result = ods_session.sql(count_sql).collect()
            rows_to_insert = count_result[0]["CNT"] if count_result else 0

            insert_sql = f"""
                INSERT INTO STG_TWITCH.{stg_table}
                SELECT * FROM ODS_TWITCH.{ods_table}
                WHERE "business_date" BETWEEN TO_TIMESTAMP('{start_str}') AND TO_TIMESTAMP('{end_str}')
            """
            stg_session.sql(insert_sql).collect()

            print(
                f"Inserted {rows_to_insert} rows into {stg_table} for {start_str} to {end_str}"
            )
        except Exception as e:
            print(f"Error processing {ods_table} -> {stg_table}: {e}")


# stage_replace_incremental()
