from snowflake.snowpark import Session
import os
from dotenv import load_dotenv

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


def bulk_copy_table(ods_table: str, stg_table: str, stg_session: Session):
    try:
        count_result = stg_session.sql(
            f"SELECT COUNT(*) AS row_count FROM ODS_TWITCH.{ods_table}"
        ).collect()
        row_count = count_result[0]["ROW_COUNT"]

        sql = f"""
            CREATE OR REPLACE TABLE {stg_table} AS
            SELECT * FROM ODS_TWITCH.{ods_table}
        """
        stg_session.sql(sql).collect()

        print(f"Bulk copied {row_count} rows from {ods_table} → {stg_table}")
    except Exception as e:
        print(f"Failed to copy {ods_table} → {stg_table}: {e}")


def stage_save():
    stg_session = get_snowpark_session("STG_TWITCH")

    tables = {
        "ods_streamer_raw": "stg_streamer_raw",
        "ods_stream_raw": "stg_stream_raw",
        "ods_message_raw": "stg_message_raw",
        "ods_user_raw": "stg_user_raw",
        "ods_dates_raw": "stg_dates_raw",
    }

    for ods_table, stg_table in tables.items():
        bulk_copy_table(ods_table, stg_table, stg_session)


# stage_save()
