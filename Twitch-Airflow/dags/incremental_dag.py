from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 6, 13),
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}


def load_all_callable():
    from ods import load_all

    return load_all()


def stage_replace_incremental_callable():
    from incremental_stage import stage_replace_incremental

    return stage_replace_incremental()


def target_storage_callable():
    from target import target_storage

    return target_storage()


def rake_analysis_callable():
    from analysis import keybert_analysis

    return keybert_analysis()


with DAG(
    "twitch_incremental",
    default_args=default_args,
    schedule_interval="@hourly",
    catchup=False,
    max_active_runs=1,
    tags=["incremental"],
) as dag:

    ods = PythonOperator(
        task_id="ods",
        python_callable=load_all_callable,
    )

    incremental_stage = PythonOperator(
        task_id="incremental_stage",
        python_callable=stage_replace_incremental_callable,
    )

    target = PythonOperator(
        task_id="target",
        python_callable=target_storage_callable,
    )

    analysis = PythonOperator(
        task_id="analysis",
        python_callable=rake_analysis_callable,
    )

    send_failure_alert = EmailOperator(
        task_id="send_failure_alert",
        to=["nischal.sha17@gmail.com", "riyamaharjan99@gmail.com"],
        subject="Incremental DAG Failed!",
        html_content="One or more tasks failed in the incremental_dag.",
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    ods >> incremental_stage >> target >> analysis
    [ods, incremental_stage, target, analysis] >> send_failure_alert
