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


def stage_historical_callable():
    from historical_stage import stage_save

    return stage_save()


def target_storage_callable():
    from target import target_storage

    return target_storage()


def rake_analysis_callable():
    from analysis import keybert_analysis

    return keybert_analysis()


with DAG(
    "twitch_historical",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=["historical"],
) as dag:

    ods = PythonOperator(
        task_id="ods",
        python_callable=load_all_callable,
    )

    historical_stage = PythonOperator(
        task_id="historical_stage",
        python_callable=stage_historical_callable,
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
        subject="historical DAG Failed!",
        html_content="One or more tasks failed in the historical_dag.",
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    ods >> historical_stage >> target >> analysis
    [ods, historical_stage, target, analysis] >> send_failure_alert
