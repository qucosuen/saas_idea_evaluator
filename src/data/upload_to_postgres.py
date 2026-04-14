"""
Upload Training Data to PostgreSQL & Update Feature Store
==========================================================
1. Creates tables in the feast_offline database for our training data
2. Uploads all 501 examples as structured rows
3. Sets up a Feast feature repo pointing at the Postgres offline store

Usage:
  .venv/bin/python scripts/upload_to_postgres.py
"""

import json
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timezone
from pathlib import Path

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "admin",
    "password": "admin",
    "dbname": "feast_offline",
}

def create_tables(cur):
    """Create tables for training data and project evaluations."""

    # Main training examples table — one row per training example
    cur.execute("""
    CREATE TABLE IF NOT EXISTS project_evaluation_training (
        example_id          VARCHAR(32) PRIMARY KEY,
        source_type         VARCHAR(32) NOT NULL,
        project_description TEXT NOT NULL,
        pain_severity       SMALLINT NOT NULL CHECK (pain_severity BETWEEN 1 AND 10),
        pain_severity_reason TEXT NOT NULL,
        pain_frequency      SMALLINT NOT NULL CHECK (pain_frequency BETWEEN 1 AND 10),
        pain_frequency_reason TEXT NOT NULL,
        existing_alternatives SMALLINT NOT NULL CHECK (existing_alternatives BETWEEN 1 AND 10),
        existing_alternatives_reason TEXT NOT NULL,
        willingness_to_pay  SMALLINT NOT NULL CHECK (willingness_to_pay BETWEEN 1 AND 10),
        willingness_to_pay_reason TEXT NOT NULL,
        market_size         SMALLINT NOT NULL CHECK (market_size BETWEEN 1 AND 10),
        market_size_reason  TEXT NOT NULL,
        scalability         SMALLINT NOT NULL CHECK (scalability BETWEEN 1 AND 10),
        scalability_reason  TEXT NOT NULL,
        profitability_potential SMALLINT NOT NULL CHECK (profitability_potential BETWEEN 1 AND 10),
        profitability_potential_reason TEXT NOT NULL,
        defensibility       SMALLINT NOT NULL CHECK (defensibility BETWEEN 1 AND 10),
        defensibility_reason TEXT NOT NULL,
        time_to_value       SMALLINT NOT NULL CHECK (time_to_value BETWEEN 1 AND 10),
        time_to_value_reason TEXT NOT NULL,
        founder_market_fit_requirement SMALLINT NOT NULL CHECK (founder_market_fit_requirement BETWEEN 1 AND 10),
        founder_market_fit_requirement_reason TEXT NOT NULL,
        overall_score       REAL NOT NULL,
        summary             TEXT NOT NULL,
        messages_json       JSONB NOT NULL,
        event_timestamp     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        created_timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """)

    # Feature table for the feature store — flattened scores for ML consumption
    cur.execute("""
    CREATE TABLE IF NOT EXISTS project_evaluation_features (
        example_id          VARCHAR(32) PRIMARY KEY,
        source_type         VARCHAR(32) NOT NULL,
        description_length  INTEGER NOT NULL,
        pain_severity       SMALLINT NOT NULL,
        pain_frequency      SMALLINT NOT NULL,
        existing_alternatives SMALLINT NOT NULL,
        willingness_to_pay  SMALLINT NOT NULL,
        market_size         SMALLINT NOT NULL,
        scalability         SMALLINT NOT NULL,
        profitability_potential SMALLINT NOT NULL,
        defensibility       SMALLINT NOT NULL,
        time_to_value       SMALLINT NOT NULL,
        founder_market_fit_requirement SMALLINT NOT NULL,
        overall_score       REAL NOT NULL,
        score_tier          VARCHAR(16) NOT NULL,
        event_timestamp     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        created_timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """)

    # Score distribution summary view
    cur.execute("""
    CREATE OR REPLACE VIEW project_evaluation_score_distribution AS
    SELECT
        score_tier,
        COUNT(*) as count,
        ROUND(AVG(overall_score)::numeric, 2) as avg_score,
        MIN(overall_score) as min_score,
        MAX(overall_score) as max_score,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as pct
    FROM project_evaluation_features
    GROUP BY score_tier
    ORDER BY avg_score;
    """)

    print("Tables and views created.")



def score_tier(overall):
    if overall <= 3: return "weak"
    elif overall <= 5: return "below_avg"
    elif overall <= 6.5: return "average"
    elif overall <= 8: return "strong"
    else: return "exceptional"


def upload_data(cur, data):
    """Upload training examples to both tables."""
    now = datetime.now(timezone.utc)

    training_rows = []
    feature_rows = []

    for ex in data:
        eid = ex["source_id"]
        src = ex["source_type"]
        desc = ex["messages"][1]["content"].replace("Evaluate this project: ", "", 1)
        ev = json.loads(ex["messages"][2]["content"])

        training_rows.append((
            eid, src, desc,
            ev["pain_severity"]["score"], ev["pain_severity"]["reason"],
            ev["pain_frequency"]["score"], ev["pain_frequency"]["reason"],
            ev["existing_alternatives"]["score"], ev["existing_alternatives"]["reason"],
            ev["willingness_to_pay"]["score"], ev["willingness_to_pay"]["reason"],
            ev["market_size"]["score"], ev["market_size"]["reason"],
            ev["scalability"]["score"], ev["scalability"]["reason"],
            ev["profitability_potential"]["score"], ev["profitability_potential"]["reason"],
            ev["defensibility"]["score"], ev["defensibility"]["reason"],
            ev["time_to_value"]["score"], ev["time_to_value"]["reason"],
            ev["founder_market_fit_requirement"]["score"], ev["founder_market_fit_requirement"]["reason"],
            ev["overall_score"], ev["summary"],
            json.dumps(ex["messages"]),
            now, now,
        ))

        feature_rows.append((
            eid, src, len(desc),
            ev["pain_severity"]["score"], ev["pain_frequency"]["score"],
            ev["existing_alternatives"]["score"], ev["willingness_to_pay"]["score"],
            ev["market_size"]["score"], ev["scalability"]["score"],
            ev["profitability_potential"]["score"], ev["defensibility"]["score"],
            ev["time_to_value"]["score"], ev["founder_market_fit_requirement"]["score"],
            ev["overall_score"], score_tier(ev["overall_score"]),
            now, now,
        ))

    # Upsert training data
    execute_values(cur, """
        INSERT INTO project_evaluation_training (
            example_id, source_type, project_description,
            pain_severity, pain_severity_reason,
            pain_frequency, pain_frequency_reason,
            existing_alternatives, existing_alternatives_reason,
            willingness_to_pay, willingness_to_pay_reason,
            market_size, market_size_reason,
            scalability, scalability_reason,
            profitability_potential, profitability_potential_reason,
            defensibility, defensibility_reason,
            time_to_value, time_to_value_reason,
            founder_market_fit_requirement, founder_market_fit_requirement_reason,
            overall_score, summary, messages_json,
            event_timestamp, created_timestamp
        ) VALUES %s
        ON CONFLICT (example_id) DO UPDATE SET
            project_description = EXCLUDED.project_description,
            overall_score = EXCLUDED.overall_score,
            messages_json = EXCLUDED.messages_json,
            event_timestamp = EXCLUDED.event_timestamp
    """, training_rows)
    print(f"  project_evaluation_training: {len(training_rows)} rows upserted")

    # Upsert feature data
    execute_values(cur, """
        INSERT INTO project_evaluation_features (
            example_id, source_type, description_length,
            pain_severity, pain_frequency, existing_alternatives,
            willingness_to_pay, market_size, scalability,
            profitability_potential, defensibility,
            time_to_value, founder_market_fit_requirement,
            overall_score, score_tier,
            event_timestamp, created_timestamp
        ) VALUES %s
        ON CONFLICT (example_id) DO UPDATE SET
            overall_score = EXCLUDED.overall_score,
            score_tier = EXCLUDED.score_tier,
            event_timestamp = EXCLUDED.event_timestamp
    """, feature_rows)
    print(f"  project_evaluation_features: {len(feature_rows)} rows upserted")


def verify(cur):
    """Print verification stats."""
    cur.execute("SELECT COUNT(*) FROM project_evaluation_training")
    print(f"\nVerification:")
    print(f"  Training rows: {cur.fetchone()[0]}")

    cur.execute("SELECT COUNT(*) FROM project_evaluation_features")
    print(f"  Feature rows: {cur.fetchone()[0]}")

    cur.execute("SELECT * FROM project_evaluation_score_distribution")
    print(f"\n  Score distribution:")
    print(f"  {'Tier':<14} {'Count':>6} {'Avg':>6} {'Min':>6} {'Max':>6} {'Pct':>6}")
    print(f"  {'-'*46}")
    for row in cur.fetchall():
        print(f"  {row[0]:<14} {row[1]:>6} {row[2]:>6} {row[3]:>6.1f} {row[4]:>6.1f} {row[5]:>5.1f}%")


def main():
    # Load data
    data_file = Path(__file__).parent.parent / "data" / "training_data_full.json"
    with open(data_file) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {data_file.name}")

    # Connect and upload
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        print("\nCreating tables...")
        create_tables(cur)

        print("\nUploading data...")
        upload_data(cur, data)

        conn.commit()
        print("\nCommitted.")

        verify(cur)

    except Exception as e:
        conn.rollback()
        print(f"\nError: {e}")
        raise
    finally:
        cur.close()
        conn.close()

    print("\nDone. Data is in feast_offline database.")


if __name__ == "__main__":
    main()
