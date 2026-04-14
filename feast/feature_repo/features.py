"""
Feast Feature Definitions for Project Evaluator
=================================================
Defines entities, feature views, and feature services for the
project evaluation training data stored in PostgreSQL.

To apply:
  cd feast/feature_repo && feast apply
"""

from datetime import timedelta
from feast import Entity, FeatureView, Field, FeatureService
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float32, Int32, String

# ---------------------------------------------------------------------------
# Entity: a single training example / project evaluation
# ---------------------------------------------------------------------------

project_example = Entity(
    name="project_example",
    join_keys=["example_id"],
    description="A single project evaluation training example",
)

# ---------------------------------------------------------------------------
# Data Source: PostgreSQL table with flattened evaluation scores
# ---------------------------------------------------------------------------

evaluation_features_source = PostgreSQLSource(
    name="project_evaluation_features_source",
    query="SELECT * FROM project_evaluation_features",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# ---------------------------------------------------------------------------
# Feature View: project evaluation scores
# ---------------------------------------------------------------------------

project_evaluation_fv = FeatureView(
    name="project_evaluation_scores",
    entities=[project_example],
    schema=[
        Field(name="source_type", dtype=String, description="How the example was generated: original, synthetic, paraphrase, edge_case"),
        Field(name="description_length", dtype=Int32, description="Character length of the project description"),
        Field(name="pain_severity", dtype=Int32, description="How painful is the problem (1-10)"),
        Field(name="pain_frequency", dtype=Int32, description="How often users face this problem (1-10)"),
        Field(name="existing_alternatives", dtype=Int32, description="How well current solutions serve the need (1=saturated, 10=no alternatives)"),
        Field(name="willingness_to_pay", dtype=Int32, description="How likely users are to pay (1-10)"),
        Field(name="market_size", dtype=Int32, description="Size of addressable market (1-10)"),
        Field(name="scalability", dtype=Int32, description="How easily the project scales (1-10)"),
        Field(name="profitability_potential", dtype=Int32, description="Revenue and margin potential (1-10)"),
        Field(name="defensibility", dtype=Int32, description="How hard to replicate (1-10)"),
        Field(name="time_to_value", dtype=Int32, description="How fast users see value (1-10)"),
        Field(name="founder_market_fit_requirement", dtype=Int32, description="Domain expertise needed (1-10)"),
        Field(name="overall_score", dtype=Float32, description="Weighted overall viability score (1-10)"),
        Field(name="score_tier", dtype=String, description="Score bucket: weak/below_avg/average/strong/exceptional"),
    ],
    source=evaluation_features_source,
    online=True,
    ttl=timedelta(days=365),
    description="Flattened project evaluation scores for ML training and feature serving",
    tags={"team": "ml", "project": "slm-evaluator"},
)

# ---------------------------------------------------------------------------
# Data Source: Full training data with descriptions and reasons
# ---------------------------------------------------------------------------

training_data_source = PostgreSQLSource(
    name="project_evaluation_training_source",
    query="SELECT * FROM project_evaluation_training",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# ---------------------------------------------------------------------------
# Feature View: full training data (for data retrieval, not online serving)
# ---------------------------------------------------------------------------

project_training_fv = FeatureView(
    name="project_evaluation_training_data",
    entities=[project_example],
    schema=[
        Field(name="source_type", dtype=String),
        Field(name="project_description", dtype=String),
        Field(name="pain_severity", dtype=Int32),
        Field(name="pain_severity_reason", dtype=String),
        Field(name="pain_frequency", dtype=Int32),
        Field(name="pain_frequency_reason", dtype=String),
        Field(name="existing_alternatives", dtype=Int32),
        Field(name="existing_alternatives_reason", dtype=String),
        Field(name="willingness_to_pay", dtype=Int32),
        Field(name="willingness_to_pay_reason", dtype=String),
        Field(name="market_size", dtype=Int32),
        Field(name="market_size_reason", dtype=String),
        Field(name="scalability", dtype=Int32),
        Field(name="scalability_reason", dtype=String),
        Field(name="profitability_potential", dtype=Int32),
        Field(name="profitability_potential_reason", dtype=String),
        Field(name="defensibility", dtype=Int32),
        Field(name="defensibility_reason", dtype=String),
        Field(name="time_to_value", dtype=Int32),
        Field(name="time_to_value_reason", dtype=String),
        Field(name="founder_market_fit_requirement", dtype=Int32),
        Field(name="founder_market_fit_requirement_reason", dtype=String),
        Field(name="overall_score", dtype=Float32),
        Field(name="summary", dtype=String),
    ],
    source=training_data_source,
    online=False,
    ttl=timedelta(days=365),
    description="Full training data with descriptions and justifications for offline retrieval",
    tags={"team": "ml", "project": "slm-evaluator"},
)

# ---------------------------------------------------------------------------
# Feature Service: bundle for training pipeline consumption
# ---------------------------------------------------------------------------

project_evaluator_service = FeatureService(
    name="project_evaluator_training",
    features=[project_evaluation_fv, project_training_fv],
    description="Complete feature service for the SLM project evaluator training pipeline",
    tags={"team": "ml", "project": "slm-evaluator"},
)


# ---------------------------------------------------------------------------
# Indie Hackers Ideas (scraped real-world project data)
# ---------------------------------------------------------------------------

indiehackers_source = PostgreSQLSource(
    name="indiehackers_idea_features_source",
    query="SELECT * FROM indiehackers_idea_features",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

indiehackers_idea = Entity(
    name="indiehackers_idea",
    join_keys=["idea_id"],
    description="A real-world startup idea scraped from Indie Hackers",
)

indiehackers_fv = FeatureView(
    name="indiehackers_ideas",
    entities=[indiehackers_idea],
    schema=[
        Field(name="name", dtype=String, description="Product/company name"),
        Field(name="description", dtype=String, description="One-line project description"),
        Field(name="category_primary", dtype=String, description="Primary category"),
        Field(name="category_count", dtype=Int32, description="Number of categories"),
        Field(name="mrr_usd", dtype=Int32, description="Monthly recurring revenue in USD"),
        Field(name="description_length", dtype=Int32, description="Character length of description"),
        Field(name="has_ai", dtype=String, description="Whether the idea involves AI"),
        Field(name="is_saas", dtype=String, description="Whether the idea is SaaS"),
        Field(name="is_marketplace", dtype=String, description="Whether the idea is a marketplace"),
    ],
    source=indiehackers_source,
    online=True,
    ttl=timedelta(days=365),
    description="Real-world startup ideas from Indie Hackers with revenue data",
    tags={"team": "ml", "project": "slm-evaluator", "source": "indiehackers"},
)
