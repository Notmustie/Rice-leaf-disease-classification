#Install psycopg2 using  "pip install psycopg2-binary"

#!/usr/bin/env python3
"""
Load inference predictions CSV into PostgreSQL with upsert.

Input: outputs/predictions.csv created by src/inference/predict.py
Output: rows inserted/updated in rice_leaf_predictions table

Usage:
  python -m src.warehouse.load_predictions \
    --csv outputs/preds_custom.csv \
    --run_id customcnn_run_001 \
    --model_name custom_cnn_6class_2nd \
    --db_url "postgresql://user:pass@host:5432/dbname"

Optional:
  --schema_sql src/warehouse/schema.sql
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


REQUIRED_COLUMNS = {
    "path",
    "true_label",
    "pred_label",
    "pred_conf",
    "rejected",
    "topk_labels",
    "topk_confs",
    "preprocess_mode",
    "reject_threshold",
}


def read_sql(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def ensure_schema(conn, schema_sql_path: Path) -> None:
    sql = read_sql(schema_sql_path)
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", type=Path, required=True, help="Predictions CSV from predict.py")
    ap.add_argument("--run_id", type=str, required=True, help="Unique run id (used for upsert uniqueness)")
    ap.add_argument("--model_name", type=str, required=True, help="Model name (for dashboard filtering)")
    ap.add_argument("--db_url", type=str, required=True,
                    help='Postgres URL like: postgresql://user:pass@host:5432/dbname')

    ap.add_argument("--schema_sql", type=Path, default=Path("src/warehouse/schema.sql"),
                    help="SQL file to create the table if missing")
    ap.add_argument("--table", type=str, default="rice_leaf_predictions")
    ap.add_argument("--dry_run", action="store_true", help="Validate only, do not write to DB")

    args = ap.parse_args()

    csv_path = args.csv.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Normalize types
    df["pred_conf"] = df["pred_conf"].astype(float)
    df["rejected"] = df["rejected"].astype(bool)
    df["reject_threshold"] = df["reject_threshold"].astype(float)

    # Add helper columns
    df["image_path"] = df["path"].astype(str)
    df["image_name"] = df["image_path"].apply(lambda s: s.split("/")[-1])
    df["run_id"] = args.run_id
    df["model_name"] = args.model_name

    # Optional: coerce empty true_label to None
    df["true_label"] = df["true_label"].where(df["true_label"].notna(), None)

    cols = [
        "run_id",
        "model_name",
        "preprocess_mode",
        "reject_threshold",
        "image_path",
        "image_name",
        "true_label",
        "pred_label",
        "pred_conf",
        "rejected",
        "topk_labels",
        "topk_confs",
    ]

    records = [tuple(row) for row in df[cols].itertuples(index=False, name=None)]

    if args.dry_run:
        print(f"✅ DRY RUN: would load {len(records)} rows into {args.table}")
        return

    conn = psycopg2.connect(args.db_url)
    try:
        # Ensure table exists
        ensure_schema(conn, args.schema_sql.expanduser().resolve())

        insert_sql = f"""
        INSERT INTO {args.table} (
            run_id, model_name, preprocess_mode, reject_threshold,
            image_path, image_name, true_label,
            pred_label, pred_conf, rejected,
            topk_labels, topk_confs
        ) VALUES %s
        ON CONFLICT (run_id, image_path) DO UPDATE SET
            model_name = EXCLUDED.model_name,
            preprocess_mode = EXCLUDED.preprocess_mode,
            reject_threshold = EXCLUDED.reject_threshold,
            image_name = EXCLUDED.image_name,
            true_label = EXCLUDED.true_label,
            pred_label = EXCLUDED.pred_label,
            pred_conf = EXCLUDED.pred_conf,
            rejected = EXCLUDED.rejected,
            topk_labels = EXCLUDED.topk_labels,
            topk_confs = EXCLUDED.topk_confs,
            created_at = NOW();
        """

        with conn.cursor() as cur:
            execute_values(cur, insert_sql, records, page_size=1000)

        conn.commit()
        print(f"✅ Loaded {len(records)} rows into {args.table} (run_id={args.run_id})")

    finally:
        conn.close()


if __name__ == "__main__":
    main()

