-- src/warehouse/schema.sql
-- Table to store inference results (image-level predictions)

CREATE TABLE IF NOT EXISTS rice_leaf_predictions (
    id               BIGSERIAL PRIMARY KEY,
    run_id           TEXT NOT NULL,
    model_name       TEXT NOT NULL,
    preprocess_mode  TEXT NOT NULL,
    reject_threshold DOUBLE PRECISION NOT NULL,

    image_path       TEXT NOT NULL,
    image_name       TEXT NOT NULL,

    true_label       TEXT NULL,
    pred_label       TEXT NOT NULL,
    pred_conf        DOUBLE PRECISION NOT NULL,
    rejected         BOOLEAN NOT NULL,

    topk_labels      TEXT NOT NULL,
    topk_confs       TEXT NOT NULL,

    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- prevents duplicate inserts when reloading the same run
    CONSTRAINT uq_run_image UNIQUE (run_id, image_path)
);

-- Helpful indexes for dashboard filters
CREATE INDEX IF NOT EXISTS idx_rice_leaf_predictions_run_id ON rice_leaf_predictions (run_id);
CREATE INDEX IF NOT EXISTS idx_rice_leaf_predictions_pred_label ON rice_leaf_predictions (pred_label);
CREATE INDEX IF NOT EXISTS idx_rice_leaf_predictions_rejected ON rice_leaf_predictions (rejected);
