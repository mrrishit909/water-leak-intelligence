-- Domain model. Names follow the blueprint's schema; additions are marked (+).
-- Positions are junction numbers on the synthetic grid, not PostGIS geometry.

CREATE TABLE districts (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, ordinal int NOT NULL, name text NOT NULL, junctions int NOT NULL, model_seed int NOT NULL, telemetry_from timestamptz NOT NULL,   -- (+) district metered areas
  UNIQUE (tenant_id, ordinal));
SELECT enable_tenant_rls('districts');

CREATE TABLE assets (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, district_id uuid NOT NULL REFERENCES districts ON DELETE CASCADE, asset_type text NOT NULL DEFAULT 'pipe', ordinal int NOT NULL,
  material text NOT NULL, installed_year int NOT NULL, diameter_mm int NOT NULL, length_m real NOT NULL, soil real NOT NULL, a_node int NOT NULL, b_node int NOT NULL,
  risk real, risk_rank int);                                                                             -- (+) chance of failing next year, from the latest model
CREATE INDEX assets_district ON assets (district_id, ordinal);
CREATE INDEX assets_risk ON assets (tenant_id, risk_rank);
SELECT enable_tenant_rls('assets');

CREATE TABLE asset_failures (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, asset_id uuid NOT NULL REFERENCES assets ON DELETE CASCADE, year int NOT NULL);
CREATE INDEX asset_failures_asset ON asset_failures (asset_id);
SELECT enable_tenant_rls('asset_failures');

CREATE TABLE sensors (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, district_id uuid NOT NULL REFERENCES districts ON DELETE CASCADE, kind text NOT NULL CHECK (kind IN ('pressure','flow')), ordinal int NOT NULL, node int NOT NULL);
SELECT enable_tenant_rls('sensors');

CREATE TABLE telemetry (sensor_id uuid NOT NULL REFERENCES sensors ON DELETE CASCADE, ts timestamptz NOT NULL, value real NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants, step int NOT NULL, PRIMARY KEY (sensor_id, ts));
SELECT enable_tenant_rls('telemetry');

CREATE TABLE leak_events (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, district_id uuid NOT NULL REFERENCES districts ON DELETE CASCADE, kind text NOT NULL CHECK (kind IN ('leak','sensor_fault','transient_demand','unexplained')),
  status text NOT NULL DEFAULT 'open' CHECK (status IN ('open','work_ordered')), onset_day int, estimated_lps real, probability real, candidates jsonb NOT NULL DEFAULT '[]', evidence jsonb NOT NULL, model text NOT NULL,
  detected_at timestamptz NOT NULL DEFAULT now(), UNIQUE (district_id, kind));
SELECT enable_tenant_rls('leak_events');

CREATE TABLE work_orders (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, leak_event_id uuid NOT NULL UNIQUE REFERENCES leak_events, kind text NOT NULL CHECK (kind IN ('find_and_repair_leak','check_sensor')),
  priority text NOT NULL, status text NOT NULL DEFAULT 'open', asset_ids uuid[] NOT NULL DEFAULT '{}', sensor_id uuid REFERENCES sensors, detail jsonb NOT NULL, note text NOT NULL, created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now());
SELECT enable_tenant_rls('work_orders');

CREATE TABLE risk_models (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, version text NOT NULL, metrics jsonb NOT NULL, trained_at timestamptz NOT NULL DEFAULT now());   -- (+)
SELECT enable_tenant_rls('risk_models');

CREATE TABLE capital_plans (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, budget_per_year numeric NOT NULL, years int NOT NULL, summary jsonb NOT NULL, items jsonb NOT NULL, created_by uuid NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now());
SELECT enable_tenant_rls('capital_plans');
