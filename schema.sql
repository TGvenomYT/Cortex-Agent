-- ═══════════════════════════════════════════════════════════════
-- Cortex — PostgreSQL Schema
-- Run against your Supabase Postgres instance:
-- psql "postgresql://supabase_admin:<password>@100.100.35.38:5435/postgres" -f schema.sql
-- ═══════════════════════════════════════════════════════════════

-- ═══ CONTEXT (pre-computed by cron collectors) ═══

CREATE TABLE IF NOT EXISTS context_store (
    id SERIAL PRIMARY KEY,
    category VARCHAR(50) NOT NULL,          -- 'git' | 'process' | 'project' | 'system' | 'shell'
    project_path VARCHAR(500),              -- null for system-level context
    key VARCHAR(200) NOT NULL,              -- 'branch', 'status', 'running_services', etc.
    value JSONB NOT NULL,                   -- flexible structured data
    collected_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,                   -- stale data auto-purges
    UNIQUE(category, project_path, key)
);

CREATE INDEX IF NOT EXISTS idx_context_category ON context_store(category);
CREATE INDEX IF NOT EXISTS idx_context_project ON context_store(project_path);
CREATE INDEX IF NOT EXISTS idx_context_expires ON context_store(expires_at);
CREATE INDEX IF NOT EXISTS idx_context_collected ON context_store(collected_at DESC);

-- ═══ MEMORY (persistent knowledge) ═══

CREATE TABLE IF NOT EXISTS memory (
    id SERIAL PRIMARY KEY,
    type VARCHAR(50) NOT NULL,              -- 'fact' | 'preference' | 'learned_command' | 'project_info'
    content TEXT NOT NULL,
    source VARCHAR(100),                    -- what conversation/action produced this
    importance INTEGER DEFAULT 3,           -- 1-5, LLM-assigned
    created_at TIMESTAMP DEFAULT NOW(),
    last_accessed TIMESTAMP,
    access_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memory_type ON memory(type);
CREATE INDEX IF NOT EXISTS idx_memory_importance ON memory(importance DESC);

-- ═══ CONVERSATIONS ═══

CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    role VARCHAR(20) NOT NULL,              -- 'user' | 'assistant' | 'system'
    content TEXT NOT NULL,
    tokens_used INTEGER,
    model_used VARCHAR(50),
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conv_time ON conversations(timestamp DESC);

-- ═══ REMINDERS ═══

CREATE TABLE IF NOT EXISTS reminders (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    due_at TIMESTAMP NOT NULL,
    repeat_rule VARCHAR(50),                -- null | 'daily' | 'weekly' | 'weekdays' | cron expr
    context_snapshot JSONB,                 -- what user was doing when reminder was set
    fired BOOLEAN DEFAULT FALSE,
    fired_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(due_at) WHERE fired = FALSE;

-- ═══ COMMAND LOG (every command Cortex executed) ═══

CREATE TABLE IF NOT EXISTS command_log (
    id SERIAL PRIMARY KEY,
    command TEXT NOT NULL,
    exit_code INTEGER,
    stdout TEXT,
    stderr TEXT,
    duration_ms INTEGER,
    triggered_by VARCHAR(100),              -- 'siri' | 'smart-terminal' | 'devops_agent' | 'api'
    project_path VARCHAR(500),
    executed_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cmdlog_time ON command_log(executed_at DESC);
CREATE INDEX IF NOT EXISTS idx_cmdlog_trigger ON command_log(triggered_by);

-- ═══ TRACKED PROJECTS ═══

CREATE TABLE IF NOT EXISTS tracked_projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    path VARCHAR(500) NOT NULL UNIQUE,
    type VARCHAR(50),                       -- 'python' | 'node' | 'rust' | 'go' | etc.
    collect_git BOOLEAN DEFAULT TRUE,
    collect_deps BOOLEAN DEFAULT TRUE,
    custom_commands JSONB,                  -- project-specific commands cortex should know
    added_at TIMESTAMP DEFAULT NOW()
);

-- ═══ DEVOPS PLANS (stored plans for approval/execution) ═══

CREATE TABLE IF NOT EXISTS devops_plans (
    id SERIAL PRIMARY KEY,
    plan_id VARCHAR(100) NOT NULL UNIQUE,
    task TEXT NOT NULL,
    project_path VARCHAR(500),
    steps JSONB NOT NULL,                   -- array of {step, cmd, desc}
    status VARCHAR(20) DEFAULT 'pending',   -- 'pending' | 'approved' | 'running' | 'completed' | 'failed'
    current_step INTEGER DEFAULT 0,
    results JSONB,                          -- per-step results after execution
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_plans_status ON devops_plans(status);
CREATE INDEX IF NOT EXISTS idx_plans_planid ON devops_plans(plan_id);

-- ═══ CLEANUP FUNCTION ═══
-- Call periodically to purge stale context
-- Can be triggered by pg_cron if available, or by the daemon itself

CREATE OR REPLACE FUNCTION purge_expired_context()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM context_store WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
