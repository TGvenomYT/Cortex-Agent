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

-- ═══ VECTOR RETRIEVAL (semantic relevance for context + memory) ═══
-- Requires: CREATE EXTENSION IF NOT EXISTS vector;
-- Embedding model: nomic-embed-text via local Ollama (768 dims). 100% local, no API key.

CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE context_store ADD COLUMN IF NOT EXISTS embedding vector(768);
ALTER TABLE memory        ADD COLUMN IF NOT EXISTS embedding vector(768);

-- HNSW indexes for fast approximate nearest-neighbor search (cosine distance)
CREATE INDEX IF NOT EXISTS idx_context_embedding ON context_store
    USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_memory_embedding ON memory
    USING hnsw (embedding vector_cosine_ops);

-- ═══ EXPERIENCES (State-Action-Reward buffer — BIG BRAIN System 2 substitute) ═══
-- Every agent turn is logged here. Reward is computed after the fact from outcomes
-- (command exit code, user correction/re-ask, explicit feedback). Retrieval over
-- high-reward experiences acts as the in-context "learning" substitute for weight
-- updates that aren't possible against a closed-weight model like GPT-4o.

CREATE TABLE IF NOT EXISTS experiences (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,

    -- STATE: what the agent saw
    user_text TEXT NOT NULL,
    state_context JSONB,                    -- snapshot of relevant context_store rows used

    -- ACTION: what the agent decided to do
    action VARCHAR(30) NOT NULL,             -- 'execute' | 'browse' | 'edit' | 'answer' | ...
    action_payload JSONB NOT NULL,           -- the full response_json the LLM returned
    speak TEXT,

    -- OUTCOME: raw signals used to compute reward
    exit_code INTEGER,                       -- for execute/devops actions
    was_corrected BOOLEAN DEFAULT FALSE,     -- user re-asked / rephrased within N turns
    was_confirmed BOOLEAN,                   -- for actions that needed confirmation
    explicit_feedback SMALLINT,              -- -1 | 0 | +1, from a future feedback API

    -- REWARD: computed scalar, -1.0 to +1.0
    reward FLOAT,
    reward_reason TEXT,                      -- human-readable why (for debugging the reward function)

    -- Retrieval (nomic-embed-text, local, 768 dims)
    embedding vector(768),

    created_at TIMESTAMP DEFAULT NOW(),
    reward_computed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_experiences_session ON experiences(session_id);
CREATE INDEX IF NOT EXISTS idx_experiences_reward ON experiences(reward DESC);
CREATE INDEX IF NOT EXISTS idx_experiences_created ON experiences(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_experiences_embedding ON experiences
    USING hnsw (embedding vector_cosine_ops);
-- Pending reward computation (experiences created but outcome not yet scored)
CREATE INDEX IF NOT EXISTS idx_experiences_pending_reward ON experiences(created_at)
    WHERE reward IS NULL;

-- ═══ GOALS + AUTONOMY LOG (self-directed background cognition) ═══
-- The autonomy loop (autonomy.py) periodically looks at current context/memory/
-- experience and decides whether to form a goal or act on an existing one, gated
-- through the same needs_confirmation() check /ask uses for live requests.

CREATE TABLE IF NOT EXISTS goals (
    id SERIAL PRIMARY KEY,
    description TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'active',    -- active | done | abandoned
    priority INTEGER DEFAULT 3,             -- 1-5
    source VARCHAR(20) DEFAULT 'self',      -- self | user
    created_at TIMESTAMP DEFAULT NOW(),
    last_evaluated_at TIMESTAMP,
    completed_at TIMESTAMP,
    result TEXT
);

CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);

CREATE TABLE IF NOT EXISTS autonomy_log (
    id SERIAL PRIMARY KEY,
    cycle_at TIMESTAMP DEFAULT NOW(),
    think TEXT,
    decision VARCHAR(20),                   -- noop | new_goal | acted | proposed
    goal_id INTEGER REFERENCES goals(id),
    action_payload JSONB,
    needs_confirmation BOOLEAN DEFAULT FALSE,
    confirmed BOOLEAN,
    exit_code INTEGER,
    notified BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_autonomy_log_time ON autonomy_log(cycle_at DESC);

-- ═══ PRIORITIES (standing values — the compass, not a one-off task) ═══
-- Unlike a goal (one-off, gets marked done/abandoned), a priority is something
-- the agent holds as important across turns and cycles. Injected into both the
-- live /ask system prompt (context_reader.py) and the autonomy loop (autonomy.py)
-- so the two paths share one opinion instead of behaving differently. Can be
-- seeded by the user, or formed by the agent itself from recurring self-goals /
-- experience patterns (source='learned' — see autonomy.py::_maybe_form_priority).

CREATE TABLE IF NOT EXISTS priorities (
    id SERIAL PRIMARY KEY,
    statement TEXT NOT NULL,
    weight INTEGER DEFAULT 3,               -- 1-5
    source VARCHAR(20) DEFAULT 'user',      -- seeded | user | learned
    status VARCHAR(20) DEFAULT 'active',    -- active | retired
    created_at TIMESTAMP DEFAULT NOW(),
    last_reinforced_at TIMESTAMP,
    reinforcement_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_priorities_status ON priorities(status);

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
