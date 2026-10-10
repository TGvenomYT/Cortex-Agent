# ─────────────────────────────────────────────────────────────
# Smart Terminal → Cortex Bridge
# Source this file in your .zshrc AFTER smart-terminal loads.
# It overrides the smart-terminal LLM call to route through Cortex.
#
# Usage: Add to ~/.zshrc:
#   source "/Users/niranjan/Documents/Cortex Agent/smart_terminal_hook.zsh"
# ─────────────────────────────────────────────────────────────

# Override the smart-terminal command generation to use Cortex
cortex-generate-command() {
    local query="$1"
    local result=$(curl -s -X POST http://localhost:7800/ask \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"generate a shell command for: $query\", \"source\": \"smart-terminal\", \"session_id\": \"terminal\"}" 2>/dev/null)
    
    local command=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('command','') or d.get('speak',''))" 2>/dev/null)
    
    if [[ -n "$command" && "$command" != "null" ]]; then
        echo "$command"
    fi
}

# Override the explain-last function to use Cortex
cortex-explain-error() {
    local last_cmd=$(fc -ln -1)
    local exit_code=$?
    local stderr_content="$1"
    
    local result=$(curl -s -X POST http://localhost:7800/ask \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"This command failed: $last_cmd\\nError: $stderr_content\\nExplain why and how to fix it.\", \"source\": \"smart-terminal\", \"session_id\": \"terminal\"}" 2>/dev/null)
    
    local explanation=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('speak',''))" 2>/dev/null)
    
    if [[ -n "$explanation" ]]; then
        echo "\033[36m  cortex:\033[0m $explanation"
    fi
}

# Hook into command-not-found
command_not_found_handler() {
    echo "\033[31m  command not found: $1\033[0m"
    cortex-explain-error "command not found: $1"
}

# Alias for natural language command generation
# Usage: ? deploy the api to staging
'?'() {
    local query="$*"
    local cmd=$(cortex-generate-command "$query")
    if [[ -n "$cmd" ]]; then
        echo "\033[2m  → $cmd\033[0m"
        print -z "$cmd"  # Put in command buffer (user presses enter to run)
    else
        echo "\033[31m  Couldn't generate a command for that.\033[0m"
    fi
}
