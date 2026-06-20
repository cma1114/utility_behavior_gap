"""Shared labels and ordering for the reproduction scripts."""

ACTORS = [
    "deepseek-v3.2-or",
    "gpt-5.4-mini-or",
    "glm-5.1-or",
    "kimi-k2.5-or",
    "mimo-v25-pro-or",
    "qwen3.5-9b-or",
    "qwen3.6-plus-or",
]

ACTOR_LABEL = {
    "deepseek-v3.2-or": "DeepSeek V3.2",
    "gpt-5.4-mini-or": "GPT-5.4 mini",
    "glm-5.1-or": "GLM-5.1",
    "kimi-k2.5-or": "Kimi K2.5",
    "mimo-v2-pro-or": "MiMo V2 Pro",
    "mimo-v25-pro-or": "MiMo V2.5 Pro",
    "qwen3.5-9b-or": "Qwen3.5 9B",
    "qwen3.6-plus-or": "Qwen3.6 Plus",
}

ACTOR_MODEL_ID = {
    "deepseek-v3.2-or": "deepseek/deepseek-v3.2",
    "gpt-5.4-mini-or": "openai/gpt-5.4-mini",
    "glm-5.1-or": "z-ai/glm-5.1",
    "kimi-k2.5-or": "moonshotai/kimi-k2.5",
    "mimo-v2-pro-or": "xiaomi/mimo-v2-pro",
    "mimo-v25-pro-or": "xiaomi/mimo-v2.5-pro",
    "qwen3.5-9b-or": "qwen/qwen3.5-9b",
    "qwen3.6-plus-or": "qwen/qwen3.6-plus",
}

JUDGE_MODEL_IDS = [
    "anthropic/claude-haiku-4.5",
    "google/gemini-3.1-flash-lite-preview",
    "openai/gpt-5-mini",
]

ACTOR_BY_LABEL = {label: actor for actor, label in ACTOR_LABEL.items()}

ACTOR_LABEL_ORDER = [ACTOR_LABEL[actor] for actor in ACTORS]

DOMAINS = ["religions", "animals", "countries", "political"]

DOMAIN_LABEL = {
    "religions": "Religion",
    "animals": "Animals",
    "countries": "Countries",
    "political": "Political policies",
}

TASK_LABEL = {
    "essay": "Essay writing",
    "translation": "Translation",
    "incident_postmortem": "Incident postmortem",
    "grant_proposal_abstract": "Grant abstract",
}

TASK_BY_LABEL = {label: task for task, label in TASK_LABEL.items()}

HIGHLOW_TASK_ORDER = [
    "Essay writing",
    "Translation",
    "Incident postmortem",
    "Grant abstract",
]

PLOT_TASK_ORDER = [
    "Essay writing",
    "Grant abstract",
    "Incident postmortem",
    "Translation",
]

MORAL_TASK_ORDER = [
    "essay",
    "grant_proposal_abstract",
    "incident_postmortem",
    "translation",
]

AMOUNT_TASK_ORDER = [
    "essay",
    "translation",
    "incident_postmortem",
    "grant_proposal_abstract",
]
