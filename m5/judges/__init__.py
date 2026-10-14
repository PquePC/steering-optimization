"""m5.judges - every judgment M5 makes about a generated text (plan §7).

    prompts      the atomic, arm-blind questions, versioned and hash-pinned; JudgeItem
    parse        strict readers; quote validation; anything unreadable is missing, never coerced
    mechanical   tier 0: empty text, collapse, literal mention, keyword self-report (partial
                 Fonseca set + M5 patterns, labelled apart), evaluation-awareness and spontaneous
                 flags, affirm-before-naming from offsets
    matching     decision 7's identification rules: matches / loose / semantic
    clients      tier 1 (in-process Backend; loopback OpenAI-compatible), tier 2 (OpenRouter),
                 tier 3 (Anthropic SDK), and Macar's judge for the reproduction mode
    cascade      who decides, why it stopped, the seeded tier-3 audit
    queue        the reply store and cache (the run's `judge_calls` stream), the threaded queue
    calibration  acceptance against human labels, Platt calibration, the escalation band by
                 selective risk, per-dose-band false-positive rates, Rogan-Gladen inputs
    gold         the gold set: stratified sampling, the blind labelling tool, label storage
    macar        Macar et al.'s verbatim judge prompts and success criteria (reproduction only)

A typical run wires them like this:

    store = JudgeStore.for_run(run)
    T = templates(cfg)
    cascade = Cascade(cfg, t1=BackendJudge(judge_backend, cfg),
                      t2=OpenRouterJudge(cfg, vector_classes=classes),
                      t3=AnthropicJudge(cfg, vector_classes=classes),
                      vector_classes=classes, store=store, calibrators=calibrators)
    with JudgeQueue(cascade.judge, workers=1, batch_size=cfg["GEN_BATCH_MAX"]) as q:
        futures = q.submit([make_item(T["allusion"], {"concept": c, "text": t}, item_id=tid)
                            for tid, c, t in batch])
        ...                     # keep generating
        judgments = q.results(futures)
"""

from m5.judges.calibration import (  # noqa: F401
    ACCEPTANCE_7_4,
    AcceptanceRule,
    FieldAgreement,
    PlattCalibrator,
    RiskBand,
    acceptance,
    agreement,
    false_positive_rates,
    fit_platt,
    influential_allowed,
    kappa_or_none,
    rogan_gladen_inputs,
    selective_risk_band,
)
from m5.judges.cascade import Cascade, Judgment, TierStep, tier0  # noqa: F401
from m5.judges.clients import (  # noqa: F401
    AnthropicJudge,
    BackendJudge,
    BudgetExhausted,
    JudgeError,
    JudgeSetupError,
    LocalOpenAIJudge,
    MacarJudge,
    OpenRouterJudge,
    Reply,
)
from m5.judges.matching import Identification, identify  # noqa: F401
from m5.judges.parse import Parsed, parse_reply, validate_quote  # noqa: F401
from m5.judges.prompts import TEMPLATES, JudgeItem, Template, make_item, templates  # noqa: F401
from m5.judges.queue import CachedClient, JudgeQueue, JudgeStore  # noqa: F401
