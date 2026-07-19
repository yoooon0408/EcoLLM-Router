from __future__ import annotations

import re
from pathlib import Path

# ── 피처 추출 ─────────────────────────────────────────────────────────────────

_COMPLEX_KW = re.compile(
    r"\b(proof|derive|solve|optimize|algorithm|complexity|theorem|"
    r"differential|integral|quantum|architecture|trade.?off|analyze|"
    r"compare|evaluate|논증|증명|풀어|설계|분석|비교)\b",
    re.IGNORECASE,
)

_SIMPLE_KW = re.compile(
    r"\b(what is|who is|when|where|define|translate|summarize|"
    r"뭐야|언제|어디|정의|번역|요약)\b",
    re.IGNORECASE,
)


def _token_count(prompt: str) -> int:
    return len(prompt.split())


def _char_count(prompt: str) -> int:
    return len(prompt)


def _complex_kw_count(prompt: str) -> int:
    return len(_COMPLEX_KW.findall(prompt))


def _simple_kw_count(prompt: str) -> int:
    return len(_SIMPLE_KW.findall(prompt))


def _question_mark_count(prompt: str) -> int:
    return prompt.count("?")


def _avg_word_length(prompt: str) -> float:
    words = prompt.split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def _extract_features(prompt: str) -> list[float]:
    return [
        _token_count(prompt),
        _char_count(prompt),
        _complex_kw_count(prompt),
        _simple_kw_count(prompt),
        _question_mark_count(prompt),
        _avg_word_length(prompt),
    ]


# ── 규칙 기반 분류기 ──────────────────────────────────────────────────────────

_TOKEN_THRESHOLD = 30
_CHAR_THRESHOLD = 150
_COMPLEX_KW_THRESHOLD = 1
_AVG_WORD_LEN_THRESHOLD = 7


def _rule_based_label(prompt: str) -> int:
    """0 = Flash, 1 = Pro."""
    ckw = _complex_kw_count(prompt)
    skw = _simple_kw_count(prompt)

    reasons = []
    if ckw >= _COMPLEX_KW_THRESHOLD:
        reasons.append(True)
    if _token_count(prompt) >= _TOKEN_THRESHOLD:
        reasons.append(True)
    if _char_count(prompt) >= _CHAR_THRESHOLD:
        reasons.append(True)
    if _avg_word_length(prompt) >= _AVG_WORD_LEN_THRESHOLD:
        reasons.append(True)

    if skw > 0 and not reasons:
        return 0
    return 1 if reasons else 0


# ── ML 모델 로드 (선택적) ─────────────────────────────────────────────────────

_MODEL_PATH = (
    Path(__file__).parents[2] / "classifier" / "artifacts" / "router_model.joblib"
)

_model = None
_use_ml = False

try:
    import joblib  # type: ignore[import]

    if _MODEL_PATH.exists():
        _model = joblib.load(_MODEL_PATH)
        _use_ml = True
except Exception:
    pass


# ── 공개 인터페이스 ───────────────────────────────────────────────────────────

def score(prompt: str) -> float:
    """복잡도 점수를 0.0~1.0으로 반환한다. 높을수록 Pro가 적합."""
    if _use_ml and _model is not None:
        features = [_extract_features(prompt)]
        if hasattr(_model, "predict_proba"):
            return float(_model.predict_proba(features)[0][1])
        return float(_model.predict(features)[0])

    # rule-based: 이진 레이블을 연속 점수로 변환.
    # 임계값 0.5와 충분한 여유를 두어 경계 오판을 줄인다.
    return 0.8 if _rule_based_label(prompt) == 1 else 0.2
