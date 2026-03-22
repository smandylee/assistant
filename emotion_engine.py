from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class EmotionState:
    label: str = "analytical"
    intensity: float = 0.35
    valence: float = 0.0
    arousal: float = 0.35

    def as_dict(self) -> dict[str, float | str]:
        return {
            "label": self.label,
            "intensity": round(self.intensity, 3),
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
        }


class EmotionEngine:
    def __init__(self) -> None:
        self.state = EmotionState()
        self._last_user_text = ""

    @staticmethod
    def _is_negated(text: str, start: int) -> bool:
        # Check short window before a keyword to catch phrases like "안 고마워".
        left = text[max(0, start - 10) : start]
        return bool(re.search(r"(전혀\s*안|별로\s*안|절대\s*안|안\s*|못\s*|않\s*|아니\s*)$", left))

    def _contains_keyword(self, text: str, pattern: str) -> bool:
        for m in re.finditer(pattern, text):
            if not self._is_negated(text, m.start()):
                return True
        return False

    def _contains_negated_keyword(self, text: str, pattern: str) -> bool:
        for m in re.finditer(pattern, text):
            if self._is_negated(text, m.start()):
                return True
        return False

    def _target_from_text(self, text: str) -> tuple[float, float]:
        t = text.strip().lower()
        if not t:
            return 0.0, 0.3

        valence = 0.0
        arousal = 0.4

        if self._contains_keyword(t, r"(고마워|감사|좋아|최고|굿|잘했|멋져|대박|훌륭)"):
            valence += 0.5
            arousal += 0.05
        if self._contains_negated_keyword(t, r"(고마워|감사|좋아|최고|굿|잘했|멋져|대박|훌륭)"):
            valence -= 0.28
            arousal += 0.08

        if self._contains_keyword(t, r"(느려|답답|짜증|빨리|왜 안|문제|에러|오류|망함|안돼|실패)"):
            valence -= 0.45
            arousal += 0.2
        if self._contains_keyword(t, r"(씨발|ㅅㅂ|병신|좆|꺼져|fuck|shit)"):
            valence -= 0.7
            arousal += 0.3

        if self._contains_keyword(t, r"(어떻게|설명|원리|구조|비교|분석|정리해)"):
            arousal -= 0.08
            valence += 0.05
        if self._contains_keyword(t, r"(궁금|호기심|왜지|뭐지)"):
            arousal += 0.08
            valence += 0.08
        if self._contains_negated_keyword(t, r"(궁금|호기심|왜지|뭐지)"):
            arousal -= 0.06
            valence -= 0.05

        if self._contains_keyword(t, r"(농담|웃기|ㅋㅋ|ㅎㅎ|드립)"):
            valence += 0.2
            arousal += 0.12
        if self._contains_keyword(t, r"(긴급|당장|즉시|바로|급해)"):
            arousal += 0.22
            valence -= 0.08
        if self._contains_keyword(t, r"(미안|죄송|실수했)"):
            valence += 0.12
            arousal -= 0.04
        if self._contains_keyword(t, r"(모르|몰라|알려줘|가르쳐)"):
            arousal += 0.05
            valence += 0.02

        if len(t) < 6:
            arousal -= 0.05
        if len(t) > 50:
            valence -= 0.1
            arousal -= 0.05

        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        return valence, arousal

    def _label_from(self, valence: float, arousal: float, text: str) -> str:
        t = text.lower()

        if re.search(r"(씨발|ㅅㅂ|병신|좆|꺼져|fuck|shit)", t) and arousal >= 0.72 and valence <= -0.62:
            return "angry"
        if re.search(r"(긴급|즉시|당장|급해)", t) and arousal >= 0.62:
            return "urgent"
        if re.search(r"(농담|웃기|ㅋㅋ|ㅎㅎ|드립)", t) and valence >= 0.1:
            return "playful"
        if re.search(r"(미안|죄송|실수했)", t):
            return "supportive"

        if valence <= -0.58 and arousal >= 0.62:
            return "frustrated"
        if valence <= -0.35 and arousal >= 0.48:
            return "annoyed"
        if valence >= 0.52 and arousal >= 0.66:
            return "excited"
        if valence >= 0.36 and arousal >= 0.46:
            return "confident"
        if valence > 0.1 and arousal > 0.6:
            return "cold_amused"
        if re.search(r"(궁금|호기심|왜지|뭐지)", t) and valence >= 0.02:
            return "curious"
        if arousal < 0.28:
            return "calm"
        if arousal < 0.38:
            return "analytical"
        return "neutral"

    def update(self, user_text: str) -> EmotionState:
        target_valence, target_arousal = self._target_from_text(user_text)
        # Faust changes emotional state slowly by design.
        alpha = 0.12
        self.state.valence = (1 - alpha) * self.state.valence + alpha * target_valence
        self.state.arousal = (1 - alpha) * self.state.arousal + alpha * target_arousal
        neg_weighted_valence = abs(self.state.valence) + max(0.0, -self.state.valence) * 0.25
        self.state.intensity = min(1.0, max(neg_weighted_valence, self.state.arousal * 0.7))
        self.state.label = self._label_from(self.state.valence, self.state.arousal, user_text)
        self._last_user_text = user_text
        return self.state

    @staticmethod
    def style_instruction(label: str) -> str:
        mapping = {
            "angry": "강한 불쾌를 표현하되 욕설·모욕은 사용하지 마십시오.",
            "frustrated": "답답함을 드러내되 해결 단계 제시를 우선하십시오.",
            "annoyed": "어조는 차갑고 단호하게 유지하되 공격적 표현은 피하십시오.",
            "urgent": "문장을 짧게 하고 우선순위와 즉시 실행 항목부터 제시하십시오.",
            "supportive": "절제된 공감 한 문장 후, 즉시 해결 행동을 안내하십시오.",
            "confident": "어조는 자신감 있고 간결하게 유지하십시오.",
            "excited": "에너지는 높게, 문장 길이는 짧게 유지하십시오.",
            "cold_amused": "약한 냉소를 허용하되 길게 농담하지 마십시오.",
            "playful": "가벼운 위트는 한 문장 이내로 제한하십시오.",
            "curious": "질문 의도를 파고드는 보충 질문을 1개까지 허용하십시오.",
            "calm": "응답 속도를 낮추고 안정적인 설명 톤을 유지하십시오.",
            "analytical": "감정 표현을 줄이고 분석 중심으로 짧게 답하십시오.",
            "neutral": "기본 파우스트 톤으로 답하십시오.",
        }
        return mapping.get(label, mapping["neutral"])

    @staticmethod
    def elevenlabs_voice_settings(emotion: dict[str, float | str] | None) -> dict[str, float]:
        e = emotion or {}
        label = str(e.get("label", "neutral"))
        intensity = float(e.get("intensity", 0.35))

        # Higher intensity -> slightly lower stability (more dynamic voice).
        stability = max(0.3, min(0.85, 0.74 - intensity * 0.3))
        similarity = max(0.65, min(0.95, 0.82 + intensity * 0.08))

        if label in {"angry", "frustrated", "annoyed"}:
            stability = max(0.3, stability - 0.05)
        elif label in {"calm", "analytical"}:
            stability = min(0.88, stability + 0.08)
        elif label in {"playful", "excited"}:
            stability = max(0.3, stability - 0.03)

        return {"stability": round(stability, 3), "similarity_boost": round(similarity, 3)}
