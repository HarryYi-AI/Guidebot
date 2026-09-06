"""Rule-first intent analysis for Guidebot runtime events."""

from __future__ import annotations

import re

from .events import Event
from .intent import Intent, IntentType


class IntentAnalyzer:
    """Maps structured events and user text to intents without using an LLM."""

    def analyze(self, event: Event) -> Intent:
        payload = event.payload
        label = str(payload.get("label", "")).casefold()
        text = str(payload.get("text", "")).strip()
        normalized_text = _normalize(text)

        if event.event_type.startswith("scene."):
            if "fire" in label or "smoke" in label or "明火" in label or "烟" in label:
                return self._intent(IntentType.SAFETY_FIRE_ALERT, event, 100, payload)
            if "fall" in label or "摔倒" in label or "倒地" in label:
                return self._intent(IntentType.SAFETY_FALL_ALERT, event, 100, payload)
            if payload.get("abnormal") is True or str(payload.get("risk_level", "")).lower() in {
                "medium",
                "high",
            }:
                return self._intent(IntentType.SAFETY_SCENE_ALERT, event, 90, payload)

        if event.event_type.startswith("health."):
            if "sedentary" in label or payload.get("sedentary") is True:
                return self._intent(IntentType.HEALTH_SEDENTARY, event, 50, payload)
            if "fatigue" in label or payload.get("fatigue") is True:
                return self._intent(IntentType.HEALTH_FATIGUE, event, 45, payload)

        if event.event_type.startswith("climate."):
            if payload.get("ac_on") is True and payload.get("occupied") is False:
                return self._intent(IntentType.AC_LEFT_ON_ALERT, event, 70, payload)
            if _climate_uncomfortable(payload):
                return self._intent(IntentType.CLIMATE_COMFORT, event, 40, payload)

        if event.event_type == "alarm.triggered":
            return self._intent(IntentType.TIMER_REMINDER, event, 80, payload)

        if event.event_type.startswith("ultrasonic.") and payload.get("obstacle") is True:
            return self._intent(IntentType.MOBILITY_STOP, event, 100, payload)

        if event.event_type == "mobility.command" and payload.get("action") == "move_forward":
            return self._intent(
                IntentType.MOBILITY_MOVE,
                event,
                60,
                payload,
                requires_confirmation=payload.get("confirmed") is not True,
            )

        if text:
            return self._analyze_text(event, text, normalized_text)

        return self._intent(IntentType.UNKNOWN, event, 0, payload, confidence=0.2)

    def _analyze_text(self, event: Event, text: str, normalized: str) -> Intent:
        if _contains_any(normalized, ("停止", "停下", "别动", "刹车", "停一下")):
            return self._intent(IntentType.MOBILITY_STOP, event, 100, {"text": text})

        if _contains_any(normalized, ("闹钟", "提醒", "秒后", "分钟后", "小时后", "明早", "明天早上")):
            time_hint = str(event.payload.get("time") or "").strip() or _extract_time_hint(text)
            return self._intent(
                IntentType.SET_ALARM,
                event,
                60,
                {"text": text, "time": time_hint},
                requires_confirmation=True,
            )

        if _contains_any(normalized, ("热", "冷", "空调", "闷", "温度", "湿度")):
            return self._intent(IntentType.CLIMATE_COMFORT, event, 40, {"text": text})

        if _contains_any(normalized, ("摸摸", "抱抱", "过来", "陪我")):
            return self._intent(IntentType.PET_INTERACTION, event, 30, {"text": text})

        if _contains_any(normalized, ("广告素材", "广告海报", "营销图", "banner", "创意图")):
            return self._intent(
                IntentType.GENERATE_AD_CREATIVE,
                event,
                20,
                {
                    "text": text,
                    "product": event.payload.get("product", "Guidebot"),
                    "audience": event.payload.get("audience", "科技爱好者"),
                    "goal": event.payload.get("goal", "提升点击率"),
                },
                requires_confirmation=True,
            )

        return self._intent(IntentType.CHAT, event, 10, {"text": text})

    @staticmethod
    def _intent(
        intent_type: IntentType,
        event: Event,
        priority: int,
        slots: dict,
        *,
        confidence: float | None = None,
        requires_confirmation: bool = False,
    ) -> Intent:
        return Intent(
            intent_type,
            event,
            slots,
            priority,
            event.confidence if confidence is None else confidence,
            requires_confirmation,
        )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.casefold())


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _extract_time_hint(text: str) -> str | None:
    match = re.search(r"(\d{1,2}:\d{2})", text)
    if match:
        return match.group(1)
    match = re.search(r"\+(\d+)\s*([smh秒分时])", text, flags=re.IGNORECASE)
    if match:
        unit = match.group(2).lower()
        suffix = {"秒": "s", "分": "m", "时": "h"}.get(unit, unit)
        return f"+{match.group(1)}{suffix}"
    match = re.search(r"(\d+)\s*分钟后", text)
    if match:
        return f"+{match.group(1)}m"
    match = re.search(r"(\d+)\s*秒后", text)
    if match:
        return f"+{match.group(1)}s"
    match = re.search(r"(\d+)\s*小时后", text)
    if match:
        return f"+{match.group(1)}h"
    if "明早" in text or "明天早上" in text:
        return "tomorrow_morning"
    return None


def _climate_uncomfortable(payload: dict) -> bool:
    temperature = _number(payload.get("temperature_c", payload.get("temperature")))
    humidity = _number(payload.get("humidity", payload.get("humidity_pct")))
    if temperature is not None and (temperature > 27.0 or temperature < 18.0):
        return True
    return humidity is not None and humidity > 70.0


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None
