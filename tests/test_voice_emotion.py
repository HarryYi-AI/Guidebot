import pytest

from guidebot.voice.emotion import AffectState, EmpathicStylePolicy


def test_empathic_policy_uses_supportive_style_for_confident_negative_affect() -> None:
    affect = AffectState("sad", valence=-0.7, arousal=0.3, confidence=0.9)

    style = EmpathicStylePolicy().choose(affect)

    assert style.name == "supportive"
    assert style.warmth > style.energy
    assert style.speaking_rate < 1


def test_low_confidence_affect_does_not_overinterpret_user() -> None:
    style = EmpathicStylePolicy().choose(AffectState("angry", -0.9, 1.0, 0.3))

    assert style.name == "neutral-warm"


def test_affect_values_are_bounded() -> None:
    with pytest.raises(ValueError, match="valence"):
        AffectState("invalid", 2.0, 0.5, 0.5)
