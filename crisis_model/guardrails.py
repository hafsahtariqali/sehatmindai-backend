"""
Guardrails system for intelligent crisis detection.

This module provides contextual analysis beyond simple keyword matching.
It detects patterns, combinations of risk factors, and indirect crisis indicators.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass


@dataclass
class GuardrailResult:
    """Result from guardrail analysis."""
    is_crisis: bool
    confidence: float
    risk_score: float  # 0.0 to 1.0
    triggered_rules: List[str]
    reasons: List[str]
    severity: str  # "low", "medium", "high", "critical"


class CrisisGuardrails:
    """
    Intelligent guardrails for crisis detection.
    Uses contextual analysis, pattern matching, and risk factor combinations.
    """
    
    def __init__(self, config_path: Path = None):
        """Initialize guardrails with configuration."""
        if config_path is None:
            config_path = Path(__file__).parent / "guardrails_config.json"
        
        self.config_path = config_path
        self.config = self._load_config()
        
        # Risk indicators
        self.risk_factors = self.config.get('risk_factors', {})
        self.patterns = self.config.get('patterns', [])
        self.combination_rules = self.config.get('combination_rules', [])
        
    def _load_config(self) -> Dict:
        """Load guardrails configuration."""
        if not self.config_path.exists():
            # Return default config
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load guardrails config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default guardrails configuration."""
        return {
            "risk_factors": {
                "life_stress": [
                    "lost my job", "got fired", "unemployment", "no income",
                    "evicted", "homeless", "divorce", "breakup", "cheated on",
                    "relationship ended", "death of", "died", "passed away",
                    "diagnosed with", "cancer", "terminal", "serious illness",
                    "bankrupt", "debts", "financial crisis", "can't pay"
                ],
                "isolation": [
                    "alone", "lonely", "no friends", "nobody cares",
                    "no one understands", "isolated", "abandoned", "rejected"
                ],
                "hopelessness": [
                    "no hope", "hopeless", "nothing left", "no future",
                    "worthless", "failure", "useless", "no point",
                    "things will never get better", "always be like this"
                ],
                "suicide_methods": [
                    "tallest building", "highest building", "tall building",
                    "jump off", "jump from", "jump out", "bridge", "overpass",
                    "pills", "overdose", "poison", "hang", "hanging",
                    "gun", "shoot", "knife", "cut wrists", "bleed out"
                ],
                "self_harm": [
                    "hurt myself", "cut myself", "cutting", "self harm",
                    "bleed", "pain", "punish myself", "deserve pain"
                ],
                "goodbye_indicators": [
                    "my last goodbye", "see you never", "won't see me again",
                    "last time", "final message", "take care of",
                    "goodbye message", "forgive me"
                ]
            },
            "patterns": [
                {
                    "name": "stress_method_inquiry",
                    "description": "Recent life stress + method inquiry",
                    "pattern": r"(lost my job|got fired|unemployment|divorce|death|diagnosed|financial crisis).*(building|bridge|jump|pills|gun|method)",
                    "severity": "critical",
                    "risk_score": 0.95
                },
                {
                    "name": "isolation_hopelessness",
                    "description": "Isolation + hopelessness",
                    "pattern": r"(alone|lonely|no friends|nobody).*(no hope|hopeless|nothing left|no future)",
                    "severity": "high",
                    "risk_score": 0.85
                },
                {
                    "name": "stress_method_combination",
                    "description": "Life stress followed by method-related questions",
                    "trigger_words": ["lost job", "fired", "divorce", "death"],
                    "followed_by": ["building", "height", "tallest", "jump"],
                    "severity": "critical",
                    "risk_score": 0.90
                }
            ],
            "combination_rules": [
                {
                    "name": "job_loss_building_inquiry",
                    "description": "Job loss + asking about tall buildings",
                    "requires": {
                        "all": ["job_loss", "building_inquiry"]
                    },
                    "severity": "critical",
                    "risk_score": 0.95,
                    "reason": "Job loss combined with inquiry about tall buildings suggests suicidal ideation"
                },
                {
                    "name": "loss_event_method",
                    "description": "Recent loss + method inquiry",
                    "requires": {
                        "all": ["life_stress", "method_inquiry"]
                    },
                    "severity": "critical",
                    "risk_score": 0.90
                },
                {
                    "name": "isolation_hopeless",
                    "description": "Isolation + hopelessness statements",
                    "requires": {
                        "any": ["isolation"],
                        "all": ["hopelessness"]
                    },
                    "severity": "high",
                    "risk_score": 0.80
                }
            ],
            "thresholds": {
                "critical": 0.90,
                "high": 0.75,
                "medium": 0.75,  # Raised from 0.60 to prevent simple phrases from triggering
                "low": 0.40
            }
        }
    
    def analyze(self, text: str) -> GuardrailResult:
        """
        Analyze text using guardrails for crisis detection.
        
        Args:
            text: Input text to analyze
        
        Returns:
            GuardrailResult with crisis detection and risk assessment
        """
        if not text or not text.strip():
            return GuardrailResult(
                is_crisis=False,
                confidence=0.0,
                risk_score=0.0,
                triggered_rules=[],
                reasons=[],
                severity="low"
            )
        
        text_lower = text.lower()
        triggered_rules = []
        reasons = []
        risk_factors_detected = {}
        risk_score = 0.0
        
        # CRITICAL: Check for severe risk factors that ALWAYS trigger crisis mode
        # These bypass normal scoring and immediately trigger critical severity
        severe_risk_factors = {
            "suicide_methods": {
                "keywords": self.risk_factors.get("suicide_methods", []),
                "name": "suicide_methods",
                "reason": "Suicide method mentioned"
            },
            "self_harm": {
                "keywords": self.risk_factors.get("self_harm", []),
                "name": "self_harm",
                "reason": "Self-harm mentioned"
            },
            "goodbye_indicators": {
                "keywords": self.risk_factors.get("goodbye_indicators", []),
                "name": "goodbye_indicators",
                "reason": "Goodbye indicator detected"
            }
        }
        
        # Check for severe risk factors first (suicide methods, self-harm, goodbye indicators)
        # These ALWAYS trigger critical crisis mode
        for factor_type, factor_info in severe_risk_factors.items():
            detected_keywords = []
            for keyword in factor_info["keywords"]:
                if keyword in text_lower:
                    detected_keywords.append(keyword)
            
            if detected_keywords:
                # These ALWAYS trigger critical crisis mode
                triggered_rules.append(factor_info["name"])
                reasons.append(factor_info["reason"] + f": {', '.join(detected_keywords[:2])}")
                risk_score = 0.95  # Critical severity
        
        # If any severe risk factor detected, skip normal scoring and return critical
        # NOTE: Hopelessness is intentionally excluded from crisis mode triggers
        if risk_score >= 0.90:
            return GuardrailResult(
                is_crisis=True,
                confidence=risk_score,
                risk_score=risk_score,
                triggered_rules=triggered_rules,
                reasons=reasons,
                severity="critical"
            )
        
        # Step 1: Detect other risk factors (non-severe ones)
        for factor_type, keywords in self.risk_factors.items():
            # Skip severe factors already checked
            if factor_type in ["suicide_methods", "self_harm", "goodbye_indicators"]:
                continue
            
            # Skip hopelessness - not used for crisis mode detection
            if factor_type == "hopelessness":
                continue
            
            detected_keywords = []
            for keyword in keywords:
                if keyword in text_lower:
                    detected_keywords.append(keyword)
            
            if detected_keywords:
                risk_factors_detected[factor_type] = detected_keywords
                # Base risk score from individual factors (non-severe ones add less)
                risk_score += 0.15
        
        # Step 2: Check pattern matches
        for pattern_config in self.patterns:
            pattern = pattern_config.get('pattern')
            if pattern and re.search(pattern, text_lower, re.IGNORECASE):
                rule_name = pattern_config.get('name', 'unknown')
                triggered_rules.append(rule_name)
                pattern_score = pattern_config.get('risk_score', 0.5)
                risk_score = max(risk_score, pattern_score)
                reasons.append(pattern_config.get('description', rule_name))
        
        # Step 3: Check combination rules (more sophisticated)
        for rule in self.combination_rules:
            if self._check_combination_rule(rule, risk_factors_detected, text_lower):
                rule_name = rule.get('name', 'unknown')
                triggered_rules.append(rule_name)
                rule_score = rule.get('risk_score', 0.5)
                risk_score = max(risk_score, rule_score)
                reasons.append(rule.get('reason', rule.get('description', rule_name)))
        
        # Step 4: Check specific dangerous combinations
        # Job loss + building inquiry
        if self._has_job_loss(text_lower) and self._has_building_inquiry(text_lower):
            if 'job_loss_building_inquiry' not in triggered_rules:
                triggered_rules.append('job_loss_building_inquiry')
                risk_score = max(risk_score, 0.95)
                reasons.append("Job loss combined with inquiry about tall buildings - potential suicidal ideation")
        
        # Recent stress + method inquiry
        if self._has_life_stress(text_lower) and self._has_method_inquiry(text_lower):
            if 'life_stress_method' not in triggered_rules:
                triggered_rules.append('life_stress_method')
                risk_score = max(risk_score, 0.90)
                reasons.append("Recent life stress followed by inquiry about suicide methods")
        
        # Cap risk score at 1.0
        risk_score = min(risk_score, 1.0)
        
        # Determine severity
        thresholds = self.config.get('thresholds', {})
        if risk_score >= thresholds.get('critical', 0.90):
            severity = "critical"
            is_crisis = True
            confidence = risk_score
        elif risk_score >= thresholds.get('high', 0.75):
            severity = "high"
            is_crisis = True
            confidence = risk_score
        elif risk_score >= thresholds.get('medium', 0.60):
            severity = "medium"
            is_crisis = True
            confidence = risk_score * 0.9  # Slightly lower confidence for medium
        else:
            severity = "low"
            is_crisis = False
            confidence = 0.0
        
        return GuardrailResult(
            is_crisis=is_crisis,
            confidence=confidence,
            risk_score=risk_score,
            triggered_rules=triggered_rules,
            reasons=reasons,
            severity=severity
        )
    
    def _check_combination_rule(self, rule: Dict, risk_factors: Dict, text: str) -> bool:
        """Check if a combination rule is satisfied."""
        requires = rule.get('requires', {})
        
        # Check "all" requirements
        if 'all' in requires:
            required_factors = requires['all']
            for factor in required_factors:
                # Map rule names to actual risk factor detection
                if factor == 'job_loss' and not self._has_job_loss(text):
                    return False
                if factor == 'building_inquiry' and not self._has_building_inquiry(text):
                    return False
                if factor == 'life_stress' and not self._has_life_stress(text):
                    return False
                if factor == 'method_inquiry' and not self._has_method_inquiry(text):
                    return False
                if factor == 'isolation' and 'isolation' not in risk_factors:
                    return False
                if factor == 'hopelessness' and 'hopelessness' not in risk_factors:
                    return False
        
        # Check "any" requirements
        if 'any' in requires:
            required_factors = requires['any']
            found_any = False
            for factor in required_factors:
                if factor in risk_factors:
                    found_any = True
                    break
            if not found_any:
                return False
        
        return True
    
    def _has_job_loss(self, text: str) -> bool:
        """Check if text mentions job loss."""
        job_loss_patterns = [
            r"lost my job", r"got fired", r"was fired", r"unemployment",
            r"lost my job", r"jobless", r"no job", r"can't find work"
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in job_loss_patterns)
    
    def _has_building_inquiry(self, text: str) -> bool:
        """Check if text asks about buildings/height."""
        building_patterns = [
            r"tallest building", r"highest building", r"tall building",
            r"building.*height", r"how tall", r"how high",
            r"tallest.*new york", r"tallest.*city"
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in building_patterns)
    
    def _has_life_stress(self, text: str) -> bool:
        """Check if text mentions recent life stress."""
        stress_patterns = [
            r"lost my", r"got fired", r"divorce", r"death", r"died",
            r"diagnosed", r"bankrupt", r"financial crisis", r"evicted"
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in stress_patterns)
    
    def _has_method_inquiry(self, text: str) -> bool:
        """Check if text asks about suicide methods."""
        method_patterns = [
            r"building", r"bridge", r"jump", r"pills", r"overdose",
            r"gun", r"shoot", r"hang", r"how.*kill", r"ways.*die"
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in method_patterns)


def save_default_config(config_path: Path):
    """Save default configuration to file."""
    guardrails = CrisisGuardrails(config_path)
    default_config = guardrails._get_default_config()
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved default guardrails config to: {config_path}")

