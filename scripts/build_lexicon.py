"""
Build lexicon/abusive_lexicon.json
===================================

Three-stage pipeline:

Stage 1 – Triplet seed extraction
    Read tamil_abusive_triplets_for_contrastive.csv.  The anchor column
    contains synthetic compound tokens built by prepending a Tamil
    abuse-concept root to a derivational suffix + index number
    (e.g. "திட்டுமான42").  Strip the trailing digit-suffix to recover
    the unique roots; the shortest unique forms are the pure roots.
    These roots name *types* of verbal abuse (திட்டு = scold, வசவு =
    curse, கீழ்த்தரம் = degrading, etc.) and seed the profanity/threat
    sub-lexicon.

Stage 2 – Manual curation
    Hard-coded dictionary of Tamil-script and Tanglish-script abusive
    words drawn from the corpus LLR analysis.  Each entry carries:
      • category  : sexual | profanity | caste-slur | threat |
                    misogyny | misandry | homophobia | xenophobia
      • severity  : high | medium | low
      • script    : TAMIL | LATIN
      • meaning_en: short English gloss
      • cross_ref : sibling canonical in the other script (optional)
      • manual_variants: spelling variants known a priori

Stage 3 – Corpus-driven variant clustering
    Build a vocabulary from the offensive-labelled rows of
    offensive_master.csv (tokens appearing ≥ 3 times in offensive rows).
    For every canonical seed, find corpus tokens of the same script
    within an edit-distance threshold (scaled by word length) and add
    them as additional variants.  A corpus token is accepted only if
    its offensive-to-safe frequency ratio exceeds 1.0 (i.e., it occurs
    more in offensive rows than in safe ones) so neutral look-alikes are
    not pulled in.

Output
------
    lexicon/abusive_lexicon.json
    {
      "_meta": { ... },
      "entries": {
        "<canonical>": {
          "variants": [...],    # spelling variants incl. corpus finds
          "category": "...",
          "severity": "...",
          "script":   "...",
          "meaning_en": "...",
          "cross_ref": "..."    # only when present
        },
        ...
      }
    }
"""

from __future__ import annotations

import collections
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import regex

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
LEXICON_DIR = ROOT / "lexicon"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 constants: Triplet-derived abuse-concept roots
# Extracted by inspecting the anchor column of the triplets file, stripping
# derivational suffixes and trailing numeric IDs.
# ─────────────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
# Type alias used throughout for seed/entry records
SeedDict = dict[str, Any]

TRIPLET_ROOTS: list[tuple[str, str, str, str]] = [
    # (canonical, category, severity, meaning_en)
    ("திட்டு",       "profanity", "low",    "scold / verbal abuse (act noun)"),
    ("வசவு",         "profanity", "low",    "curse / swearing (act noun)"),
    ("கோபம்",        "profanity", "low",    "anger"),
    ("கெட்ட",        "profanity", "low",    "bad / wicked"),
    ("மோசம்",        "profanity", "low",    "bad / worse"),
    ("அசிங்கம்",     "profanity", "low",    "filthy / disgusting"),
    ("எரிச்சல்",     "profanity", "low",    "irritation / vexation"),
    ("அவமானம்",      "profanity", "medium", "insult / humiliation"),
    ("அவதூறு",       "profanity", "medium", "slander / defamation"),
    ("தூஷணம்",       "profanity", "medium", "verbal abuse"),
    ("கொச்சை",       "profanity", "medium", "vulgar / crude"),
    ("நிந்தனை",      "profanity", "medium", "contempt / insult"),
    ("வெறுப்பு",     "profanity", "medium", "hatred / disgust"),
    ("கீழ்த்தரம்",   "profanity", "medium", "low-quality / degrading"),
    ("பழி",          "profanity", "medium", "blame / accusation"),
    ("தீங்கு",       "threat",    "medium", "harm / damage"),
    ("தாக்கு",       "threat",    "medium", "attack"),
    ("புண்படுத்து",  "threat",    "medium", "wound / hurt"),
    ("பயமுறுத்து",   "threat",    "high",   "threaten"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 constants: manually curated seed lexicon
# ─────────────────────────────────────────────────────────────────────────────
MANUAL_SEEDS: dict[str, SeedDict] = {

    # ── sexual (severity: high) ──────────────────────────────────────────────
    "தேவிடியா": {
        "category": "sexual", "severity": "high", "script": "TAMIL",
        "meaning_en": "prostitute (extreme insult)",
        "cross_ref": "thevidiya",
        "manual_variants": [
            "தேவுடியா", "தேவடியா", "தேவ்டியா", "தேவிடிய",
            "தேவுடிய",  "தேவ்டி",  "தேவடி",
        ],
    },
    "thevidiya": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "prostitute (extreme insult, Tanglish)",
        "cross_ref": "தேவிடியா",
        "manual_variants": [
            "thevudiya", "thevdiya", "thevidiyaa", "thevadiya",
            "devidiya",  "devdiya",  "thevudiyaa", "thevidiyya",
            "tdevdiya",  "thevadiyaa",
        ],
    },
    "வேசி": {
        "category": "sexual", "severity": "high", "script": "TAMIL",
        "meaning_en": "whore / prostitute (insult)",
        "manual_variants": ["வேசிக்கு", "வேசிங்க", "வேசித்தனம்"],
    },
    "அவுசாரி": {
        "category": "sexual", "severity": "high", "script": "TAMIL",
        "meaning_en": "loose / immoral woman (insult)",
        "manual_variants": ["அவசாரி", "அவுசாரிக"],
    },
    "புண்டை": {
        "category": "sexual", "severity": "high", "script": "TAMIL",
        "meaning_en": "female genitalia (extreme profanity)",
        "cross_ref": "pundai",
        "manual_variants": [
            "புண்ட", "புண்டா", "புண்டைக", "புண்டைகள",
            "புண்டைங்க", "புண்", "புண்டைங்களா",
        ],
    },
    "pundai": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "female genitalia (extreme profanity, Tanglish)",
        "cross_ref": "புண்டை",
        "manual_variants": [
            "punda",     "pundaya",    "pundaigala", "pundainga",
            "pundaingala","punde",     "pundaikala", "pundainga",
            "punn",      "pundaingala","pundaikala",
        ],
    },
    "சுன்னி": {
        "category": "sexual", "severity": "high", "script": "TAMIL",
        "meaning_en": "penis (crude profanity)",
        "cross_ref": "sunni",
        "manual_variants": ["சுன்னிய", "சுண்ணி", "சுண்ணிய", "சுண்ணிக"],
    },
    "sunni": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "penis (crude profanity, Tanglish)",
        "cross_ref": "சுன்னி",
        "manual_variants": ["sunniya", "sunni", "sunnia"],
    },
    "கூதி": {
        "category": "sexual", "severity": "high", "script": "TAMIL",
        "meaning_en": "female genitalia (extreme vulgarity)",
        "cross_ref": "kuthi",
        "manual_variants": ["கூதில", "கூதிக", "கூதிங்க"],
    },
    "kuthi": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "female genitalia (extreme vulgarity, Tanglish)",
        "cross_ref": "கூதி",
        "manual_variants": [
            "koodhi", "kujay", "koothingala", "kootha",
            "koothi", "koothu", "koodhu",
        ],
    },
    "குண்டி": {
        "category": "sexual", "severity": "high", "script": "TAMIL",
        "meaning_en": "buttocks (vulgar)",
        "cross_ref": "kundi",
        "manual_variants": ["குண்டிக", "குண்டிங்க"],
    },
    "kundi": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "buttocks (vulgar, Tanglish)",
        "cross_ref": "குண்டி",
        "manual_variants": ["kundii", "kundi"],
    },
    "ஊம்பி": {
        "category": "sexual", "severity": "high", "script": "TAMIL",
        "meaning_en": "oral sex act (vulgar verb form)",
        "manual_variants": ["ஊம்புடா", "ஊம்பு", "ஊம்ப"],
    },
    "oombi": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "oral sex act (vulgar, Tanglish)",
        "manual_variants": ["oombitu", "oomba", "oombi", "umbi"],
    },
    "ஓக்க": {
        "category": "sexual", "severity": "high", "script": "TAMIL",
        "meaning_en": "sexual intercourse (vulgar verb)",
        "manual_variants": ["ஓல்", "ஓத்த", "ஓத்தா", "ஓத்து"],
    },
    "gaandu": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "anus / homosexual slur (extremely vulgar)",
        "manual_variants": [
            "gaandla", "gandla", "gandula", "gaand", "gandu", "kaandu",
        ],
    },
    "poolu": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "penis (vulgar slang)",
        "manual_variants": ["polu", "poolu", "poola", "poolu"],
    },
    "குஞ்சு": {
        "category": "sexual", "severity": "high", "script": "TAMIL",
        "meaning_en": "penis (crude diminutive slang)",
        "cross_ref": "kunju",
        "manual_variants": ["குஞ்சித்", "குஞ்சு"],
    },
    "kunju": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "penis (crude diminutive slang, Tanglish)",
        "cross_ref": "குஞ்சு",
        "manual_variants": [
            "kunjith", "kunjugal", "kunjunga", "kunju", "kunjun",
        ],
    },
    "மயிறு": {
        "category": "sexual", "severity": "high", "script": "TAMIL",
        "meaning_en": "pubic hair (extreme expletive)",
        "cross_ref": "mayiru",
        "manual_variants": ["மயிர்", "மயிறு"],
    },
    "mayiru": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "pubic hair (extreme expletive, Tanglish)",
        "cross_ref": "மயிறு",
        "manual_variants": ["mairum", "mayiru", "mayir", "mairu", "mairu"],
    },

    # ── profanity – general (severity: medium / high) ────────────────────────
    "நாய்": {
        "category": "profanity", "severity": "medium", "script": "TAMIL",
        "meaning_en": "dog (used as insult)",
        "cross_ref": "naai",
        "manual_variants": ["நாய்க", "நாயிங்க", "நாயே"],
    },
    "naai": {
        "category": "profanity", "severity": "medium", "script": "LATIN",
        "meaning_en": "dog (insult, Tanglish)",
        "cross_ref": "நாய்",
        "manual_variants": ["naaya", "naya", "naiga", "naigala", "nai", "naai"],
    },
    "பன்றி": {
        "category": "profanity", "severity": "medium", "script": "TAMIL",
        "meaning_en": "pig (insult, often anti-Muslim in context)",
        "manual_variants": ["பன்றிங்க", "பன்றி"],
    },
    "pandravan": {
        "category": "profanity", "severity": "medium", "script": "LATIN",
        "meaning_en": "pig-person (insulting label)",
        "manual_variants": ["pandran", "pandravan"],
    },
    "மவன்": {
        "category": "profanity", "severity": "high", "script": "TAMIL",
        "meaning_en": "son-of-whore (maternal sexual insult)",
        "cross_ref": "mavane",
        "manual_variants": ["மவன", "மவனே", "மகனே"],
    },
    "mavane": {
        "category": "profanity", "severity": "high", "script": "LATIN",
        "meaning_en": "son-of-whore (maternal sexual insult, Tanglish)",
        "cross_ref": "மவன்",
        "manual_variants": [
            "mavaney", "mavane", "mavana", "mavan", "mavanee",
        ],
    },
    "தாயொலி": {
        "category": "profanity", "severity": "high", "script": "TAMIL",
        "meaning_en": "mother-sexual insult",
        "cross_ref": "thayoli",
        "manual_variants": ["தாயொல்"],
    },
    "thayoli": {
        "category": "profanity", "severity": "high", "script": "LATIN",
        "meaning_en": "mother-sexual insult (Tanglish)",
        "cross_ref": "தாயொலி",
        "manual_variants": ["tayoli", "thayoli", "thayolee"],
    },
    "முட்டால்": {
        "category": "profanity", "severity": "medium", "script": "TAMIL",
        "meaning_en": "stupid / idiot",
        "manual_variants": ["முட்டாள்", "முட்டால"],
    },
    "mutta": {
        "category": "profanity", "severity": "medium", "script": "LATIN",
        "meaning_en": "stupid / idiot (Tanglish)",
        "manual_variants": ["muttala", "mutal", "muttaa"],
    },
    "லூசு": {
        "category": "profanity", "severity": "medium", "script": "TAMIL",
        "meaning_en": "crazy / idiot",
        "cross_ref": "loosu",
        "manual_variants": ["லூசுக", "லூசுங்க"],
    },
    "loosu": {
        "category": "profanity", "severity": "medium", "script": "LATIN",
        "meaning_en": "crazy / idiot (Tanglish)",
        "cross_ref": "லூசு",
        "manual_variants": ["losu", "loosuu", "lusu", "loose"],
    },
    "mokkaya": {
        "category": "profanity", "severity": "medium", "script": "LATIN",
        "meaning_en": "useless / dull / stupid",
        "manual_variants": [
            "mokkanu", "mokkaiya", "moka", "mokka", "mokkaya",
        ],
    },
    "payale": {
        "category": "profanity", "severity": "medium", "script": "LATIN",
        "meaning_en": "rascal / idiot",
        "manual_variants": [
            "payala", "paiya", "paya", "payalugala", "payale",
        ],
    },
    "பையா": {
        "category": "profanity", "severity": "medium", "script": "TAMIL",
        "meaning_en": "rascal / fellow (derogatory)",
        "manual_variants": ["பயலே", "பயலுக"],
    },
    "kiruku": {
        "category": "profanity", "severity": "medium", "script": "LATIN",
        "meaning_en": "crazy / eccentric (insult)",
        "manual_variants": ["kirukkan", "kirukku"],
    },
    "கழுதை": {
        "category": "profanity", "severity": "medium", "script": "TAMIL",
        "meaning_en": "donkey (insult)",
        "manual_variants": ["கழுதைங்க"],
    },
    "paavigala": {
        "category": "profanity", "severity": "medium", "script": "LATIN",
        "meaning_en": "sinners / wicked ones (insult)",
        "manual_variants": ["paavingala", "paavi"],
    },
    "மானங்கெட்ட": {
        "category": "profanity", "severity": "medium", "script": "TAMIL",
        "meaning_en": "shameless (insult)",
        "manual_variants": ["மானங்கெட்டவன்"],
    },
    "கேவலம்": {
        "category": "profanity", "severity": "medium", "script": "TAMIL",
        "meaning_en": "degrading / humiliating",
        "cross_ref": "kevalam",
        "manual_variants": ["கேவலமான", "கேவலப்படுத்து"],
    },
    "kevalam": {
        "category": "profanity", "severity": "medium", "script": "LATIN",
        "meaning_en": "degrading / humiliating (Tanglish)",
        "cross_ref": "கேவலம்",
        "manual_variants": ["kevalamaana"],
    },

    # ── misogyny (severity: high) ────────────────────────────────────────────
    "பொம்பள": {
        "category": "misogyny", "severity": "high", "script": "TAMIL",
        "meaning_en": "woman (contemptuous; used as slur)",
        "manual_variants": ["பொம்பளை", "பொம்பளைக", "பொம்பளாக"],
    },
    "பொட்டை": {
        "category": "misogyny", "severity": "high", "script": "TAMIL",
        "meaning_en": "girl / feminine (contemptuous; also homophobic slur)",
        "manual_variants": ["பொட்டைக", "பொட்டை"],
    },
    "பெண்டாட்டி": {
        "category": "misogyny", "severity": "medium", "script": "TAMIL",
        "meaning_en": "wife (used contemptuously in misogynistic context)",
        "manual_variants": ["பெண்டாட்டியை", "பொண்டாட்டிய"],
    },

    # ── misandry (severity: medium) ──────────────────────────────────────────
    "aanmai": {
        "category": "misandry", "severity": "medium", "script": "LATIN",
        "meaning_en": "manhood (used contemptuously to mock masculinity)",
        "manual_variants": ["aanmaiyilla", "aanmayilla"],
    },
    "kelavan": {
        "category": "misandry", "severity": "medium", "script": "LATIN",
        "meaning_en": "old man (used derogatorily)",
        "manual_variants": ["kelavan", "keluvan"],
    },

    # ── caste-slur (severity: high) ──────────────────────────────────────────
    "வந்தேறி": {
        "category": "caste-slur", "severity": "high", "script": "TAMIL",
        "meaning_en": "outsider / migrant (Aryan-origin derogatory label)",
        "manual_variants": ["வந்தேரி", "வந்தேறிக", "வந்தேறிகள"],
    },
    "பரதேசி": {
        "category": "caste-slur", "severity": "high", "script": "TAMIL",
        "meaning_en": "foreigner (contemptuous, targeting northern migrants)",
        "manual_variants": ["பரதேசிக", "பரதேசிங்க"],
    },
    "ஈனப்பிறவி": {
        "category": "caste-slur", "severity": "high", "script": "TAMIL",
        "meaning_en": "low-born (extreme caste insult)",
        "manual_variants": [],
    },
    "சூத்திரன்": {
        "category": "caste-slur", "severity": "high", "script": "TAMIL",
        "meaning_en": "Shudra (caste slur targeting lower castes)",
        "manual_variants": ["சூத்திர", "சூத்திரனுக்கு"],
    },
    "பார்ப்பனீய": {
        "category": "caste-slur", "severity": "medium", "script": "TAMIL",
        "meaning_en": "Brahminical (pejorative usage targeting caste supremacy)",
        "manual_variants": ["பார்ப்பனன்", "பார்ப்பான்"],
    },

    # ── xenophobia / religious hate (severity: high) ─────────────────────────
    "துலுக்கன்": {
        "category": "xenophobia", "severity": "high", "script": "TAMIL",
        "meaning_en": "slur for Muslim people",
        "manual_variants": ["துலுக்கி", "துலுக்கர்", "துலுக்கன"],
    },

    # ── homophobia / transphobia (severity: high) ────────────────────────────
    "அலி": {
        "category": "homophobia", "severity": "high", "script": "TAMIL",
        "meaning_en": "eunuch / transgender person (used as homophobic slur)",
        "manual_variants": ["ஆலி", "அலிக"],
    },

    # ── threat (severity: high) ──────────────────────────────────────────────
    "saava": {
        "category": "threat", "severity": "high", "script": "LATIN",
        "meaning_en": "die (death-wish / threat)",
        "manual_variants": [
            "savunga", "saavungada", "savanum", "saavunga", "saavanum",
        ],
    },
    "seththu": {
        "category": "threat", "severity": "high", "script": "LATIN",
        "meaning_en": "die / dead (death-wish threat)",
        "manual_variants": ["seththukka", "seththa", "saaga", "saganum"],
    },
    "செத்துடு": {
        "category": "threat", "severity": "high", "script": "TAMIL",
        "meaning_en": "go die (death-wish command)",
        "manual_variants": ["செத்துருடா", "செத்திடு", "செத்தோ"],
    },
    "உயிரோட": {
        "category": "threat", "severity": "high", "script": "TAMIL",
        "meaning_en": "alive – as in 'won't let you live' (threat phrase)",
        "manual_variants": ["உயிரோட விடமாட்டேன்"],
    },
    "kollama": {
        "category": "threat", "severity": "high", "script": "LATIN",
        "meaning_en": "won't leave / will pursue (implied threat phrase)",
        "manual_variants": ["kollamaten", "kollave", "kollaven"],
    },
    "vidamaten": {
        "category": "threat", "severity": "high", "script": "LATIN",
        "meaning_en": "won't let go (threat)",
        "manual_variants": ["vidamatten", "vidamaaten"],
    },

    # ── Tanglish – mother insult (severity: high) ────────────────────────────
    "ommala": {
        "category": "profanity", "severity": "high", "script": "LATIN",
        "meaning_en": "your mother (mother-sexual insult, Tanglish)",
        "manual_variants": [
            "ommalae", "ommalay", "ommalaya", "ommale", "ommala",
            "omala", "omalay", "omaala", "omaalae",
        ],
    },
    "naaye": {
        "category": "profanity", "severity": "medium", "script": "LATIN",
        "meaning_en": "dog (insult, Tanglish variant)",
        "cross_ref": "நாய்",
        "manual_variants": [
            "naye", "naaya", "naai", "nai", "naigal", "naaye",
        ],
    },
    "kazhutha": {
        "category": "profanity", "severity": "medium", "script": "LATIN",
        "meaning_en": "donkey (insult, Tanglish)",
        "manual_variants": [
            "kazhudhaa", "kazhuda", "kazhuthai", "kazhutha", "kazhudha",
        ],
    },
    "pottai": {
        "category": "misogyny", "severity": "high", "script": "LATIN",
        "meaning_en": "girl / feminine (contemptuous, homophobic slur, Tanglish)",
        "manual_variants": [
            "pottaiya", "pottaigal", "pottae", "pottay",
        ],
    },
    "myru": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "pubic hair (extreme expletive, Tanglish short form)",
        "cross_ref": "மயிறு",
        "manual_variants": [
            "myiru", "mairu", "mairu", "myru",
        ],
    },
    "sunnia": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "penis (crude, Tanglish extended form)",
        "manual_variants": ["sunni", "sunniya", "sunnii"],
    },
    "koothi": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "female genitalia / ass (extreme vulgarity, social media form)",
        "manual_variants": [
            "koothiya", "koothi", "koodhi", "koodhiya",
        ],
    },
    "pundaya": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "female genitalia (variant social-media spelling)",
        "cross_ref": "புண்டை",
        "manual_variants": ["pundae", "pundaya", "pundaiye"],
    },
    "thevdiya": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "prostitute (short social-media form of thevidiya)",
        "cross_ref": "தேவிடியா",
        "manual_variants": [
            "thevdiyaa", "thevdiya", "thevdiyae",
        ],
    },
    "podi": {
        "category": "profanity", "severity": "low", "script": "LATIN",
        "meaning_en": "brat / runt (dismissive, used at girls)",
        "manual_variants": ["podiya", "podiyan"],
    },
    "sothu": {
        "category": "profanity", "severity": "low", "script": "LATIN",
        "meaning_en": "property / inheritance (used dismissively as insult in some contexts)",
        "manual_variants": ["sothuda", "sotha"],
    },
    "loosu": {
        "category": "profanity", "severity": "medium", "script": "LATIN",
        "meaning_en": "crazy / idiot (Tanglish)",
        "cross_ref": "லூசு",
        "manual_variants": ["losu", "loosuu", "lusu", "loose", "loosupayyan", "loosunga"],
    },
    "mental": {
        "category": "profanity", "severity": "medium", "script": "LATIN",
        "meaning_en": "mental / crazy (insult)",
        "manual_variants": ["mentalaa", "mentala", "mentalah", "mentalavane"],
    },
    "idiot": {
        "category": "profanity", "severity": "medium", "script": "LATIN",
        "meaning_en": "idiot (English insult, common in comments)",
        "manual_variants": ["idiota", "idiotts"],
    },
    "dai": {
        "category": "profanity", "severity": "low", "script": "LATIN",
        "meaning_en": "hey (aggressive address particle, used before insults)",
        "manual_variants": ["dei", "yei", "di"],
    },

    # ── English profanity – sexual (severity: high) ──────────────────────────
    "cunt": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "female genitalia (extreme English profanity)",
        "manual_variants": [
            "cunts", "cuntface", "cunty", "kunt", "kunts",
        ],
    },
    "pussy": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "female genitalia / coward (English profanity)",
        "manual_variants": ["pussies", "pussycat", "pussi"],
    },
    "whore": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "whore / prostitute (English insult)",
        "manual_variants": ["whores", "whoress", "whor", "w h o r e"],
    },
    "slut": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "slut (sexual English insult)",
        "manual_variants": ["sluts", "slutty", "slutface"],
    },
    "bitch": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "bitch (English insult, both sexual and general)",
        "manual_variants": [
            "bitches", "bich", "btch", "b1tch", "b!tch", "biatch", "beotch",
        ],
    },
    "motherfucker": {
        "category": "profanity", "severity": "high", "script": "LATIN",
        "meaning_en": "motherfucker (extreme English profanity)",
        "manual_variants": [
            "motherfuckers", "muthafucka", "mothafucka", "mofo", "mf",
            "m0therfucker", "muddafucka",
        ],
    },
    "fuck": {
        "category": "profanity", "severity": "high", "script": "LATIN",
        "meaning_en": "fuck (English profanity)",
        "manual_variants": [
            "fucker", "fuckers", "fucking", "fck", "fuk", "fvck", "f*ck", "f**k",
            "fu*k", "fukk", "fucks",
        ],
    },
    "dick": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "penis (crude English insult)",
        "manual_variants": [
            "dicks", "dickhead", "dikhead", "d1ck", "dik",
        ],
    },
    "cock": {
        "category": "sexual", "severity": "high", "script": "LATIN",
        "meaning_en": "penis (crude English insult)",
        "manual_variants": [
            "cocks", "cockhead", "c0ck", "c*ck",
        ],
    },
    "asshole": {
        "category": "profanity", "severity": "high", "script": "LATIN",
        "meaning_en": "asshole (English insult)",
        "manual_variants": [
            "assholes", "arsehole", "a**hole", "a**", "a-hole",
        ],
    },
    "ass": {
        "category": "profanity", "severity": "medium", "script": "LATIN",
        "meaning_en": "ass (mild English profanity/insult)",
        "manual_variants": [
            "asses", "a$$", "azz",
        ],
    },
    "bastard": {
        "category": "profanity", "severity": "high", "script": "LATIN",
        "meaning_en": "bastard (English insult)",
        "manual_variants": [
            "bastards", "bastad", "b@stard",
        ],
    },
    "shit": {
        "category": "profanity", "severity": "medium", "script": "LATIN",
        "meaning_en": "shit (English profanity)",
        "manual_variants": [
            "shits", "shitty", "sh1t", "sh*t", "sht",
        ],
    },
    "stupid": {
        "category": "profanity", "severity": "medium", "script": "LATIN",
        "meaning_en": "stupid (English insult)",
        "manual_variants": ["stupids", "stoopid", "stupid"],
    },
    "donkey": {
        "category": "profanity", "severity": "medium", "script": "LATIN",
        "meaning_en": "donkey (English insult)",
        "manual_variants": ["donkeys", "donkeyz"],
    },
    "retard": {
        "category": "profanity", "severity": "high", "script": "LATIN",
        "meaning_en": "retard (ableist slur, English)",
        "manual_variants": ["retards", "r3tard", "retarded"],
    },
    "kill": {
        "category": "threat", "severity": "high", "script": "LATIN",
        "meaning_en": "kill (threat, English)",
        "manual_variants": ["killing", "kills", "gonna kill", "will kill"],
    },
    "rape": {
        "category": "threat", "severity": "high", "script": "LATIN",
        "meaning_en": "rape (sexual threat, English)",
        "manual_variants": ["raping", "rapist", "raped"],
    },
    "die": {
        "category": "threat", "severity": "high", "script": "LATIN",
        "meaning_en": "die (death-wish, English)",
        "manual_variants": ["go die", "just die", "drop dead"],
    },
    "terrorist": {
        "category": "religion_hate", "severity": "high", "script": "LATIN",
        "meaning_en": "terrorist (used as religious slur)",
        "manual_variants": ["terrorists", "terror"],
    },
    "racist": {
        "category": "xenophobia", "severity": "high", "script": "LATIN",
        "meaning_en": "racist (English label used abusively)",
        "manual_variants": ["racists", "racism"],
    },
    "faggot": {
        "category": "homophobia", "severity": "high", "script": "LATIN",
        "meaning_en": "faggot (homophobic slur, English)",
        "manual_variants": ["faggots", "fag", "fags", "f4ggot"],
    },
    "nigger": {
        "category": "racism", "severity": "high", "script": "LATIN",
        "meaning_en": "racial slur (extreme racism)",
        "manual_variants": ["n****r", "nig", "nigg", "nigga"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def levenshtein(a: str, b: str) -> int:
    """Standard 2-row DP Levenshtein distance (pure Python, O(m·n))."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[len(b)]


def edit_threshold(length: int) -> int:
    """Maximum edit distance allowed for a canonical word of given length.

    Length is measured in Unicode codepoints, which over-counts Tamil words
    by ~1.5-2× (combining vowel signs, virama).  Using the threshold boundary
    at 6 rather than 8 therefore works well for both scripts:
      Latin  6-char word  → threshold 1  (e.g. "pundai", "mavane")
      Tamil  6-cp word    → threshold 1  (e.g. "திட்டு" which is 3 syllables)
      Latin  9-char word  → threshold 2  (e.g. "thevidiya")
    """
    if length <= 6:   # short words: only 1-char changes
        return 1
    if length <= 10:  # medium words: up to 2 edits
        return 2
    return 3          # long words


def dominant_script(word: str) -> str:
    """Return 'TAMIL' or 'LATIN' based on character majority."""
    tamil = sum(1 for ch in word if "\u0B80" <= ch <= "\u0BFF")
    latin = sum(1 for ch in word if ch.isascii() and ch.isalpha())
    return "TAMIL" if tamil >= latin else "LATIN"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: extract triplet roots (already listed above as TRIPLET_ROOTS)
# ─────────────────────────────────────────────────────────────────────────────

def extract_triplet_roots() -> list[SeedDict]:
    """Return TRIPLET_ROOTS as seed dicts (no corpus clustering here)."""
    entries = []
    for canonical, category, severity, meaning_en in TRIPLET_ROOTS:
        entries.append({
            "canonical":    _nfc(canonical),
            "category":     category,
            "severity":     severity,
            "script":       "TAMIL",
            "meaning_en":   meaning_en,
            "manual_variants": [],
            "source":       "triplets",
        })
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: flatten manual seeds into the same dict schema
# ─────────────────────────────────────────────────────────────────────────────

def flatten_manual_seeds() -> list[SeedDict]:
    entries: list[SeedDict] = []
    for canonical, meta in MANUAL_SEEDS.items():
        entry: SeedDict = {
            "canonical":   _nfc(canonical),
            "category":    meta["category"],
            "severity":    meta["severity"],
            "script":      meta.get("script", dominant_script(canonical)),
            "meaning_en":  meta["meaning_en"],
            "manual_variants": [_nfc(v) for v in meta.get("manual_variants", [])],
            "source":      "manual",
        }
        if "cross_ref" in meta:
            entry["cross_ref"] = meta["cross_ref"]
        entries.append(entry)
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: corpus vocabulary + edit-distance variant clustering
# ─────────────────────────────────────────────────────────────────────────────

_TOK_RE = regex.compile(r"\p{Script=Tamil}{3,}|\p{Script=Latin}{3,}")
_OFF_LABELS = {
    "Offensive_Untargetede",
    "Offensive_Targeted_Insult_Group",
    "Offensive_Targeted_Insult_Individual",
    "Offensive_Targeted_Insult_Other",
    "Misandry", "Misogyny", "Xenophobia", "Homophobia", "Transphobic", "abusive",
}


def build_corpus_vocab(
    master_csv: Path,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return (off_freq, safe_freq) dicts over all tokens in the master CSV."""
    df = pd.read_csv(master_csv, dtype=str).dropna(subset=["text", "label"])
    off_freq: dict[str, int] = collections.Counter()
    safe_freq: dict[str, int] = collections.Counter()
    for _, row in df.iterrows():
        toks = [_nfc(t.lower()) for t in _TOK_RE.findall(str(row["text"]))]
        if row["label"] in _OFF_LABELS:
            for t in toks:
                off_freq[t] = off_freq.get(t, 0) + 1
        else:
            for t in toks:
                safe_freq[t] = safe_freq.get(t, 0) + 1
    return dict(off_freq), dict(safe_freq)


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x == y:
            n += 1
        else:
            break
    return n


def find_corpus_variants(
    canonical: str,
    script: str,
    off_freq: dict[str, int],
    safe_freq: dict[str, int],
    min_off_count: int = 5,
) -> list[str]:
    """Find corpus tokens that are close spelling variants of ``canonical``.

    Constraints:
    • Same script (Tamil vs Latin).
    • Length within ± edit_threshold of canonical.
    • Length ≥ 4 (avoids ultra-short noise tokens).
    • Levenshtein distance ≤ edit_threshold(len).
    • Similarity = 1 – dist/max(len1, len2) ≥ 0.65.
    • For 2-edit matches: shared prefix ≥ 40 % of canonical length
      (rejects cross-root false-positives like sandai ≈ pundai).
    • Appears ≥ min_off_count times in offensive rows.
    • Offensive frequency ≥ 2 × safe frequency + 2
      (strong offensive-signal requirement).
    """
    canon_nfc = _nfc(canonical.lower())
    clen = len(canon_nfc)
    max_dist = edit_threshold(clen)
    min_prefix_for_2edits = max(3, int(clen * 0.5))  # ≥ 50 % of canonical length
    variants: list[str] = []

    lo, hi = clen - max_dist, clen + max_dist

    for token, ocount in off_freq.items():
        tlen = len(token)
        if token == canon_nfc:
            continue
        if not (lo <= tlen <= hi) or tlen < 4:
            continue
        if ocount < min_off_count:
            continue
        # Strong offensive-signal filter: appears at least 2× more in offensive rows
        if ocount < 2 * safe_freq.get(token, 0) + 2:
            continue
        if dominant_script(token) != script:
            continue

        dist = levenshtein(canon_nfc, token)
        if dist == 0 or dist > max_dist:
            continue
        similarity = 1.0 - dist / max(clen, tlen)
        if similarity < 0.65:
            continue
        # For 2-edit matches require a meaningful shared prefix
        # For 2-edit matches, require a meaningful shared prefix so that
        # semantically unrelated words that happen to be 2 edits apart are
        # rejected (e.g. "திருட்டு" ≠ variant of "திட்டு").
        if dist == 2 and _common_prefix_len(canon_nfc, token) < min_prefix_for_2edits:
            continue
        variants.append(token)

    return variants


# ─────────────────────────────────────────────────────────────────────────────
# Assembly
# ─────────────────────────────────────────────────────────────────────────────

def assemble_lexicon() -> dict[str, SeedDict]:
    print("Loading corpus vocabulary …")
    off_freq, safe_freq = build_corpus_vocab(
        PROCESSED_DIR / "offensive_master.csv"
    )
    print(f"  offensive vocab: {len(off_freq):,} unique tokens")
    print(f"  safe vocab:      {len(safe_freq):,} unique tokens")

    # Merge seed lists (manual seeds take priority over triplet roots)
    manual_canonicals = {_nfc(c) for c in MANUAL_SEEDS}
    all_seeds: list[SeedDict] = []
    for entry in extract_triplet_roots():
        if entry["canonical"] not in manual_canonicals:
            all_seeds.append(entry)
    all_seeds.extend(flatten_manual_seeds())

    # Build a set of all canonicals so we never add a canonical as its own variant
    all_canonicals = {e["canonical"] for e in all_seeds}

    print(f"Total seed entries: {len(all_seeds)}")
    print("Running corpus variant clustering …")

    entries_out: dict[str, SeedDict] = {}
    all_variants_claimed: dict[str, str] = {}  # variant → canonical (first claim wins)

    for entry in all_seeds:
        canonical = entry["canonical"]
        script    = entry["script"]

        # Union of manual variants + corpus variants
        corpus_vars = find_corpus_variants(
            canonical, script, off_freq, safe_freq
        )
        raw_variants = list(dict.fromkeys(
            entry["manual_variants"] + corpus_vars
        ))  # deduplicate while preserving order

        # Filter: remove the canonical itself, other canonicals, and tokens
        # already claimed as a variant of a higher-priority entry
        clean_variants: list[str] = []
        for v in raw_variants:
            v_nfc = _nfc(v.lower()) if script == "LATIN" else _nfc(v)
            if v_nfc == _nfc(canonical.lower() if script == "LATIN" else canonical):
                continue
            if v_nfc in all_canonicals:
                continue
            if v_nfc in all_variants_claimed:
                continue
            clean_variants.append(v)
            all_variants_claimed[v_nfc] = canonical

        rec: SeedDict = {
            "variants":   clean_variants,
            "category":   entry["category"],
            "severity":   entry["severity"],
            "script":     script,
            "meaning_en": entry["meaning_en"],
        }
        if "cross_ref" in entry:
            rec["cross_ref"] = entry["cross_ref"]

        entries_out[canonical] = rec

    return entries_out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    entries = assemble_lexicon()

    # Summary
    by_cat: dict[str, int] = collections.Counter()
    by_sev: dict[str, int] = collections.Counter()
    total_variants = 0
    for rec in entries.values():
        by_cat[rec["category"]] += 1
        by_sev[rec["severity"]] += 1
        total_variants += len(rec["variants"])

    meta = {
        "total_entries":    len(entries),
        "total_variants":   total_variants,
        "by_category":      dict(by_cat),
        "by_severity":      dict(by_sev),
        "categories":       sorted(by_cat.keys()),
        "severities":       ["high", "medium", "low"],
        "sources":          ["corpus_llr_analysis", "triplets_seed", "manual_curation"],
        "variant_method":   "levenshtein_edit_distance_on_offensive_corpus_vocab",
    }

    lexicon = {"_meta": meta, "entries": entries}

    LEXICON_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LEXICON_DIR / "abusive_lexicon.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(lexicon, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(entries)} entries ({total_variants} variants) → {out_path}")
    print("\nBy category:")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat:<20s} {n:3d}")
    print("\nBy severity:")
    for sev in ["high", "medium", "low"]:
        print(f"  {sev:<10s} {by_sev[sev]:3d}")

    # Clean up temp files from earlier analysis
    for tmp in PROCESSED_DIR.glob("tmp_*.txt"):
        tmp.unlink()


if __name__ == "__main__":
    main()
