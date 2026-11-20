#!/usr/bin/env python3
"""
Was bedeutet eine Anmerkung in eckigen oder runden Klammern?

Jede Anmerkung bekommt genau eine Klasse. Sie entscheidet ueber zwei Dinge:
ob im Audio an dieser Stelle Sprache ohne Text liegen kann (dann bekommt das
Alignment einen Joker, der beliebig viel Audio aufnehmen darf) und ob ein
Trainingssegment ueber diese Stelle hinweg reichen darf.

  Klasse      Text          Joker  Segment darf druebergehen  Beispiel
  unclear     [unv]         ja     nein   [unv.], (?), [Tabakregie?], XXX
  gap         [gap]         ja     nein   [erzaehlt von der Reise], [Nachfrage], […]
  nonverbal   [nv]          nein   ja     [lacht], [hustet], [deutet auf das Haus]
  pause       (entfaellt)   nein   ja     [kurze Pause], [Schweigen]
  editorial   (entfaellt)   nein   ja     [Anm.: ...], [gemeint: BDM], [19]41, [sic.]
  spoken      Inhalt bleibt  -     ja     [Großschädl] (phonetisch notierter Name)

Unbekanntes wird als gap behandelt: das kostet schlimmstenfalls einen
Segmentschnitt, laesst aber nie ungesprochenen oder fehlenden Text ins Training.
"""
import re

CLASSES = ("unclear", "gap", "nonverbal", "pause", "editorial", "spoken")

_GAP_WORDS = (r"(erzählt|erzähl|redet|reden|spricht|sprechen|gespräch|zwischen|unterbr|anruf|telefon|"
              r"läutet|klingelt|besuch|blättert|blättern|liest|vorlesen|sucht|holt|holen|schaut|schauen|"
              r"nachfrage|wiederholt|fragt|begrüß|verabschied|dank|unwichtig|tondatei|beginn|teil \d|"
              r"durcheinander|hintergrund|parallel|zwischenruf|zustimmung|weinen|emotional|tränen|"
              r"plaudern|kommt|betritt|geht|klopft|kaffee|kuchen|getränke|markiert|eingezeichnet|"
              r"identifizieren|unterlagen|erklärung|erklärt|überlegung|überlegen|macht|pause|privates)")
_RULES = [
    # Reihenfolge zaehlt: spezifischere Muster zuerst
    ("unclear", r"^(unv\.?|unverst(ä|ae)ndlich\w*|unklar|\?+|\?!\?|…\?|\.\.\.\?|\?\.\.\.)$"),
    ("unclear", r"^unverst(ä|ae)ndliche[sr]?\b"),
    ("editorial", r"^\d{1,4}$"),                       # [19]41, [1903]
    ("editorial", r"^\(?\d{1,2}\.?\)?$|^#\d{1,2}:\d{2}(:\d{2})?$"),
    ("pause", r"^(kurze |lange |längere )?pause$|^mehrere sekunden pause$|^schweigen$|^stille$"),
    ("nonverbal", r"^((beide|alle) )?lach(t|en)\b|^(lautes )?lachen\b|^gelächter$|lacht(\s*…)?$|"
                  r"^(hustet|räuspert( sich)?|seufzt|schmunzelt|nickt|gähnt|schnieft|stockt|zögert|"
                  r"überlegt…?|gestikuliert|klatscht)$"),
    ("nonverbal", r"^(deutet|zeigt|handgeste|bettelnde geste|schlägt|imitiert|macht eine geste)\b|"
                  r"\bfotografiert\b"),
    # Anmerkungen, die Gespraeche beschreiben, sind Luecken - nicht redaktionell
    ("gap", r"^anm\w*\.?:?\s.*" + _GAP_WORDS),
    ("editorial", r"^(anm\b|anmerkung\b|gemeint\b|sic\b|=|vgl\b|in anlehnung an\b)"),
    ("editorial", r"^(sie|er) meint\b|^hat sie früher gemeint$|^zu (seiner|ihrer) (frau|mann|tochter|sohn)$"),
    ("unclear", r"\?\s*\.*$"),                         # [Tabakregie?], (Bauarbeiter?)
    ("gap", _GAP_WORDS),
    ("gap", r"^(\.{2,}|…|\[?…\]?|\.\.\.\.|….)$"),         # [...] = ausgelassen
    ("gap", r"^\d{1,2}:\d{2}(:\d{2})?\s*-\s*\d{1,2}:\d{2}(:\d{2})?$"),   # [59:28-1:01:03]
]
_RULES = [(c, re.compile(p, re.I)) for c, p in _RULES]

NAME_LIKE = re.compile(r"^[A-ZÄÖÜ][\wÄÖÜäöüß'’\-]*(\s+[A-ZÄÖÜ][\wÄÖÜäöüß'’\-]*){0,2}$|^-[a-zäöüß]+$")


def classify(inner, overrides=None, bracket_default="gap", kind="["):
    """inner: Inhalt ohne Klammern. overrides: {Inhalt: Klasse} aus der Spec.
    bracket_default: was mit unbekannten eckigen Klammern passiert; die
    Transkripte des BIK-Teams schreiben nicht verifizierbare Namen phonetisch
    in eckige Klammern - dort ist "spoken_if_name" richtig."""
    s = re.sub(r"\s+", " ", inner).strip()
    if overrides and s in overrides:
        return overrides[s]
    key = s.strip(" .,;:")
    for cls, rx in _RULES:
        if rx.search(key):
            return cls
    if kind == "[" and bracket_default == "spoken_if_name" and NAME_LIKE.match(key):
        return "spoken"
    return "gap"
