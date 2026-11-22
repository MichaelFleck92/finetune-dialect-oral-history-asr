#!/usr/bin/env python3
"""
What does an annotation in square or round brackets mean?

Every annotation gets exactly one class. It decides two things: whether the
audio at this point may contain speech without text (then the alignment gets
a wildcard that may absorb any amount of audio) and whether a training
segment may extend across this point.

  class       text           wildcard  segment may cross  example
  unclear     [unv]          yes       no                 [unv.], (?), [Tabakregie?], XXX
  gap         [gap]          yes       no                 [erzaehlt von der Reise], [Nachfrage], […]
  nonverbal   [nv]           no        yes                [lacht], [hustet], [deutet auf das Haus]
  pause       (dropped)      no        yes                [kurze Pause], [Schweigen]
  editorial   (dropped)      no        yes                [Anm.: ...], [gemeint: BDM], [19]41, [sic.]
  spoken      content kept   -         yes                [Mustermann] (name noted phonetically)

Anything unknown is treated as gap: at worst this costs one segment cut,
but never lets unspoken or missing text into training.
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
    # order matters: more specific patterns first
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
    # annotations describing conversation are gaps - not editorial
    ("gap", r"^anm\w*\.?:?\s.*" + _GAP_WORDS),
    ("editorial", r"^(anm\b|anmerkung\b|gemeint\b|sic\b|=|vgl\b|in anlehnung an\b)"),
    ("editorial", r"^(sie|er) meint\b|^hat sie früher gemeint$|^zu (seiner|ihrer) (frau|mann|tochter|sohn)$"),
    ("unclear", r"\?\s*\.*$"),                         # [Tabakregie?], (Bauarbeiter?)
    ("gap", _GAP_WORDS),
    ("gap", r"^(\.{2,}|…|\[?…\]?|\.\.\.\.|….)$"),         # [...] = omitted
    ("gap", r"^\d{1,2}:\d{2}(:\d{2})?\s*-\s*\d{1,2}:\d{2}(:\d{2})?$"),   # [59:28-1:01:03]
]
_RULES = [(c, re.compile(p, re.I)) for c, p in _RULES]

NAME_LIKE = re.compile(r"^[A-ZÄÖÜ][\wÄÖÜäöüß'’\-]*(\s+[A-ZÄÖÜ][\wÄÖÜäöüß'’\-]*){0,2}$|^-[a-zäöüß]+$")


def classify(inner, overrides=None, bracket_default="gap", kind="["):
    """inner: content without brackets. overrides: {content: class} from the spec.
    bracket_default: what happens to unknown square brackets; the
    transcripts of the BIK team write unverifiable names phonetically in
    square brackets - there "spoken_if_name" is right."""
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
