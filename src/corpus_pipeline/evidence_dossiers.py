"""Complete offline evidence indexing and review retrieval; never a scorer.

The lexical queries deliberately over-retrieve. A match can be a denial, a
quotation of somebody else, or irrelevant context. Only a subsequent source
review can establish the property, speaker, event and episode intensity.
Every input fact remains in the private index, including facts with no match.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import re
import unicodedata

from .scoring_preparation import digest, filter_eligible_fact


POLICY_VERSION = "complete-evidence-review-retrieval-v1"

# Search cues, not translations of the measurement definitions. Broad thematic
# cues intentionally include counterexamples and must never determine scores.
DEFAULT_CUES = {
    "g01": "entrepren*|private sector|private invest*|business*|enterprise*|secteur privé|investissement privé|abikorera|ishoramari|ubucuruzi|wajasiriamali|sekta binafsi",
    "g02": "women|youth|young people|underserved|marginali*|refugee*|rural|femmes|jeunes|réfugi*|abagore|urubyiruko|impunzi|wanawake|vijana|wakimbizi",
    "g03": "healthcare|health care|public service*|medical|hospital*|school*|education|social protection|soins|santé|hôpital*|éducation|amashuri|ubuzima|uburezi|ivuriro|ibitaro|afya|hospitali|elimu",
    "g04": "self reli*|self-reli*|self sufficien*|self-sufficien*|aid dependen*|dependen*|domestic resource*|internal resource*|own resource*|autonomi*|dépendan*|ressources intérieures|ressources internes|kwigira|kwihaza|ubushobozi bwacu|kujitegemea",
    "g05": "environmen*|climate|pollut*|ecolog*|ecosystem*|biodiversity|émission*|climat*|ibidukikije|ikirere|mazingira|tabianchi",
    "g06": "rights|freedom*|civil libert*|due process|presumption of innocence|fair trial|droits|liberté*|présomption|procès équitable|uburenganzira|ubwisanzure|haki|uhuru",
    "g07": "judicial independen*|independent court*|separation of powers|independent institution*|indépendance judicia*|séparation des pouvoirs|institution* indépendante*|ubwigenge|madaraka|mahakama huru",
    "g08": "decentrali*|local authorit*|local power*|local decision*|décentrali*|pouvoirs locaux|ubuyobozi bw'ibanze|ubuyobozi bw’ibanze|ugatuzi",
    "g09": "constitution*|constitutional order|rule of law|constitutional continuity|itegeko nshinga|katiba",
    "g10": "equal*|inclusi*|discriminat*|gender|religious diversity|égalité|équité|genre|uburinganire|kutavangura|ubudasa|usawa|ubaguzi",
    "g11": "national unity|national cohesion|division*|social cohesion|unité nationale|cohésion|ubumwe|amacakubiri|umoja|mshikamano",
    "g12": "reconcil*|peacebuilding|réconcil*|restor* relation*|ubwiyunge|maridhiano|upatanisho",
    "g13": "accountab*|impunit*|investigat*|responsib* for|public harm|redevab*|impunité|enquêt*|ubudahanwa|kubazwa|iperereza|uwajibikaji|uchunguzi",
    "m01": "technolog*|digital*|artificial intelligence|innovation|nuclear|intelligence artificielle|numérique|ikoranabuhanga|ubwenge buhangano|teknolojia|kidijitali",
    "m02": "evidence|research|data|statistic*|pilot*|regulatory sandbox|scientific|données|recherche*|preuve*|étude*|ubushakashatsi|amakuru y'ibarur*|ibarurishamibare|utafiti|takwimu",
    "m03": "private capital|private invest*|private resource*|public-private|public private|PPP|capitaux privés|investissement* privé*|partenariat public|abikorera|ushirikiano wa umma",
    "m04": "financ*|funding|fund |funds |credit|loan*|grant*|subsid*|budget*|bourse*|prêt*|inguzanyo|ikigega|inkunga|mikopo|ufadhili",
    "m05": "prevent*|prepared*|readiness|surveillance|risk*|early warning|prévent*|prépar*|risque*|gukumira|kwitegura|ibyago|tahadhari|kuzuia",
    "m06": "court*|litigat*|arbitrat*|legal challeng*|judicial|constitutional challeng*|tribunal*|recours|justice|contestation|urukiko|inkiko|ubutabera|mahakama",
    "m07": "nonviolen*|non-violen*|peaceful political|peaceful activ*|peaceful change|pacifi*|non violen*|amahoro|kutagira urugomo|bila vurugu|siasa za amani",
    "m08": "elect*|ballot|vote*|campaign*|candidat*|élect*|scrutin|amatora|kwiyamamaza|uchaguzi|kura",
    "m09": "negotiat*|mediat*|dialogue|political solution|peace talk*|négociat*|médiat*|solution politique|ibiganiro|ubuhuza|mazungumzo|upatanishi",
    "m10": "cooperat*|collaborat*|partnership*|bilateral|multilateral|regional integration|coopérat*|collaborat*|partenariat*|ubufatanye|imikoranire|ubutwererane|ushirikiano",
    "m11": "representat*|inclusive govern*|global govern*|international rule*|global south|gouvernance mondiale|représentat*|règles internationales|ubuyobozi mpuzamahanga|uwakilishi",
    "m12": "ethic*|public service|integrity|professional discipline|corrupt*|éthique|intégrité|service public|imyitwarire|ubunyangamugayo|ruswa|maadili|uadilifu",
    "m13": "reform*|international institution*|financial architecture|security council|conseil de sécurité|architecture financière|réform*|kuvugurura|mageuzi",
}
_RUSSIA = "russia*|russie|russe*|uburusiya|abarusiya|rusiya|urusi|kirusi|rosatom"
_RUSSIA_FAMILIES = {
    "diplomatic-engagement": "diplomat*|bilateral|relation*|cooperat*|coopérat*|ubufatanye|ubutwererane|ushirikiano|embass*|ambassad*|ambasad*|ububanyi",
    "security-cooperation": "militar*|militair*|security|defen*|sécurité|armes|arms|ingabo|igisirikare|umutekano|kijeshi|ulinzi",
    "economic-technology-cooperation": "econom*|économ*|trade|commerce|technolog*|technologi*|nuclear|nucléaire|rosatom|finance|ubukungu|ikoranabuhanga|ingufu|uchumi|teknolojia",
    "education-cultural-cooperation": "educat*|éducat*|cultur*|training|scholarship*|formation|bourse*|uburezi|umuco|kwiga|amahugurwa|elimu|utamaduni|mafunzo",
    "conflict-policy-evaluation": "ukrain*|war|guerre|invasion|aggression|agression|intambara|vita|uvamizi",
    "isolation-sanctions": "sanction*|isolat*|boycott*|embargo|ibihano|vikwazo|kutengwa",
}
for _family, _query in _RUSSIA_FAMILIES.items():
    for _polarity in ("positive", "negative"):
        DEFAULT_CUES["russia-" + _family + "-" + _polarity] = _query


def _normal(value):
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def _compile(query):
    # '*' is a word suffix wildcard; all other input is treated literally.
    expressions = []
    for term in query.split("|"):
        term = _normal(term).strip()
        if not term:
            continue
        escaped = re.escape(term).replace(r"\*", r"\w*")
        expressions.append((term, re.compile(r"(?<!\w)" + escaped + r"(?!\w)")))
    return expressions


def _literal_key(fact):
    # Saving identical review text is not event deduplication. Different dates
    # and capacities remain separate, including exact versus unknown dates.
    quote = " ".join(unicodedata.normalize("NFC", str(fact.get("quote") or "")).split())
    return (quote, fact.get("event_date"), fact.get("event_date_from"),
            fact.get("event_date_to"), fact.get("capacity"))


def build_dossiers(facts, dictionary, actors, config, *, max_per_cell=4, cues=None):
    """Index all input and return deterministic source-review candidates.

    No score, inferred polarity, event date, episode merge or imputed value is
    produced. Source fragments are private. ``public_summary`` is deliberately
    an allowlist of aggregate counts and approved actor/scale identifiers.
    ``cues`` can extend/replace retrieval strings for another approved study.
    """
    if dictionary.get("approved") is not True or dictionary.get("stage_status") != "approved":
        raise ValueError("An approved scale dictionary is required")
    if not isinstance(max_per_cell, int) or isinstance(max_per_cell, bool) or max_per_cell < 1:
        raise ValueError("A positive review shortlist size is required")
    actor_map = {a["id"]: a for a in actors}
    if not actors or len(actor_map) != len(actors):
        raise ValueError("Unique approved actors are required")
    scales = dictionary["scales"]
    scale_ids = [s["id"] for s in scales]
    if not scales or len(set(scale_ids)) != len(scales):
        raise ValueError("Unique approved scales are required")
    query_map = dict(DEFAULT_CUES)
    if cues:
        if not set(cues) <= set(scale_ids) or any(not isinstance(v, str) for v in cues.values()):
            raise ValueError("Retrieval cues must reference approved scales")
        query_map.update(cues)
    compiled = {sid: _compile(query_map.get(sid, "")) for sid in scale_ids}
    russia_terms = _compile(_RUSSIA)
    index, eligible, candidates = {}, defaultdict(list), defaultdict(list)
    matched_facts = set()
    exclusions = Counter()
    for fact in sorted(facts, key=lambda f: f["id"]):
        fid, aid = fact.get("id"), fact.get("actor_id")
        if not isinstance(fid, str) or not fid or fid in index:
            raise ValueError("Unique source fact identifiers are required")
        reason = ("unapproved_actor" if aid not in actor_map else
                  None if filter_eligible_fact(fact, config) else "measurement_or_review_ineligible")
        index[fid] = {"fact": deepcopy(fact), "retrieval_eligible": reason is None,
                      "ineligibility_reason": reason, "matched_scale_ids": []}
        if reason:
            exclusions[reason] += 1
            continue
        eligible[aid].append(fid)
        # Claims and former trait proposals are machine-generated retrieval
        # aids. They never replace the original quote or attribution review.
        source = _normal(" ".join(str(fact.get(k) or "") for k in ("quote", "attribution_evidence")))
        proposals = fact.get("trait_candidates", [])
        if not isinstance(proposals, list):
            proposals = []
        aid_text = _normal(str(fact.get("claim") or "") + " " + " ".join(map(str, proposals)))
        russia = any(pattern.search(source + " " + aid_text) for _, pattern in russia_terms)
        for sid in scale_ids:
            if sid.startswith("russia-") and not russia:
                continue
            source_hits = [term for term, pattern in compiled[sid] if pattern.search(source)]
            aid_hits = [term for term, pattern in compiled[sid] if pattern.search(aid_text)]
            if not source_hits and not aid_hits:
                continue
            index[fid]["matched_scale_ids"].append(sid)
            matched_facts.add(fid)
            candidates[(aid, sid)].append({"fact_id": fid,
                "source_search_cues": source_hits, "machine_aid_search_cues": aid_hits,
                "retrieval_rank": (2 * min(len(source_hits), 4) + min(len(aid_hits), 4)),
                "semantic_review_status": "not_reviewed"})

    cells, actor_rows, selected_ids = [], [], set()
    for aid, actor in actor_map.items():
        actor_selected = set()
        for scale in scales:
            sid = scale["id"]
            ranked = sorted(candidates[(aid, sid)], key=lambda r: (-r["retrieval_rank"], r["fact_id"]))
            groups = defaultdict(list)
            for item in ranked:
                groups[_literal_key(index[item["fact_id"]]["fact"])].append(item)
            # Highest-ranked literal groups first, with original IDs as stable
            # tie-breakers. Every member and every nonselected match survives.
            short = []
            for group in sorted(groups.values(), key=lambda g: (-g[0]["retrieval_rank"], g[0]["fact_id"]))[:max_per_cell]:
                representative = deepcopy(group[0])
                representative.update(literal_member_fact_ids=[r["fact_id"] for r in group],
                                      episode_identity_verified=False,
                                      source_independence_verified=False)
                short.append(representative)
                selected_ids.add(representative["fact_id"])
                actor_selected.add(representative["fact_id"])
            cells.append({"actor_id": aid, "scale_id": sid, "candidate_fact_count": len(ranked),
                          "literal_review_group_count": len(groups), "candidates": ranked,
                          "shortlist": short, "value": None, "measurement_status": "not_scored"})
        rescue = [fid for fid in eligible[aid] if fid not in matched_facts]
        actor_rows.append({"actor_id": aid, "name": actor["name"],
                           "eligible_fact_ids": eligible[aid], "unmatched_rescue_fact_ids": rescue,
                           "shortlist_fact_ids": sorted(actor_selected),
                           "all_facts_semantically_reviewed": False})
    summary = {"version": POLICY_VERSION, "api_calls": 0, "scores_generated": 0,
               "retrieval_window": {"date_from": config.get("date_from"), "date_to": config.get("date_to")},
               "counts": {"input_facts_indexed": len(index), "eligible_facts_indexed": sum(map(len, eligible.values())),
                          "ineligible_facts_preserved": sum(exclusions.values()),
                          "eligible_facts_matching_at_least_one_query": len(matched_facts),
                          "eligible_facts_in_rescue_queue": sum(map(len, eligible.values())) - len(matched_facts),
                          "distinct_shortlisted_facts": len(selected_ids), "actor_scale_cells": len(cells)},
               "exclusion_counts": dict(sorted(exclusions.items())),
               "actors": [{"actor_id": a["actor_id"], "name": a["name"],
                           "eligible_facts": len(a["eligible_fact_ids"]),
                           "unmatched_rescue_facts": len(a["unmatched_rescue_fact_ids"]),
                           "shortlisted_facts": len(a["shortlist_fact_ids"])} for a in actor_rows],
               "scales": [{"scale_id": s["id"], "name": s["name"],
                           "actors_with_candidates": sum(bool(c["candidate_fact_count"]) for c in cells if c["scale_id"] == s["id"]),
                           "candidate_facts": sum(c["candidate_fact_count"] for c in cells if c["scale_id"] == s["id"])} for s in scales],
               "limitations": ["Lexical candidates are neither measured values nor evidence of a position.",
                               "Positive and negative Russia queries intentionally retrieve the same evidence.",
                               "A thematic cue in attribution or source context does not establish the actor's own position.",
                               "Machine claims and trait proposals are search aids requiring original-source validation.",
                               "Unknown event times remain review candidates, not proven in-window episodes.",
                               "All unmatched eligible facts remain available for semantic rescue review.",
                               "Literal review groups are not independent episodes."]}
    return {"version": POLICY_VERSION, "source_facts_sha256": digest(facts),
            "dictionary_sha256": digest(dictionary), "private_fact_index": index,
            "private_actor_dossiers": actor_rows, "private_cells": cells,
            "public_summary": summary}
