"""Exploratory text structure of saved extraction claims; no model API.

This is latent semantic analysis (LSA), not factor analysis of the approved
intensity scales. Zero word mass means no occurrence in the selected material,
not a politician's rejection of a goal. Missing source material stays unknown.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import re
import unicodedata

import numpy as np
import scipy
import sklearn
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfTransformer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

from .scoring_preparation import digest, filter_eligible_fact


VERSION = "rwanda-saved-claims-lsa-v1"
SEED = 20261018
# These remove reporting formulae and role/identity terms, not political goals.
# Tokens are fixed before looking at components. Country names other than Rwanda
# remain because they may be substantively relevant international topics.
BOILERPLATE = frozenset("""
rwanda rwandan rwandans urwanda ruanda ruandais rwandaise republic country
president presidency presidential minister ministers ministry prime senator
senate ambassador governor mayor secretary chairman chairperson director chief
officer excellency honorable honourable hon mr mrs ms dr prof professor
deputy speaker executive official officials authority authorities
general chair chairs councillor commissioner
said says say stated states statement statements spoke speech speaking
argued argues emphasized emphasised highlighted highlights underscored noted
announced announcements reiterated reaffirmed expressed expressing welcomed
called calls calling urged urges discussed discussion discussions meeting met
meetings held attended attendance participated participation received receiving
congratulated thanked greetings gratitude visited visit visits
declared declares remarked remarks message interview addressed addressing
described describes reported reports report according told tells
the rt right honour h e
support supports supportive supporting pro anti stance position positions
la le les de du des et ou au aux en dans sur pour par un une ce ces cette cet
il elle ils elles est sont avec sans son sa ses leur leurs que qui nous vous
""".split())

# A transparent descriptive overlay for reading text components. These are not
# the approved V2 scales; overlap is lexical and is never a property score.
FACETS = {
    "international_cooperation": ("diplomacy", "diplomatic", "cooperation", "partnership", "partnerships", "regional", "international", "multilateral", "integration"),
    "security_conflict": ("security", "military", "defence", "defense", "peace", "conflict", "conflicts", "war", "armed", "ceasefire", "mediation", "negotiations"),
    "rights_legal_process": ("rights", "freedom", "freedoms", "justice", "judicial", "court", "courts", "legal", "constitutional", "law", "arbitration", "detention"),
    "economy_private_activity": ("economic", "economy", "business", "businesses", "investment", "investments", "private", "capital", "entrepreneurship", "entrepreneurs", "trade", "finance", "financing"),
    "public_services_health": ("health", "healthcare", "medical", "hospital", "hospitals", "disease", "diseases", "epidemic", "education", "school", "schools", "services", "refugees"),
    "technology_research": ("technology", "technologies", "digital", "innovation", "innovations", "ai", "artificial intelligence", "research", "science", "data", "nuclear"),
    "unity_memory": ("unity", "reconciliation", "genocide", "memory", "remembrance", "commemoration", "ethnic", "division", "divisions", "cohesion"),
    "ethics_accountability": ("ethics", "ethical", "integrity", "accountability", "corruption", "impunity", "responsibility", "discipline", "professionalism"),
    "environment_climate": ("climate", "environment", "environmental", "ecological", "pollution", "adaptation", "biodiversity", "conservation", "emissions"),
    "russia_mentions": ("russia", "russian", "moscow", "rosatom", "lavrov"),
}


def _plain(value):
    return " ".join(unicodedata.normalize("NFC", str(value or "")).split())


def identity_terms(roster, aliases=None):
    """Remove all approved names/components from features, preserving labels."""
    terms = set()
    for actor in roster:
        for key in ("name", "name_original", "name_ru", "canonical_latin"):
            terms.update(re.findall(r"[^\W\d_]{2,}", _plain(actor.get(key)).casefold(), re.UNICODE))
        terms.update(re.findall(r"[^\W\d_]{2,}", actor["id"].replace("-", " ").casefold(), re.UNICODE))
    for actor in (aliases or {}).get("actors", []):
        for row in actor.get("aliases", []):
            text = row.get("text", "") if isinstance(row, dict) else row
            terms.update(re.findall(r"[^\W\d_]{2,}", _plain(text).casefold(), re.UNICODE))
    return terms


def prepare_units(facts, roster, *, aliases=None, config=None, include_trait_candidates=True):
    """Collapse actor/quote/time duplicates, not alleged independent events.

    Unknown time is kept apart from a known time; publication time never fills
    an event date. A repeated literal quotation at distinct known times remains
    separate. Source IDs are retained only in the private audit.
    """
    cfg = config or {"aspect": "public_positions"}
    actors = {actor["id"] for actor in roster}
    groups, rejected = defaultdict(list), Counter()
    for fact in facts:
        if fact.get("actor_id") not in actors:
            rejected["outside_approved_roster"] += 1
            continue
        if not filter_eligible_fact(fact, cfg):
            rejected["not_eligible_own_position_candidate"] += 1
            continue
        quote = _plain(fact.get("quote"))
        # A missing source quotation is not a deduplication key shared by all
        # unquoted claims. Source-backed extraction normally guarantees this.
        literal = quote if quote else "missing-quote-" + digest(fact)
        key = (fact["actor_id"], literal, fact.get("event_date"),
               fact.get("event_date_from"), fact.get("event_date_to"))
        groups[key].append(fact)
    removed_names = identity_terms(roster, aliases)
    stops = frozenset(ENGLISH_STOP_WORDS) | BOILERPLATE | removed_names
    units = []
    for key, members in sorted(groups.items(), key=lambda item: str(item[0])):
        # One representation per literal group; paraphrases or extra labels in
        # reposts cannot increase that group's weight or vocabulary.
        representative = min(members, key=lambda f: (str(f.get("id", "")), digest(f)))
        labels = [x for x in representative.get("trait_candidates", []) if isinstance(x, str)]
        raw = " ".join([_plain(representative.get("claim"))] + (labels if include_trait_candidates else []))
        tokens = [t for t in re.findall(r"[^\W\d_]{2,}", raw.casefold().replace("_", " "), re.UNICODE)
                  if t not in stops]
        text = " ".join(tokens)
        units.append({"actor_id": key[0], "text": text,
                      "literal_key_sha256": digest(key), "member_fact_ids": [f.get("id") for f in members],
                      "member_count": len(members),
                      "event_date": key[2], "event_date_from": key[3], "event_date_to": key[4],
                      "has_known_event_time": bool(key[2] or key[3] and key[4]),
                      "has_general_risk_flag": any(f.get("correction_flags") for f in members),
                      "source_document_count": len({f.get("document_id") for f in members if f.get("document_id")}),
                      "is_verified_independent_event": False})
    return {"units": units, "rejected": dict(sorted(rejected.items())),
            "removed_identity_terms": sorted(removed_names),
            "eligible_fact_count": sum(u["member_count"] for u in units),
            "literal_group_count": len(units),
            "literal_repeat_excess": sum(u["member_count"] - 1 for u in units)}


def _actor_matrix(units, roster, *, minimum_actor_df=2, maximum_actor_df=.90, maximum_features=12000):
    nonempty = [u for u in units if u["text"]]
    active_ids = {u["actor_id"] for u in nonempty}
    actors = [a for a in roster if a["id"] in active_ids]
    if len(actors) < 3:
        raise ValueError("Text structure requires at least three actors with textual material")
    vectorizer = CountVectorizer(lowercase=False, ngram_range=(1, 2), token_pattern=r"(?u)\b[^\W\d_]{2,}\b")
    counts = vectorizer.fit_transform(u["text"] for u in nonempty)
    terms = vectorizer.get_feature_names_out()
    counts = normalize(counts, norm="l1", axis=1)
    rows = {a["id"]: index for index, a in enumerate(actors)}
    n_units = Counter(u["actor_id"] for u in nonempty)
    aggregate = sparse.csr_matrix((np.asarray([1 / n_units[u["actor_id"]] for u in nonempty]),
                                   ([rows[u["actor_id"]] for u in nonempty], np.arange(len(nonempty)))),
                                  shape=(len(actors), len(nonempty)))
    average_terms = aggregate @ counts
    actor_df = np.asarray((average_terms > 0).sum(axis=0)).ravel()
    keep = np.flatnonzero((actor_df >= minimum_actor_df) & (actor_df <= maximum_actor_df * len(actors)))
    if len(keep) > maximum_features:
        mass = np.asarray(average_terms.sum(axis=0)).ravel()
        keep = np.asarray(sorted(keep, key=lambda k: (-mass[k], terms[k]))[:maximum_features])
    keep = np.asarray(sorted(keep, key=lambda k: terms[k]))
    if len(keep) < 2:
        raise ValueError("Insufficient shared text vocabulary after identity and boilerplate removal")
    # IDF is learned across actor profiles, not across thousands of Kagame
    # articles. Each actor is one row with equal L2 norm in the model.
    matrix = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False).fit_transform(average_terms[:, keep])
    nonzero_rows = np.asarray(matrix.getnnz(axis=1)).ravel() > 0
    empty_ids = [a["id"] for a, good in zip(actors, nonzero_rows) if not good]
    actors = [a for a, good in zip(actors, nonzero_rows) if good]
    matrix = matrix[nonzero_rows]
    return {"matrix": matrix, "actors": actors, "terms": terms[keep], "actor_df": actor_df[keep],
            "nonempty_units": nonempty, "empty_after_shared_vocabulary_actor_ids": empty_ids,
            "preselection_vocabulary_size": len(terms), "minimum_actor_df": minimum_actor_df,
            "maximum_actor_df": maximum_actor_df}


def _top_terms(weights, terms, actor_df, *, positive=True, limit=12):
    indexes = np.flatnonzero(weights > 1e-12 if positive else weights < -1e-12)
    indexes = sorted(indexes, key=lambda j: (-abs(weights[j]), terms[j]))[:limit]
    return [{"term": str(terms[j]), "weight": float(weights[j]), "actors_with_term": int(actor_df[j])} for j in indexes]


def _facets(weights, terms):
    squared_total = float(np.square(weights).sum())
    result = []
    for facet, vocabulary in FACETS.items():
        indexes = [j for j, term in enumerate(terms) if term in vocabulary]
        mass = float(np.square(weights[indexes]).sum()) if indexes else 0
        if mass:
            result.append({"facet": facet, "lexical_squared_weight_share": mass / squared_total,
                           "matched_terms": [str(terms[j]) for j in indexes if abs(weights[j]) >= 1e-12]})
    return sorted(result, key=lambda r: (-r["lexical_squared_weight_share"], r["facet"]))


def fit_structure(units, roster, *, name="all_eligible_candidates", excluded_actor_ids=(), max_q=4):
    """Fit one max-q uncentered SVD, comparing nested q=2/3/4 prefixes.

    The centered variance share is var(projected actor scores) divided by the
    total centered TF-IDF profile variance. Uncentered reconstruction energy is
    reported separately; neither denominator is variance of V2 rating scales.
    """
    excluded = set(excluded_actor_ids)
    model_roster = [a for a in roster if a["id"] not in excluded]
    model_units = [u for u in units if u["actor_id"] not in excluded]
    built = _actor_matrix(model_units, model_roster)
    matrix, actors, terms = built["matrix"], built["actors"], built["terms"]
    qmax = min(max_q, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if qmax < 2:
        raise ValueError("Insufficient actor/text dimensions for a two-component comparison")
    estimator = TruncatedSVD(n_components=qmax, algorithm="randomized", n_iter=10, random_state=SEED)
    scores = estimator.fit_transform(matrix)
    components = estimator.components_.copy()
    for k in range(qmax):
        pivot = int(np.argmax(np.abs(components[k])))
        if components[k, pivot] < 0:
            components[k] *= -1
            scores[:, k] *= -1
    total_variance = float(np.asarray(matrix.multiply(matrix).mean(axis=0)).sum()
                           - np.square(np.asarray(matrix.mean(axis=0))).sum())
    total_energy = float(matrix.multiply(matrix).sum())
    explained = np.var(scores, axis=0) / total_variance
    energy = np.square(scores).sum(axis=0) / total_energy
    details = []
    for k in range(qmax):
        order = np.argsort(scores[:, k])
        details.append({"component": k + 1, "centered_text_variance_share": float(explained[k]),
                        "uncentered_text_energy_share": float(energy[k]),
                        "positive_terms": _top_terms(components[k], terms, built["actor_df"]),
                        "negative_terms": _top_terms(components[k], terms, built["actor_df"], positive=False),
                        "lexical_facets": _facets(components[k], terms),
                        "low_coordinate_actors": [{"actor_id": actors[i]["id"], "name": actors[i].get("name", actors[i]["id"]),
                                                   "coordinate": float(scores[i, k])} for i in order[:5]],
                        "high_coordinate_actors": [{"actor_id": actors[i]["id"], "name": actors[i].get("name", actors[i]["id"]),
                                                    "coordinate": float(scores[i, k])} for i in order[-5:][::-1]]})
    by_actor = {a["id"]: index for index, a in enumerate(actors)}
    unit_counts, fact_counts, risk_counts, time_counts = Counter(), Counter(), Counter(), Counter()
    for unit in model_units:
        unit_counts[unit["actor_id"]] += 1
        fact_counts[unit["actor_id"]] += unit["member_count"]
        risk_counts[unit["actor_id"]] += unit["has_general_risk_flag"]
        time_counts[unit["actor_id"]] += unit["has_known_event_time"]
    cards = []
    for actor in roster:
        aid = actor["id"]
        index = by_actor.get(aid)
        status = "projected_text_profile" if index is not None else "excluded_in_sensitivity" if aid in excluded else "no_eligible_text_material" if not unit_counts[aid] else "no_shared_vocabulary"
        cards.append({"actor_id": aid, "name": actor.get("name", aid), "name_original": actor.get("name_original"),
                      "status": status, "text_coordinates": [float(x) for x in scores[index]] if index is not None else None,
                      "literal_group_count": unit_counts[aid], "candidate_fact_count": fact_counts[aid],
                      "risk_flagged_literal_groups": risk_counts[aid], "known_event_time_literal_groups": time_counts[aid],
                      "is_v2_intensity_score": False})
    public = {"name": name, "method": "actor_balanced_tfidf_truncated_svd_lsa",
              "actor_count": len(actors), "roster_actor_count": len(roster), "feature_count": len(terms),
              "eligible_fact_count": sum(u["member_count"] for u in model_units), "literal_group_count": len(model_units),
              "excluded_actor_ids": sorted(excluded),
              "minimum_actor_term_document_frequency": built["minimum_actor_df"],
              "maximum_actor_term_document_frequency_fraction": built["maximum_actor_df"],
              "preselection_vocabulary_size": built["preselection_vocabulary_size"],
              "components": details, "actors": cards,
              "solutions": [{"q": q, "centered_text_variance_share": float(explained[:q].sum()),
                             "uncentered_text_energy_share": float(energy[:q].sum()),
                             "component_prefix_of_shared_max_q_fit": True} for q in (2, 3, 4) if q <= qmax],
              "total_centered_tfidf_profile_variance": total_variance,
              "raw_claims_published": False, "raw_quotes_published": False,
              "factor_analysis_of_approved_scales": False, "russia_factor_test_performed": False}
    return {"public": public, "private": {"terms": terms.tolist(), "components": components.tolist(),
                                           "actor_ids": [a["id"] for a in actors], "scores": scores.tolist(),
                                           "actor_term_matrix": matrix.toarray().tolist()}}


def compare_structures(baseline, sensitivity):
    """Match axes on shared terms and compare distances on shared actor rows."""
    left, right = baseline["private"], sensitivity["private"]
    terms = sorted(set(left["terms"]) & set(right["terms"]))
    li, ri = {t: i for i, t in enumerate(left["terms"])}, {t: i for i, t in enumerate(right["terms"])}
    a, b = np.asarray(left["components"])[:, [li[t] for t in terms]], np.asarray(right["components"])[:, [ri[t] for t in terms]]
    a, b = normalize(a), normalize(b)
    similarities = a @ b.T
    row_ids, col_ids = linear_sum_assignment(-np.abs(similarities))
    common_actors = sorted(set(left["actor_ids"]) & set(right["actor_ids"]))
    lrows, rrows = {x: i for i, x in enumerate(left["actor_ids"])}, {x: i for i, x in enumerate(right["actor_ids"])}
    ls, rs = np.asarray(left["scores"])[[lrows[x] for x in common_actors]], np.asarray(right["scores"])[[rrows[x] for x in common_actors]]
    distances = []
    for q in (2, 3, 4):
        if q > ls.shape[1] or q > rs.shape[1]:
            continue
        dleft, dright = pairwise_distances(ls[:, :q]), pairwise_distances(rs[:, :q])
        triangle = np.triu_indices(len(common_actors), 1)
        statistic = spearmanr(dleft[triangle], dright[triangle]).statistic
        distances.append({"q": q, "shared_actor_pairwise_distance_spearman": float(statistic) if np.isfinite(statistic) else None,
                          "descriptive_not_independent_sample_p_value": True})
    return {"sensitivity": sensitivity["public"]["name"], "shared_feature_count": len(terms),
            "shared_actor_count": len(common_actors),
            "matched_text_components": [{"baseline_component": int(i + 1), "sensitivity_component": int(j + 1),
                                          "absolute_shared_vocabulary_cosine": float(abs(similarities[i, j])),
                                          "orientation_sign": 1 if similarities[i, j] >= 0 else -1} for i, j in zip(row_ids, col_ids)],
            "pairwise_distance_comparisons": distances,
            "caution": "Vocabulary and actor-level IDF are refit; component signs and order are arbitrary. Pair distances are dependent descriptive comparisons."}


def analyze(facts, roster, *, aliases=None, config=None):
    prepared = prepare_units(facts, roster, aliases=aliases, config=config)
    baseline = fit_structure(prepared["units"], roster)
    scenario_specs = [("without_kagame", ("paul-kagame",)),
                      ("without_nduhungirehe", ("olivier-nduhungirehe",)),
                      ("without_both_high_volume_actors", ("paul-kagame", "olivier-nduhungirehe"))]
    scenarios = [fit_structure(prepared["units"], roster, name=name, excluded_actor_ids=ids) for name, ids in scenario_specs]
    sparse_ids = tuple(a["id"] for a in roster if sum(u["actor_id"] == a["id"] for u in prepared["units"]) < 8)
    scenarios.append(fit_structure(prepared["units"], roster, name="at_least_8_literal_groups_per_actor", excluded_actor_ids=sparse_ids))
    claims_only = prepare_units(facts, roster, aliases=aliases, config=config, include_trait_candidates=False)
    scenarios.append(fit_structure(claims_only["units"], roster, name="saved_claims_without_trait_labels"))
    public = {"version": VERSION, "api_calls": 0, "new_api_cost_usd": 0,
              "method": "Exploratory LSA of saved English extraction claims and trait labels",
              "source_facts_content_sha256": digest(facts),
              "input_counts": {"all_extracted_facts": len(facts), "eligible_fact_count": prepared["eligible_fact_count"],
                               "literal_group_count": prepared["literal_group_count"], "literal_repeat_excess": prepared["literal_repeat_excess"],
                               "roster_actors": len(roster), "actors_with_text_profiles": baseline["public"]["actor_count"]},
              "rejected_counts": prepared["rejected"], "baseline": baseline["public"],
              "reproducibility": {"random_seed": SEED, "svd_algorithm": "randomized", "svd_power_iterations": 10,
                                  "maximum_text_components": 4, "feature_ngrams": [1, 2],
                                  "maximum_features": 12000, "numpy_version": np.__version__,
                                  "scipy_version": scipy.__version__, "scikit_learn_version": sklearn.__version__,
                                  "identity_and_boilerplate_policy": "fixed code lists; all approved alias name components removed",
                                  "stopwords_content_sha256": digest(sorted(ENGLISH_STOP_WORDS | BOILERPLATE)),
                                  "facet_lexicon_content_sha256": digest(FACETS)},
              "sensitivities": [s["public"] for s in scenarios],
              "comparisons": [compare_structures(baseline, s) for s in scenarios],
              "measurement_contract": {"approved_v2_dictionary_modified": False,
                                       "approved_v2_ratings_generated": False, "approved_v2_factor_admission_changed": False,
                                       "missing_person_material_replaced_by_zero": False,
                                       "word_absence_interpreted_as_rejected_position": False,
                                       "literal_groups_claimed_independent_events": False,
                                       "publication_date_used_as_event_date": False,
                                       "actor_level_idf": True, "unit_l1_before_actor_averaging": True,
                                       "actor_rows_l2_normalized_before_svd": True,
                                       "trait_labels_in_primary_text_representation": True,
                                       "separate_claims_only_sensitivity": True,
                                       "source_roles_cannot_be_separated_from_topics": True},
              "variance_definition": "Centered variance of projected actor TF-IDF profiles divided by total centered actor TF-IDF variance; not variance of V2 property intensities. Uncentered reconstruction energy is separate.",
              "limitations": ["This is a text-theme map, not factor analysis of the approved 38 scales or measurement of political beliefs.",
                              "Most extraction candidates have only programmatic checks, not full semantic review; their original extraction errors may affect this exploratory map.",
                              "Claims are saved model paraphrases, not independent translations or ground truth; the multilingual source quotations were not retranslated.",
                              "Unknown event times remain included as text material published in the corpus, not verified positions inside the five-year research window.",
                              "Literal repetitions are removed but paraphrases, translations and multiple passages of an event may still repeat its topic.",
                              "Words, candidates and reporting emphasis measure textual content, not the strength or frequency-independent intensity of an approved property.",
                              "Equal actor row norms reduce article-volume dominance but cannot repair selection bias, one-fragment profiles, institution/role specialization or missing material.",
                              "The first uncentered SVD component may represent shared vocabulary rather than an opposing political axis.",
                              "q=2/3/4 are descriptive nested text summaries; explained variance does not select a political factor solution.",
                              "Lexical facet overlays are transparent keyword aids, may overlap and are not validated factors or attitude scores.",
                              "No Russia attitude factor or 2-percent criterion of the approved V2 protocol is established by a Russia-related word component."],
              "raw_claims_published": False, "raw_quotes_published": False, "raw_source_ids_published": False}
    private = {"prepared": prepared, "baseline": baseline["private"],
               "sensitivities": {s["public"]["name"]: s["private"] for s in scenarios}}
    return {"public": public, "private": private}
