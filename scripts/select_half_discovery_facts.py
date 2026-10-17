"""Offline actor-balanced discovery selection; never deletes evidence or calls API.

Literal same-actor quote duplicates are alternative discovery representatives,
not independent events. A global document capacity is never silently relaxed.
An augmenting-path assignment can change the representative of an identical
quote to another source document instead of needlessly producing a shortfall.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import sys
import unicodedata
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.corpus_pipeline.models import _eligible_fact

SEED = "rwanda-half-discovery-2026-10-17-v1"
CORPUS_ID = "corpus-b2d1452fae0acf9c20f5"
POLICY = "actor-round-robin-literal-diversity-document-cap-v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _topic(value):
    return " ".join(unicodedata.normalize("NFC", value).casefold().split()) if isinstance(value, str) else ""


def _source(fact):
    try:
        parsed = urlsplit(str(fact.get("source_url") or ""))
        return parsed.hostname.casefold() if parsed.hostname else ""
    except ValueError:
        return ""


def select_facts(snapshot, target=600, seed=SEED, max_per_document=2):
    """Return (selected fact-ID array, private selection audit), deterministically.

    Actors with fewer eligible distinct literal quotes go first in each round.
    Within an actor, prefer novelty of document, source hostname and literal
    topic, then lower reuse counts and SHA256(seed, fact_id). This is a diversity
    heuristic, not a semantic clustering or measure of intensity. If a document
    is full, capacity-preserving representative swaps are allowed only within
    the exact same actor/quote unit. No accepted unit is discarded by a swap.
    """
    if type(target) is not int or target < 1 or type(max_per_document) is not int or not 1 <= max_per_document <= 2:
        raise ValueError("Require a positive target and an explicit document cap of 1 or 2")
    if not isinstance(seed, str) or not seed:
        raise ValueError("A fixed nonempty seed is required")
    config = snapshot["corpus"]["config"]
    objects = config["objects"]
    actor_order = [obj["id"] for obj in objects]
    if len(actor_order) != len(set(actor_order)):
        raise ValueError("Approved actor IDs must be unique")
    roster_position = {actor: i for i, actor in enumerate(actor_order)}
    facts, index, units = {}, {}, {}
    eligible_counts, exclusions = Counter(), Counter()
    repeated_input_rows = 0
    for fact in snapshot["facts"]:
        identifier = fact.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Every source fact needs a nonempty ID")
        if identifier in facts:
            if facts[identifier] != fact:
                raise ValueError("Conflicting repeated source fact ID")
            repeated_input_rows += 1
            continue
        facts[identifier] = fact
        actor, document = fact.get("actor_id"), fact.get("document_id")
        row = {"fact_id": identifier, "actor_id": actor, "document_id": document}
        index[identifier] = row
        if actor not in roster_position:
            reason = "unapproved_actor"
        elif not _eligible_fact(fact, config):
            reason = "measurement_ineligible"
        elif not isinstance(document, str) or not document or not isinstance(fact.get("quote"), str) or not fact["quote"].strip():
            reason = "missing_document_or_literal_quote"
        else:
            eligible_counts[actor] += 1
            unit = _digest([actor, fact["quote"]])  # Exact quote, without case/whitespace/negation changes.
            units.setdefault(unit, {"actor_id": actor, "fact_ids": []})["fact_ids"].append(identifier)
            row.update(status="eligible", reason="not_selected", literal_quote_unit=unit)
            continue
        row.update(status="excluded", reason=reason)
        exclusions[reason] += 1
    ranks = {identifier: _digest([seed, identifier]) for identifier in facts}
    features = {identifier: (fact.get("document_id"), _source(fact), _topic(fact.get("topic")))
                for identifier, fact in facts.items()}
    for unit in units.values():
        unit["fact_ids"].sort(key=lambda identifier: (ranks[identifier], identifier))
    actor_units = defaultdict(set)
    for identifier, unit in units.items():
        actor_units[unit["actor_id"]].add(identifier)
    actor_unique_counts = {actor: len(actor_units[actor]) for actor in actor_order}
    ordered_actors = sorted((actor for actor in actor_order if actor_units[actor]),
                            key=lambda actor: (actor_unique_counts[actor], roster_position[actor]))
    remaining = {actor: set(actor_units[actor]) for actor in ordered_actors}
    assignment, document_units = {}, defaultdict(set)
    selected_order, selected_rounds, swaps = [], {}, []

    def profiles():
        result = defaultdict(lambda: {"document": Counter(), "source": Counter(), "topic": Counter()})
        for identifier in assignment.values():
            fact = facts[identifier]
            row = result[fact["actor_id"]]
            document, source, topic = features[identifier]
            row["document"][document] += 1
            if source:
                row["source"][source] += 1
            if topic:
                row["topic"][topic] += 1
        return result

    def fact_key(identifier, counts):
        fact = facts[identifier]
        actor = counts[fact["actor_id"]]
        document, source, topic = features[identifier]
        dc, sc, tc = actor["document"][document], actor["source"][source], actor["topic"][topic]
        novelty = int(dc == 0) + int(bool(source) and sc == 0) + int(bool(topic) and tc == 0)
        return -novelty, dc, sc, tc, ranks[identifier], identifier

    def augment(start, counts):
        """Find one augmenting path in quote-unit ↔ source-document capacity."""
        queue, visited, document_parent = deque([start]), {start}, {}
        while queue:
            unit_id = queue.popleft()
            for fact_id in sorted(units[unit_id]["fact_ids"], key=lambda fid: fact_key(fid, counts)):
                document = facts[fact_id]["document_id"]
                if document in document_parent:
                    continue
                document_parent[document] = (unit_id, fact_id)
                if len(document_units[document]) < max_per_document:
                    # Walk backwards; every move releases the next required slot.
                    while True:
                        moving, replacement = document_parent[document]
                        previous = assignment.get(moving)
                        assignment[moving] = replacement
                        document_units[document].add(moving)
                        if previous is None:
                            return True
                        old_document = facts[previous]["document_id"]
                        document_units[old_document].remove(moving)
                        if previous != replacement:
                            swaps.append({"previous_fact_id": previous, "selected_fact_id": replacement,
                                          "reason": "same_literal_quote_representative_swap_for_document_capacity"})
                        document = old_document
                for occupied in sorted(document_units[document], key=lambda uid: (ranks[assignment[uid]], uid)):
                    if occupied not in visited:
                        visited.add(occupied)
                        queue.append(occupied)
        return False

    round_number = 0
    while len(assignment) < target:
        round_number += 1
        progress = False
        for actor in ordered_actors:
            if not remaining[actor]:
                continue
            counts = profiles()
            options = sorted(remaining[actor], key=lambda uid: min(fact_key(fid, counts) for fid in units[uid]["fact_ids"]))
            for unit_id in options:
                remaining[actor].remove(unit_id)
                if augment(unit_id, counts):
                    selected_order.append(unit_id)
                    selected_rounds[unit_id] = round_number
                    progress = True
                    break
                # Insertion of more selected units cannot create free capacity
                # for a unit with no augmenting path; mark this exhausted unit.
                for identifier in units[unit_id]["fact_ids"]:
                    index[identifier].update(reason="global_document_capacity_blocked")
            if len(assignment) == target:
                break
        if not progress:
            break
    selected_ids = [assignment[unit] for unit in selected_order]
    selected_by_actor = Counter(facts[identifier]["actor_id"] for identifier in selected_ids)
    for unit_id, unit in units.items():
        selected = assignment.get(unit_id)
        for identifier in unit["fact_ids"]:
            if identifier == selected:
                index[identifier].update(status="selected", reason="actor_round_robin_diversity",
                                         selected_round=selected_rounds[unit_id])
            elif selected is not None:
                index[identifier].update(status="excluded_from_discovery_only", reason="literal_same_actor_quote_duplicate",
                                         representative_fact_id=selected)
            elif index[identifier]["reason"] != "global_document_capacity_blocked":
                index[identifier].update(reason="target_reached_before_candidate")
    actual_doc_counts = Counter(facts[identifier]["document_id"] for identifier in selected_ids)
    actor_coverage = []
    for obj in objects:
        chosen = [facts[identifier] for identifier in selected_ids if facts[identifier]["actor_id"] == obj["id"]]
        actor_coverage.append({"actor_id": obj["id"], "name": obj["name"], "eligible_facts": eligible_counts[obj["id"]],
            "eligible_literal_quote_units": actor_unique_counts[obj["id"]], "selected_facts": selected_by_actor[obj["id"]],
            "selected_documents": len({fact["document_id"] for fact in chosen}),
            "selected_source_hosts": len({value for fact in chosen if (value := _source(fact))}),
            "selected_literal_topics": len({value for fact in chosen if (value := _topic(fact.get("topic")))})})
    counts = {"input_facts": len(facts), "repeated_input_rows": repeated_input_rows,
        "eligible_facts": sum(eligible_counts.values()), "eligible_literal_quote_units": len(units),
        "literal_duplicate_records": sum(len(unit["fact_ids"]) - 1 for unit in units.values()),
        "selected_facts": len(selected_ids), "target": target, "shortfall": max(0, target - len(selected_ids)),
        "actors_with_eligible_facts": len(ordered_actors), "actors_selected": len(selected_by_actor),
        "selected_documents": len(actual_doc_counts), "actual_max_facts_per_document": max(actual_doc_counts.values(), default=0),
        "configured_max_facts_per_document": max_per_document, "representative_swaps": len(swaps),
        "excluded_by_rule": dict(sorted(exclusions.items()))}
    audit = {"version": POLICY, "seed": seed, "aspect": config.get("aspect", "public_positions"),
        "counts": counts, "target_reached": len(selected_ids) == target, "capacity_relaxed": False,
        "capacity_maximum_reached_when_short": len(selected_ids) < target,
        "shortfall_reasons": ([] if len(selected_ids) == target else
            (["eligible_literal_quote_units_below_target"] if len(units) < target else [])
            + (["global_document_capacity"] if any(row["reason"] == "global_document_capacity_blocked" for row in index.values()) else [])),
        "facts_snapshot_sha256": _digest(sorted(snapshot["facts"], key=lambda fact: fact["id"])),
        "actor_round_order": ordered_actors, "actor_coverage": actor_coverage,
        "selection_index": [index[identifier] for identifier in sorted(index)], "representative_swaps": swaps,
        "api_calls": 0, "source_records_changed": False, "queue_changed": False,
        "limitations": ["Discovery selection is not an intensity score or a population frequency estimate.",
            "Literal quote duplicates are alternate representatives only; no independent-event equivalence is claimed.",
            "Source-host and original topic-label diversity are lexical heuristics, not verified source independence or semantic clusters.",
            "No Russia term, earlier scale, loading, factor or cluster result participates in ranking.",
            "A shortfall is reported without changing the document cap or borrowing ineligible evidence."]}
    return selected_ids, audit


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target", type=int, default=600)
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--max-per-document", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    snapshot = args.snapshot or args.root / ".corpus_runtime" / "pilots" / CORPUS_ID / "half-70-v1" / "results_private.json"
    output = args.output_dir or snapshot.parent / "describe_selection"
    raw = snapshot.read_bytes()
    selected, audit = select_facts(json.loads(raw.decode("utf-8-sig")), args.target, args.seed, args.max_per_document)
    audit["private_snapshot_sha256"] = hashlib.sha256(raw).hexdigest()
    _write(output / "selected_fact_ids.json", selected)
    _write(output / "selection_audit_private.json", audit)
    print(json.dumps({"counts": audit["counts"], "target_reached": audit["target_reached"],
                      "capacity_relaxed": False, "api_calls": 0}, ensure_ascii=False, allow_nan=False))
    return selected, audit


if __name__ == "__main__":
    main()
