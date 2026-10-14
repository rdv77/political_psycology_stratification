from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class Country(BaseModel):
    name: str


class Party(BaseModel):
    name_original: str
    country: str
    ideology: Optional[str] = None
    leaders: List[str] = Field(default_factory=list)
    summary: Optional[str] = None


class TraitList(BaseModel):
    traits: List[str]


class ScoredParty(BaseModel):
    party_name: str
    country: str
    scores: Dict[str, float]  # trait -> 0-10


class FactorResults(BaseModel):
    traits: List[str]
    loadings: List[List[float]]  # shape: traits x factors
    explained_variance: List[float]
    factor_scores: Dict[str, List[float]]  # party_name -> factor scores vector


class ClusterResults(BaseModel):
    party_to_cluster: Dict[str, int]
    party_to_center_distance: Dict[str, float]
    k: int


