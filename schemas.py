"""
Database Schemas for the Strategy Prototype

Each Pydantic model corresponds to a MongoDB collection.
Class name lowercased is used as collection name (e.g., Game -> "game").
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Tuple

# --- Core Map/World ---
Terrain = Literal["plains", "forest", "fields", "mountain", "water", "bridge", "wall", "relic"]

class MapTile(BaseModel):
    x: int
    y: int
    terrain: Terrain = "plains"
    structure: Optional[Literal["bridge", "wall"]] = None
    owner: Optional[str] = None  # faction id string

class GameMap(BaseModel):
    width: int
    height: int
    tiles: List[MapTile]

# --- Factions & Units ---
UnitType = Literal["worker", "infantry", "ruler", "catapult"]

class Unit(BaseModel):
    id: str
    type: UnitType
    faction_id: str
    hp: int = 10
    x: int
    y: int
    carried_relic: bool = False
    rallied_to: Optional[str] = None  # captain unit id

class Faction(BaseModel):
    id: str
    name: str
    color: str
    resources: Dict[str, int] = Field(default_factory=lambda: {"food": 0, "lumber": 0})
    morale: int = 100
    units: List[Unit] = Field(default_factory=list)

# --- Orders & Turn ---
OrderType = Literal["move", "harvest", "build", "bombard", "rally"]

class Order(BaseModel):
    unit_id: str
    type: OrderType
    target: Optional[Tuple[int, int]] = None
    payload: Optional[Dict] = None

class Game(BaseModel):
    id: Optional[str] = None
    turn: int = 1
    phase: Literal["player", "ai"] = "player"
    map: GameMap
    factions: List[Faction]
    relic_pos: Tuple[int, int]
    neutral_guardians: List[Unit] = Field(default_factory=list)
    queued_orders: List[Order] = Field(default_factory=list)
