import os
import random
import uuid
from typing import List, Dict, Tuple, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import db, create_document, get_documents
from schemas import Game, GameMap, MapTile, Faction, Unit, Order

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Utility generation ----
TERRAINS = ["plains", "forest", "fields", "mountain", "water"]
WALKABLE = {"plains", "forest", "fields", "bridge"}
BLOCKING = {"mountain", "water", "wall"}


def neighbors(x: int, y: int, w: int, h: int):
    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            yield nx, ny


def generate_map(width: int = 24, height: int = 24) -> GameMap:
    tiles: List[MapTile] = []
    # Simple noise-like distribution
    for y in range(height):
        for x in range(width):
            r = random.random()
            if r < 0.65:
                t = "plains"
            elif r < 0.78:
                t = "forest"
            elif r < 0.90:
                t = "fields"
            elif r < 0.96:
                t = "water"
            else:
                t = "mountain"
            tiles.append(MapTile(x=x, y=y, terrain=t))

    # Carve a central plateau for the relic
    cx, cy = width // 2, height // 2
    for ny in range(cy-1, cy+2):
        for nx in range(cx-1, cx+2):
            idx = ny * width + nx
            tiles[idx].terrain = "plains"
    return GameMap(width=width, height=height, tiles=tiles)


def place_factions(gmap: GameMap) -> List[Faction]:
    w, h = gmap.width, gmap.height
    corners = [(1,1), (w-2, h-2)]  # player vs AI two-faction start
    colors = ["#60a5fa", "#f87171"]
    names = ["Player", "AI"]
    factions: List[Faction] = []
    for i, (sx, sy) in enumerate(corners):
        fid = f"F{i+1}"
        units = [
            Unit(id=str(uuid.uuid4()), type="ruler", faction_id=fid, hp=14, x=sx, y=sy),
            Unit(id=str(uuid.uuid4()), type="infantry", faction_id=fid, x=sx+1, y=sy),
            Unit(id=str(uuid.uuid4()), type="infantry", faction_id=fid, x=sx, y=sy+1),
            Unit(id=str(uuid.uuid4()), type="worker", faction_id=fid, x=sx+2, y=sy),
            Unit(id=str(uuid.uuid4()), type="catapult", faction_id=fid, x=sx, y=sy+2),
        ]
        factions.append(Faction(id=fid, name=names[i], color=colors[i], resources={"food": 2, "lumber": 2}, units=units))
    return factions


# ---- Influence Map ----

def compute_influence(game: Game) -> Dict[str, List[float]]:
    w, h = game.map.width, game.map.height
    size = w * h
    threat = [0.0] * size
    safety = [0.0] * size
    attraction = [0.0] * size

    def idx(x: int, y: int) -> int:
        return y * w + x

    # Threat projection: combat units radiate 1/distance falloff
    for f in game.factions:
        for u in f.units:
            base = 3 if u.type == "ruler" else 2 if u.type == "infantry" else 2.5 if u.type == "catapult" else 0
            if base == 0:
                continue
            for y in range(h):
                for x in range(w):
                    d = abs(x - u.x) + abs(y - u.y)
                    if d == 0:
                        v = base
                    else:
                        v = base / (d + 0.5)
                    # Enemy threat; own units add to safety instead
                    if f.name == "AI":
                        threat[idx(x, y)] += v
                    else:
                        safety[idx(x, y)] += v

    # Safety crowding penalty for player's side
    for y in range(h):
        for x in range(w):
            i = idx(x, y)
            # Too much safety becomes inefficient
            if safety[i] > 4:
                safety[i] -= (safety[i] - 4) * 0.5

    # Attraction toward vulnerable workers and relic
    rx, ry = game.relic_pos
    for y in range(h):
        for x in range(w):
            d = abs(x - rx) + abs(y - ry)
            attraction[idx(x, y)] += 3.0 / (d + 1)
    for f in game.factions:
        for u in f.units:
            if u.type == "worker":
                for y in range(h):
                    for x in range(w):
                        d = abs(x - u.x) + abs(y - u.y)
                        attraction[idx(x, y)] += 1.0 / (d + 1)

    return {"threat": threat, "safety": safety, "attraction": attraction}


# ---- Movement & Resolution ----

def is_passable(tile: MapTile) -> bool:
    if tile.structure == "bridge":
        return True
    if tile.structure == "wall":
        return False
    return tile.terrain in WALKABLE


def roll_movement() -> int:
    # stochastic logistics: skewed 1-3 steps
    r = random.random()
    if r < 0.15:
        return 0  # stall
    elif r < 0.55:
        return 1
    elif r < 0.85:
        return 2
    else:
        return 3


class NewGameRequest(BaseModel):
    width: int = 24
    height: int = 24


@app.post("/games")
def create_game(req: NewGameRequest):
    gmap = generate_map(req.width, req.height)
    factions = place_factions(gmap)
    relic_pos = (gmap.width // 2, gmap.height // 2)
    game = Game(
        map=gmap,
        factions=factions,
        relic_pos=relic_pos,
        neutral_guardians=[],
    )
    gid = create_document("game", game.model_dump())
    game.id = gid
    return game


@app.get("/games/{gid}")
def get_game(gid: str):
    docs = get_documents("game", {"_id": {"$exists": True, "$type": "objectId"}})
    # Fallback: fetch by string matching stored id field
    matches = [g for g in docs if str(g.get("_id", "")) == gid or g.get("id") == gid]
    if not matches:
        # Try any stored game
        docs = get_documents("game")
        if not docs:
            raise HTTPException(404, "Game not found")
        doc = docs[-1]
    else:
        doc = matches[0]
    # Convert to pydantic-friendly dict
    doc["id"] = gid if gid else str(doc.get("_id"))
    return doc


class CommandRequest(BaseModel):
    unit_id: str
    type: str
    target: Optional[Tuple[int, int]] = None
    payload: Optional[Dict] = None


@app.post("/games/{gid}/command")
def queue_command(gid: str, cmd: CommandRequest):
    # naive: store command list under the game document
    docs = get_documents("game")
    target = None
    for d in docs:
        if str(d.get("_id")) == gid or d.get("id") == gid:
            target = d
            break
    if not target:
        raise HTTPException(404, "Game not found")
    q = target.get("queued_orders", [])
    q.append(cmd.model_dump())
    # direct update
    db["game"].update_one({"_id": target["_id"]}, {"$set": {"queued_orders": q}})
    return {"ok": True, "queued": len(q)}


@app.get("/games/{gid}/influence")
def get_influence(gid: str):
    docs = get_documents("game")
    target = None
    for d in docs:
        if str(d.get("_id")) == gid or d.get("id") == gid:
            target = d
            break
    if not target:
        raise HTTPException(404, "Game not found")
    game = Game(**{k: v for k, v in target.items() if k != "_id"})
    infl = compute_influence(game)
    return infl


@app.post("/games/{gid}/end-turn")
def end_turn(gid: str):
    docs = get_documents("game")
    target = None
    for d in docs:
        if str(d.get("_id")) == gid or d.get("id") == gid:
            target = d
            break
    if not target:
        raise HTTPException(404, "Game not found")

    game = Game(**{k: v for k, v in target.items() if k != "_id"})
    w, h = game.map.width, game.map.height

    # Execute queued orders (simplified)
    tile_at = lambda x, y: game.map.tiles[y * w + x]
    for order in game.queued_orders:
        try:
            if order.type == "move" and order.target is not None:
                u = next(u for f in game.factions for u in f.units if u.id == order.unit_id)
                steps = roll_movement()
                tx, ty = order.target
                # simple greedy step toward target within steps
                for _ in range(steps):
                    dx = 1 if tx > u.x else -1 if tx < u.x else 0
                    dy = 1 if ty > u.y else -1 if ty < u.y else 0
                    nx, ny = u.x + dx, u.y + dy
                    if 0 <= nx < w and 0 <= ny < h and is_passable(tile_at(nx, ny)):
                        u.x, u.y = nx, ny
                    else:
                        break
            elif order.type == "harvest":
                u = next(u for f in game.factions for u in f.units if u.id == order.unit_id)
                t = tile_at(u.x, u.y).terrain
                f = next(f for f in game.factions if f.id == u.faction_id)
                if t == "forest":
                    f.resources["lumber"] = f.resources.get("lumber", 0) + 1
                if t == "fields":
                    f.resources["food"] = f.resources.get("food", 0) + 1
            elif order.type == "build" and order.payload:
                # build bridge over water or wall on plains if resources
                u = next(u for f in game.factions for u in f.units if u.id == order.unit_id)
                f = next(f for f in game.factions if f.id == u.faction_id)
                kind = order.payload.get("structure")
                t = tile_at(u.x, u.y)
                if kind == "bridge" and t.terrain == "water" and f.resources.get("lumber", 0) >= 2:
                    t.structure = "bridge"
                    f.resources["lumber"] -= 2
                if kind == "wall" and t.terrain in {"plains", "fields", "forest"} and f.resources.get("lumber", 0) >= 3:
                    t.structure = "wall"
                    f.resources["lumber"] -= 3
            elif order.type == "bombard" and order.target is not None:
                # catapult damages walls from 2-4 tiles away
                u = next(u for f in game.factions for u in f.units if u.id == order.unit_id)
                tx, ty = order.target
                d = abs(tx - u.x) + abs(ty - u.y)
                if u.type == "catapult" and 2 <= d <= 4:
                    tt = tile_at(tx, ty)
                    if tt.structure == "wall":
                        tt.structure = None
        except StopIteration:
            continue

    # Clear orders
    game.queued_orders = []

    # Simple AI: choose a captain and move units along attraction-safety gradient toward relic
    if game.phase == "player":
        game.phase = "ai"
    else:
        infl = compute_influence(game)
        def best_step(u: Unit):
            best = (u.x, u.y)
            best_score = -1e9
            for nx, ny in neighbors(u.x, u.y, w, h):
                if not is_passable(tile_at(nx, ny)):
                    continue
                i = ny * w + nx
                score = infl["attraction"][i] - infl["threat"][i] + infl["safety"][i] * 0.3
                if score > best_score:
                    best_score = score
                    best = (nx, ny)
            return best
        for f in game.factions:
            if f.name != "AI":
                continue
            # choose a captain (ruler preferred)
            captain = next((u for u in f.units if u.type == "ruler"), f.units[0])
            # rally nearby units (within 4)
            cohort = []
            for u in f.units:
                d = abs(u.x - captain.x) + abs(u.y - captain.y)
                if d <= 4:
                    cohort.append(u)
            # move cohort 1 step along gradient
            for u in cohort:
                nx, ny = best_step(u)
                u.x, u.y = nx, ny
        game.phase = "player"
        game.turn += 1

    # Victory checks: if a ruler carries relic to corner, win (simplified placeholder)
    # Relic pickup
    rx, ry = game.relic_pos
    for f in game.factions:
        for u in f.units:
            if (u.x, u.y) == (rx, ry) and u.type in {"infantry", "ruler"}:
                u.carried_relic = True

    # Persist
    db["game"].update_one({"_id": target["_id"]}, {"$set": game.model_dump()})
    return game


@app.get("/")
def read_root():
    return {"message": "Strategy Prototype Backend Running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set"
            response["database_name"] = db.name
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
