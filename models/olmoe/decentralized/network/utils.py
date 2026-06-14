from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

def load_placement(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        placement = yaml.safe_load(f)

    validate_placement(placement)
    return placement


def validate_placement(placement: dict[str, Any]) -> None:
    num_clients = placement["num_clients"]
    num_experts = placement["num_experts"]
    clients = placement["clients"]

    if len(clients) != num_clients:
        raise ValueError(
            f"Expected {num_clients} clients, got {len(clients)} clients in config"
        )

    seen_experts: set[int] = set()

    for client in clients:
        rank = client["rank"]
        expert_ids = client["expert_ids"]

        if len(expert_ids) == 0:
            raise ValueError(f"Client rank={rank} has no experts")

        for expert_id in expert_ids:
            if expert_id in seen_experts:
                raise ValueError(f"Duplicate expert_id={expert_id} in placement")
            if expert_id < 0 or expert_id >= num_experts:
                raise ValueError(
                    f"Invalid expert_id={expert_id}; expected range [0, {num_experts - 1}]"
                )
            seen_experts.add(expert_id)

    expected = set(range(num_experts))
    missing = expected - seen_experts
    extra = seen_experts - expected

    if missing:
        raise ValueError(f"Missing experts: {sorted(missing)}")

    if extra:
        raise ValueError(f"Unexpected experts: {sorted(extra)}")


def get_client_config(placement: dict[str, Any], rank: int) -> dict[str, Any]:
    for client in placement["clients"]:
        if client["rank"] == rank:
            return client

    raise KeyError(f"No client with rank={rank} found in placement")


def map_ec (placement: dict[str, Any]) -> dict[int, dict[str, Any]]:
    expert_to_client: dict[int, dict[str, Any]] = {}

    for client in placement["clients"]:
        address = f"tcp://{client['host']}:{client['port']}"

        for expert_id in client["expert_ids"]:
            expert_to_client[expert_id] = {
                "rank": client["rank"],
                "host": client["host"],
                "port": client["port"],
                "address": address,
                "device": client.get("device", "cpu"),
            }

    return expert_to_client