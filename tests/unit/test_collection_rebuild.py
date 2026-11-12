"""Stateful Qdrant REST boundary for offline alias/rebuild regressions."""

from copy import deepcopy
from dataclasses import dataclass


@dataclass
class Response:
    status_code: int
    result: object

    def json(self):
        return {"result": self.result}

    @property
    def text(self):
        return str(self.result)


class FakeQdrant:
    def __init__(self):
        self.collections = {}
        self.aliases = {}
        self.requests = []
        self.fail_write = False
        self.drop_write = False
        self.fail_publish = False

    def request(self, method, path, json=None):
        self.requests.append((method, path, deepcopy(json)))
        if path == "/aliases":
            return Response(
                200,
                {
                    "aliases": [
                        {"alias_name": alias, "collection_name": name}
                        for alias, name in self.aliases.items()
                    ]
                },
            )
        if path == "/collections/aliases":
            if self.fail_publish:
                return Response(500, "publish failed")
            aliases = dict(self.aliases)
            for action in json["actions"]:
                if "delete_alias" in action:
                    aliases.pop(action["delete_alias"]["alias_name"])
                else:
                    alias = action["create_alias"]["alias_name"]
                    name = action["create_alias"]["collection_name"]
                    if alias in self.collections or name not in self.collections:
                        return Response(409, "alias conflict")
                    aliases[alias] = name
            self.aliases = aliases
            return Response(200, True)
        parts = path.split("?")[0].strip("/").split("/")
        logical = parts[1]
        name = self.aliases.get(logical, logical)
        collection = self.collections.get(name)
        if len(parts) == 2:
            if method == "GET":
                if collection is None:
                    return Response(404, None)
                return Response(
                    200,
                    {
                        "config": {
                            "params": {
                                "vectors": {
                                    "size": collection["size"],
                                    "distance": collection["distance"],
                                }
                            }
                        }
                    },
                )
            if method == "PUT":
                assert name not in self.collections
                self.collections[name] = {**json["vectors"], "points": {}}
                return Response(200, True)
        assert collection is not None, (method, path)
        action = "/".join(parts[2:])
        if action == "index":
            return Response(200, True)
        if action == "points" and method == "PUT":
            for index, point in enumerate(json["points"]):
                if self.drop_write:
                    continue
                collection["points"][point["id"]] = deepcopy(point)
                if self.fail_write and index == 0:
                    return Response(500, "partial batch failure")
            return Response(200, True)
        if action == "points/scroll":
            assert json["with_vector"] and json["with_payload"]
            points = list(collection["points"].values())
            offset = int(json.get("offset", 0))
            # Exercise pagination even for small fixtures.
            end = offset + 2
            return Response(
                200,
                {
                    "points": deepcopy(points[offset:end]),
                    "next_page_offset": end if end < len(points) else None,
                },
            )
        if action == "points/search":
            hits = [
                {
                    "id": point["id"],
                    "payload": deepcopy(point["payload"]),
                    "score": sum(
                        a * b for a, b in zip(point["vector"], json["vector"], strict=True)
                    ),
                }
                for point in collection["points"].values()
            ]
            hits.sort(key=lambda point: -point["score"])
            return Response(200, hits[: json["limit"]])
        # Delete document/collection operations must never occur on this route.
        raise AssertionError(f"unexpected request {method} {path}")

    def close(self):
        pass
