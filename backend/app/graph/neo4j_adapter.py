"""
Neo4j Graph Database Adapter
Provides CRUD operations for the FragilityGraph dependency graph.
"""
import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase

from app.config import settings

logger = logging.getLogger(__name__)


class Neo4jAdapter:
    _instance: Optional["Neo4jAdapter"] = None

    def __init__(self):
        self.driver = None

    # ── Singleton ──────────────────────────────────────────
    @classmethod
    def get_instance(cls) -> "Neo4jAdapter":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Connection lifecycle ───────────────────────────────
    def connect(self):
        if not settings.NEO4J_URI:
            logger.warning("NEO4J_URI not configured – graph features disabled")
            return
        try:
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
            )
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s", settings.NEO4J_URI)
        except Exception as e:
            logger.error("Failed to connect to Neo4j: %s", e)
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    # ── Write operations ───────────────────────────────────
    def create_node(
        self,
        node_id: str,
        label: str,
        node_type: str,
        file_path: str,
        line_number: int,
        fragility: float = 0.0,
    ):
        if not self.driver:
            return
        query = """
        MERGE (n:CodeNode {id: $id})
        SET n.label     = $label,
            n.type      = $node_type,
            n.file_path = $file_path,
            n.line_number = $line_number,
            n.fragility = $fragility
        """
        with self.driver.session() as session:
            session.run(
                query,
                id=node_id,
                label=label,
                node_type=node_type,
                file_path=file_path,
                line_number=line_number,
                fragility=fragility,
            )

    def create_dependency(self, source_id: str, target_id: str, relationship: str = "CALLS"):
        if not self.driver:
            return
        query = f"""
        MATCH (a:CodeNode {{id: $source}})
        MATCH (b:CodeNode {{id: $target}})
        MERGE (a)-[r:{relationship}]->(b)
        """
        with self.driver.session() as session:
            session.run(query, source=source_id, target=target_id)

    def update_fragility_score(self, node_id: str, score: float):
        if not self.driver:
            return
        query = """
        MATCH (n:CodeNode {id: $id})
        SET n.fragility = $score
        """
        with self.driver.session() as session:
            session.run(query, id=node_id, score=score)

    # ── Read operations ────────────────────────────────────
    def get_all_nodes(self) -> List[Dict[str, Any]]:
        if not self.driver:
            return []
        query = "MATCH (n:CodeNode) RETURN n"
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record["n"]) for record in result]

    def get_all_edges(self) -> List[Dict[str, Any]]:
        if not self.driver:
            return []
        query = "MATCH (a:CodeNode)-[r]->(b:CodeNode) RETURN a.id AS source, b.id AS target, type(r) AS relationship"
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    def get_dependencies(self, node_id: str, depth: int = 3) -> List[Dict[str, Any]]:
        if not self.driver:
            return []
        query = f"""
        MATCH (n:CodeNode {{id: $id}})-[*1..{depth}]->(dep:CodeNode)
        RETURN DISTINCT dep
        """
        with self.driver.session() as session:
            result = session.run(query, id=node_id)
            return [dict(record["dep"]) for record in result]

    # ── Bulk operations ────────────────────────────────────
    def clear_all(self):
        if not self.driver:
            return
        with self.driver.session() as session:
            session.run("MATCH (n:CodeNode) DETACH DELETE n")

    def bulk_upsert(self, nodes: List[Dict], edges: List[Dict]):
        """Efficiently upsert a batch of nodes and edges."""
        if not self.driver:
            return

        with self.driver.session() as session:
            # Upsert nodes
            for node in nodes:
                session.run(
                    """
                    MERGE (n:CodeNode {id: $id})
                    SET n.label = $label,
                        n.type = $type,
                        n.file_path = $file_path,
                        n.line_number = $line_number,
                        n.fragility = $fragility
                    """,
                    **node,
                )
            # Upsert edges
            for edge in edges:
                rel = edge.get("relationship", "CALLS")
                session.run(
                    f"""
                    MATCH (a:CodeNode {{id: $source}})
                    MATCH (b:CodeNode {{id: $target}})
                    MERGE (a)-[:{rel}]->(b)
                    """,
                    source=edge["source"],
                    target=edge["target"],
                )
