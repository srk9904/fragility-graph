import os
import logging
from neo4j import GraphDatabase
from ..services.config import settings

logger = logging.getLogger(__name__)

class Neo4jAdapter:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI, 
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def update_from_ast(self, payload):
        """Updates the graph with nodes, edges, and file relationships."""
        file_path = payload.get("file_path")
        nodes = payload.get("nodes", [])
        
        if not self.driver:
            logger.warning("Neo4j Driver not initialized. Skipping graph update.")
            return

        with self.driver.session() as session:
            try:
                # 1. Ensure File node exists
                session.run("MERGE (f:File {path: $path})", path=file_path)

                for node in nodes:
                    # 2. Create/Update Function Node
                    session.run(
                        "MERGE (n:Function {name: $name, file: $file}) "
                        "SET n.complexity = $complexity",
                        name=node["name"], file=file_path, complexity=node.get("complexity", 1)
                    )
                    
                    # 3. Link Function to File
                    session.run(
                        "MATCH (f:File {path: $path}), (n:Function {name: $name, file: $path}) "
                        "MERGE (n)-[:BELONGS_TO]->(f)",
                        path=file_path, name=node["name"]
                    )

                    # 4. Create Edges (Calls)
                    for call in node.get("calls", []):
                        session.run(
                            "MATCH (a:Function {name: $a_name, file: $file}), (b:Function {name: $b_name}) "
                            "MERGE (a)-[:CALLS]->(b)",
                            a_name=node["name"], b_name=call, file=file_path
                        )
                logger.info(f"Graph updated for {file_path}")
            except Exception as e:
                logger.error(f"Neo4j Update Error: {e}")

    def add_co_change(self, file_a, file_b, frequency):
        """Adds a temporal relationship based on Git history."""
        if not self.driver: return
        with self.driver.session() as session:
            session.run(
                "MATCH (a:File {path: $file_a}), (b:File {path: $file_b}) "
                "MERGE (a)-[r:CO_CHANGES_WITH]->(b) "
                "SET r.frequency = $freq",
                file_a=file_a, file_b=file_b, freq=frequency
            )

    @staticmethod
    def _create_file_node(tx, path):
        tx.run("MERGE (f:File {path: $path}) ON CREATE SET f.last_modified = timestamp()", path=path)

    @staticmethod
    def _upsert_symbol_node(tx, node_id, node_data):
        tx.run("""
            MERGE (s:Symbol {id: $id})
            SET s.name = $name,
                s.type = $type,
                s.line_start = $line_start,
                s.line_end = $line_end,
                s.complexity = $complexity
            """, 
            id=node_id, 
            name=node_data['name'],
            type=node_data.get('type', 'function'),
            line_start=node_data.get('line_start'),
            line_end=node_data.get('line_end'),
            complexity=node_data.get('complexity', 1)
        )

    @staticmethod
    def _link_file_to_symbol(tx, file_path, node_id):
        tx.run("""
            MATCH (f:File {path: $file_path})
            MATCH (s:Symbol {id: $node_id})
            MERGE (f)-[:DEFINES]->(s)
            """, file_path=file_path, node_id=node_id)

    @staticmethod
    def _create_call_edge(tx, source_id, target_id):
        tx.run("""
            MERGE (s1:Symbol {id: $source_id})
            MERGE (s2:Symbol {id: $target_id})
            MERGE (s1)-[:CALLS]->(s2)
            """, source_id=source_id, target_id=target_id)

    def get_neighbors(self, node_id, hops=2):
        # Implementation for GNN neighborhood retrieval
        pass
