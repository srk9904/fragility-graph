// Fragility Graph - Neo4j Schema Initialization

// 1. Uniqueness Constraints
CREATE CONSTRAINT file_path IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE;
CREATE CONSTRAINT func_id IF NOT EXISTS FOR (f:Function) REQUIRE f.id IS UNIQUE;
CREATE CONSTRAINT class_id IF NOT EXISTS FOR (c:Class) REQUIRE c.id IS UNIQUE;

// 2. Indexes for Performance
CREATE INDEX node_name_idx IF NOT EXISTS FOR (n:Function) ON (n.name);
CREATE INDEX file_lang_idx IF NOT EXISTS FOR (f:File) ON (f.language);

// 3. Initial "Structural Knowledge" Nodes (Optional Seed)
MERGE (s:System {name: "FragilityGraph"})
SET s.initialized_at = datetime(),
    s.version = "1.0.0";
