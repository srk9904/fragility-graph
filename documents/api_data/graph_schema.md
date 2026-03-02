# Graph Schema Definition (Neo4j)

## 1. Node Labels

### `File`
- `path`: String (Unique ID)
- `extension`: String
- `last_modified`: ISO8601

### `Function` / `Method`
- `id`: String (file_path::name)
- `name`: String
- `line_start`: Integer
- `line_end`: Integer
- `complexity`: Integer (Cyclomatic)
- `fragility_score`: Float (Predicted)

### `Class`
- `id`: String (file_path::name)
- `name`: String

## 2. Relationship Types

### `CALLS`
- **Source:** `Function`
- **Target:** `Function`
- Directional: A calls B.

### `DEFINES`
- **Source:** `File` or `Class`
- **Target:** `Function` or `Class`

### `IMPORTS`
- **Source:** `File`
- **Target:** `File`

### `CO_CHANGE` (Inferred from Git)
- **Source:** `File`
- **Target:** `File`
- `weight`: Integer (frequency of simultaneous changes)

## 3. Graph Logic
- The Graph Neural Network (GNN) will use the `CALLS` and `IMPORTS` topology as the primary adjacency matrix.
- `fragility_score` is a property updated by the ML engine.
