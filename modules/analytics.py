# This file is part of Mimir.

# Copyright (C) 2025 Andrés Lillo Ortiz

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import pandas as pd

def get_stats(graph_db):
    """Returns basic graph statistics (node and edge counts)."""
    query = """
    MATCH (n)
    OPTIONAL MATCH (n)-[r]->()
    RETURN count(DISTINCT n) as nodes, count(r) as edges
    """
    data = graph_db.query(query)
    return data[0]

def run_pagerank(graph_db, limit=10):
    """
    Executes PageRank to find the most influential entities in the graph.
    """
    graph_name = "mimir_projection"

    # 1. Clean up previous projection if it exists (safeguard)
    try:
        graph_db.query(f"CALL gds.graph.drop('{graph_name}', false) YIELD graphName")
    except:
        pass

    try:
        # 2. Create In-Memory Projection (Required by GDS)
        # We project everything to analyze the full structure
        graph_db.query(f"""
        CALL gds.graph.project(
            '{graph_name}',
            '*',
            '*'
        )
        """)

        # 3. Execute PageRank Stream
        # We filter out 'Chunk' nodes to focus on Concepts/Entities
        query = f"""
        CALL gds.pageRank.stream('{graph_name}')
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS node, score
        WHERE NOT 'Chunk' IN labels(node) AND node.id IS NOT NULL
        RETURN node.id AS Entity, score AS Score
        ORDER BY Score DESC
        LIMIT {limit}
        """
        result = graph_db.query(query)

        # 4. Cleanup
        graph_db.query(f"CALL gds.graph.drop('{graph_name}', false) YIELD graphName")

        return pd.DataFrame(result)

    except Exception as e:
        print(f"GDS Error: {e}")
        # Cleanup in case of error
        try: graph_db.query(f"CALL gds.graph.drop('{graph_name}', false) YIELD graphName")
        except: pass
        return pd.DataFrame()

def run_community_detection(graph_db):
    """
    Executes Louvain algorithm to detect communities (clustered topics).
    """
    graph_name = "mimir_community"

    try:
        graph_db.query(f"CALL gds.graph.drop('{graph_name}', false) YIELD graphName")

        # Only projecting Entity-to-Entity relationships for cleaner topics
        graph_db.query(f"CALL gds.graph.project('{graph_name}', '*', '*')")

        query = f"""
        CALL gds.louvain.stream('{graph_name}')
        YIELD nodeId, communityId
        WITH gds.util.asNode(nodeId) AS node, communityId
        WHERE NOT 'Chunk' IN labels(node) AND node.id IS NOT NULL
        RETURN communityId as Community, count(node) as Members, collect(node.id)[..5] as Examples
        ORDER BY Members DESC
        LIMIT 10
        """
        result = graph_db.query(query)

        graph_db.query(f"CALL gds.graph.drop('{graph_name}', false) YIELD graphName")

        return pd.DataFrame(result)
    except Exception as e:
        print(f"GDS Error: {e}")
        try: graph_db.query(f"CALL gds.graph.drop('{graph_name}', false) YIELD graphName")
        except: pass
        return pd.DataFrame()