# graph_operations.py

from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
# Uncomment the following line if you plan to use GraphWidget for visualization
# from yfiles_jupyter_graphs import GraphWidget

class Neo4jHandler:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri=uri, auth=(username, password))
        self.graph = Neo4jGraph()
    
    def add_documents(self, graph_documents):
        self.graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
    
    def run_cypher(self, cypher):
        session = self.driver.session()
        result = session.run(cypher)
        return result.graph()
    
    def create_fulltext_index(self):
        with self.driver.session() as session:
            session.run("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
    
    # If you plan to visualize the graph, uncomment this method
    # def show_graph(self, cypher):
    #     widget = GraphWidget(graph=self.run_cypher(cypher))
    #     widget.node_label_mapping = 'id'
    #     display(widget)
    #     return widget
