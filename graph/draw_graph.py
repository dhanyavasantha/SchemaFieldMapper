"""
draw_graph.py
-------------
Utility to visualise the Schema Field Mapper LangGraph as a PNG image.

Usage:
    python3 -m graph.draw_graph
"""
import sys
import os

# Add project root directory to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from graph.mapper_graph import build_schema_mapper_graph


def draw():
    """Draw the LangGraph pipeline and save as PNG."""
    graph = build_schema_mapper_graph()
    
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        output_path = "schema_mapper_graph.png"
        with open(output_path, "wb") as f:
            f.write(png_data)
        print(f"Graph saved to: {output_path}")
    except Exception as e:
        # Fallback: print mermaid source
        print("Could not render PNG (install graphviz or use mermaid).")
        print(f"Error: {e}")
        print("\nMermaid diagram source:")
        print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    draw()
