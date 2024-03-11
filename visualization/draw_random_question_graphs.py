import argparse
import logging
import os
import random

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Optional, TypedDict


def clean_entity(entity: str) -> str:
    """
    This function cleans the entity such that it can be used as a filename.

    Args:
        entity (str): The entity to clean.

    Returns:
        str: The cleaned entity.
    """
    return entity.replace(" ", "_").replace(",", "").replace(":", "_")


def split_node_labels(node_label: str, max_line_length=16, min_line_length=4):
    """
    This function splits the node labels into multiple lines if they are too long.
    It ensures that the label is not split into lines that are too short.

    Args:
        node_label (str): The label of the node.
        max_line_length (int, optional): The maximum length of a line. Defaults to 16.
        min_line_length (int, optional): The minimum length of a line. Defaults to 4.

    Returns:
        str: The node label split into multiple lines if necessary.
    """
    label_length = len(node_label)
    accum_length = 0  # accumulated length
    lines = []
    words = node_label.split()
    current_string = f"{words[0]}"

    # Iterate over the rest of the words
    for word in words[1:]:
        current_length = len(current_string)
        word_length = len(word)

        # If adding the current word to the current string would make it too long,
        # and the remaining label is long enough to form another line,
        # add the current string to the list of lines and start a new string with the current word
        if (
            current_length + word_length > max_line_length
            and label_length - accum_length > min_line_length
        ):
            lines.append(current_string)
            current_string = word
            accum_length += len(current_string)
        else:
            # Otherwise, add the current word to the current string
            current_string += f" {word}"

    # If there is a non-empty string left, add it to the list of lines
    if len(current_string) > 0:
        lines.append(current_string)

    # Join the lines with newline characters and return the result
    return "\n".join(lines)


def parse_args():
    """
    Parse command line arguments.

    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_tsv", type=str, default="../data/tsv/train.tsv"
    )  # Path to the TSV file
    parser.add_argument(
        "--num_questions", type=int, default=10, required=False
    )  # Number of questions to sample
    parser.add_argument(
        "--output_dir", type=str, default="../question_graph_examples/"
    )  # Output directory for the images

    args = parser.parse_args()

    return args


class Node(TypedDict):
    type: str
    name_: str
    id: int
    label: str


class Link(TypedDict):
    name_: str
    source: int
    target: int
    label: str


class Graph(TypedDict):
    nodes: List[Node]
    links: List[Link]
    directed: Optional[bool]


def main(args):
    input_tsv: str = args.input_tsv
    num_questions: int = args.num_questions
    output_dir: str = args.output_dir
    if os.path.exists(output_dir) and output_dir != "":
        for question_folder in os.scandir(output_dir):
            for file in os.scandir(question_folder):
                os.remove(file.path)

    if not os.path.exists(output_dir) and output_dir != "":
        os.makedirs(output_dir)

    df = pd.read_csv(input_tsv, sep="\t", encoding="utf-8")
    qs = tuple(df["question"].unique())
    random_qs = random.sample(qs, k=num_questions)

    for i, question in enumerate(random_qs):
        ddf = df[df["question"] == question]  # All the rows for the question
        for _, row in ddf.iterrows():
            true_flag = row["correct"]
            assert isinstance(true_flag, bool)
            assert row["question"] == question
            graph_json: Graph = eval(row["graph"])
            graph_json["directed"] = True

            sample_id = row["sample_id"]
            candidate_entity: str = (
                row["answerEntity"].replace("/", "|").replace("\\", "|")
            )
            true_entity: str = (
                row["groundTruthAnswerEntity"].replace("/", "|").replace("\\", "|")
            )

            nx_graph = nx.node_link_graph(
                graph_json,
            )

            candidate_color = "green" if true_flag else "red"
            color_map = {
                "ANSWER_CANDIDATE_ENTITY": candidate_color,
                "QUESTIONS_ENTITY": "#2A4CC6",
            }
            node_colors: dict[int, str] = {}
            labels: dict[int, str] = {}
            for node_dict in graph_json["nodes"]:
                node_id = node_dict["id"]
                node_label = node_dict["label"]
                node_label = "None" if node_label is None else node_label
                node_type = node_dict["type"]
                node_label = split_node_labels(
                    node_label, max_line_length=13, min_line_length=4
                )
                labels[node_id] = node_label
                color = (
                    color_map[node_type]
                    if color_map.get(node_type) is not None
                    else "#808080"
                )
                node_colors[node_id] = color

            node_colors_np = [node_colors[key] for key in sorted(node_colors.keys())]
            node_colors_np = np.array(node_colors_np)
            edge_labels: dict[tuple[int, int], str] = {}
            for edge_dict in graph_json["links"]:
                src_i = edge_dict["source"]
                trg_i = edge_dict["target"]
                e_label = edge_dict["label"]
                e_label = split_node_labels(
                    e_label, max_line_length=12, min_line_length=3
                )
                edge_labels[(src_i, trg_i)] = e_label

            plt.title(
                split_node_labels(question, max_line_length=64, min_line_length=16),
                fontsize=12,
            )
            # pos = nx.spring_layout(nx_graph)
            try:
                pos = nx.planar_layout(nx_graph, scale=0.05)  # type: ignore
            except Exception:
                pos = nx.spring_layout(nx_graph, scale=0.1)  # type: ignore
            nx.draw(
                nx_graph,
                pos=pos,
                node_size=250,
                alpha=0.8,
                node_color=node_colors_np,
                font_size=12,
                font_weight="bold",
            )

            pos_edge_labels = {}
            y_off = 0.05
            max_pos = max(v[1] for v in pos.values())
            min_pos = min(v[1] for v in pos.values())
            delta_pos = max_pos - min_pos
            for k, v in pos.items():
                pos_edge_labels[k] = (v[0], v[1] + y_off * delta_pos)

            nx.draw_networkx_edge_labels(
                nx_graph,
                pos_edge_labels,
                edge_labels=edge_labels,
                font_color="red",
                label_pos=0.375,
                font_size=6,
            )
            pos_node_labels = {}
            y_off = 0.075  # offset on the y axis
            max_pos = max(v[1] for v in pos.values())
            min_pos = min(v[1] for v in pos.values())
            delta_pos = max_pos - min_pos

            for k, v in pos.items():
                offset = y_off * delta_pos

                pos_node_labels[k] = (v[0], v[1] - offset)
            nx.draw_networkx_labels(nx_graph, pos_node_labels, labels, font_size=8)

            cleaned_candidate_entity = clean_entity(candidate_entity)
            cleaned_true_entity = clean_entity(true_entity)

            fname = f"{cleaned_true_entity}_{sample_id}_{cleaned_candidate_entity}.png"
            output_subdir = os.path.join(output_dir, f"question_{i}/")
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            x0, x1 = plt.xlim()
            y0, y1 = plt.ylim()
            plt.xlim(x0 * 1.1, x1 * 1.1)
            plt.ylim(y0 * 1.1, y1 * 1.1)

            output_graph_path = os.path.join(output_dir, f"question_{i}/", fname)
            plt.savefig(output_graph_path, format="PNG")
            plt.clf()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    main(args)
