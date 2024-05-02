# pylint: disable=redefined-outer-name
r"""
  ______                      _ ____           _                _
 |  ____|                    | |  _ \         | |              | |
 | |__ _ __ _   _  __ _  __ _| | |_) | ___  __| |_ __ ___   ___| | __
 |  __| '__| | | |/ _` |/ _` | |  _ < / _ \/ _` | '__/ _ \ / __| |/ /
 | |  | |  | |_| | (_| | (_| | | |_) |  __/ (_| | | | (_) | (__|   <
 |_|  |_|   \__,_|\__, |\__,_|_|____/ \___|\__,_|_|  \___/ \___|_|\_\
                   __/ |
                  |___/

FrugalGPT-style LLM cascades for fighting hallucinations.

  .-'''-.
 /* * * *\
:_.-:`:-._;
    (_)
 \|/(_)\|/
"""

import os
import json

from operator import itemgetter
from typing import (
    Any,
    List,
    Tuple,
    TypedDict
)

# HuggingFace 🤗
from datasets import load_dataset
from transformers import pipeline

# LangChain Core 🦜🔗
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnableBranch,
    RunnablePassthrough
)

# LangChain AWS ☁️
from langchain_aws import ChatBedrock

# LangGraph 🦜🕸️
from langgraph.graph import StateGraph, END

####################
# Scoring Function #
####################

# For more information, see
# https://vectara.com/blog/automating-hallucination-detection-introducing-vectara-factual-consistency-score/
hallucination_detector = pipeline(
    task="text-classification",
    model="vectara/hallucination_evaluation_model"
)

def scorer(inpt: dict) -> float:
    """
    Returns a score between 0 and 1 indicating whether the model is hallucinating.
    """
    result = hallucination_detector({'text': inpt['context'], 'text_pair': inpt['answer']})
    return result['score']  # 0 -> hallucination,  1 -> factually consistent

def batch_scorer(inpts: List[dict]) -> List[float]:
    """
    Scores multiple inputs in one go.
    """
    inputs = list(map(lambda inpt: {'text': inpt['context'], 'text_pair': inpt['answer']}, inpts))
    results = hallucination_detector(inputs)
    return list(map(lambda result: result['score'], results))

#############
# LLM Chain #
#############

def build_chain(model_id: str, threshold: float, scoringf: callable = scorer):
    """
    Creates a model chain for question answering.
    """
    prompt = ChatPromptTemplate.from_template("{context}\n\n{question}")
    llm = ChatBedrock(model_id=model_id, model_kwargs={'temperature': 0})
    output_parser = StrOutputParser()
    return RunnableParallel({
        'context': itemgetter('context'),
        'question': itemgetter('question'),
        'model': lambda _: model_id,
        'answer': prompt | llm | output_parser
    }) | RunnableParallel({
        'context': itemgetter('context'),
        'question': itemgetter('question'),
        'model': itemgetter('model'),
        'answer': itemgetter('answer'),
        'score': RunnableLambda(scoringf),
        'threshold': lambda _: str(threshold)
    })

###############
# LLM Chain   #
# (LangGraph) #
###############

def build_chain_lg(model_id: str):
    """
    Creates a model chain for question answering (LangGraph version).
    """
    prompt = ChatPromptTemplate.from_template("{context}\n\n{question}")
    llm = ChatBedrock(model_id=model_id, model_kwargs={'temperature': 0})
    output_parser = StrOutputParser()
    return RunnableParallel({
        'context': itemgetter('context'),
        'question': itemgetter('question'),
        'model': lambda _: model_id,
        'answer': prompt | llm | output_parser
    })

###############
# LLM Cascade #
###############

def llm_cascade(model_ids: List[str], thresholds: List[float]):
    """
    Creates a static LLM cascade for question answering.
    """
    if len(model_ids) != len(thresholds):
        raise ValueError("The list of models and thresholds must have the same size.")

    def check_score(output):
        score = float(output['score'])
        threshold = float(output['threshold'])
        if score < threshold:
            return True
        return False

    chains = [
        build_chain(model_id, threshold) for model_id, threshold in zip(model_ids, thresholds)
    ]

    llmc = chains[0]
    for chain in chains[1:]:
        llmc = llmc | RunnableBranch(
            (lambda output: check_score(output), chain),  # pylint: disable=unnecessary-lambda
            RunnablePassthrough()
        )

    return llmc

###############
# LLM Cascade #
# (LangGraph) #
###############

class ModelState(TypedDict):
    """
    State class to pass the model output along the sequence.
    """
    context: str
    question: str
    model: str
    answer: str

def llm_cascade_lg(model_ids: List[str], thresholds: List[float], scoringf: callable = scorer):
    """
    Creates a static LLM cascade for question answering (LangGraph version).
    """
    if len(model_ids) != len(thresholds):
        raise ValueError("The list of models and thresholds must have the same size.")

    def check_score(state: ModelState):
        if scoringf(state) < thresholds[model_ids.index(state['model'])]:
            return "next_model"
        return "end"

    graph = StateGraph(ModelState)
    for i, model_id in enumerate(model_ids):
        graph.add_node(model_id, build_chain_lg(model_id))
        if i < len(model_ids) - 1:
            graph.add_conditional_edges(model_id, check_score, {
                'next_model': model_ids[i+1],
                'end': END
            })
        else:
            graph.add_edge(model_ids[-1], END)
    graph.set_entry_point(model_ids[0])

    return graph.compile()

#################
# Cost Tracking #
#################

if os.path.isfile("pricing.json"):
    with open("pricing.json", encoding="utf-8") as f:
        BEDROCK_PRICING = json.load(f)
else:
    BEDROCK_PRICING = None

def compute_cost(response: dict, models: List) -> Tuple[float, float]:
    """
    Returns the actual cost of running the chain and the
    predicted cost if we were to use only the best model.
    """
    end_model = response['model']
    query = response['context'] + response['question']
    answer = response['answer']

    # Assumption: 1 token ~ 4 chars
    in_tokens = len(query) // 4
    out_tokens = len(answer) // 4

    cost = 0
    for model in models:
        model_pricing = BEDROCK_PRICING[model]
        cost += (in_tokens/1000)*model_pricing.get('input', 0.0) + \
                (out_tokens/1000)*model_pricing.get('output', 0.0)
        if model == end_model:
            break

    # Assumption: output tokens (LLM cascade) == # output tokens (best model)
    best_model_pricing = BEDROCK_PRICING[models[-1]]
    best_cost = (in_tokens/1000)*best_model_pricing.get('input', 0.0) + \
                (out_tokens/1000)*best_model_pricing.get('output', 0.0)

    return cost, best_cost

###########
# Dataset #
###########

# Load a small sample of the SQuAD dataset
# https://huggingface.co/datasets/rajpurkar/squad
# https://huggingface.co/docs/transformers/en/tasks/question_answering#load-squad-dataset
squad = load_dataset("squad", split="train[:10]")

# and split it up
squad = squad.train_test_split(test_size=0.2, seed=42)

def test_llm_cascade(llmc: Any, dataset: List[dict]):
    """
    Invokes the LLM cascade against a collection of context/question pairs
    and returns the final answer and a cost analysis.
    """
    for sample in dataset:
        response = llmc.invoke({'context': sample['context'], 'question': sample['question']})
        print(json.dumps(response, indent=4))
        cost, best_cost = compute_cost(response, models)
        cost_delta = (cost - best_cost) * 100 / best_cost
        print(f"Cost: ${cost} ({cost_delta:+.2f}%)\n")

##############
# Evaluation #
##############

##### LLM Cascade configuration #####

# Mistral + Anthropic Claude 3 models
models = [
    "mistral.mistral-7b-instruct-v0:2",
    "mistral.mixtral-8x7b-instruct-v0:1",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0"
]

# Higher thresholds means we don't trust the model, while
# the last threshold set to 0 means we accept everything
thresholds = [0.9, 0.8, 0.7, 0.0]

##### LangChain #####

llmc_lc = llm_cascade(models, thresholds)

# Test
test_llm_cascade(llmc_lc, squad['train'])

##### LangGraph #####

llmc_lg = llm_cascade_lg(models, thresholds)

# ASCII
llmc_lg.get_graph().print_ascii()

# Mermaid
# https://mermaid.js.org/
print(llmc_lg.get_graph().draw_mermaid())

# Test
test_llm_cascade(llmc_lg, squad['train'])
