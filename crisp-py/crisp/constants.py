# BEIR instructions and other constants

BEIR_INSTRUCTIONS = {
    "arguana": "Given a claim, find documents that refute the claim.",
    "climate-fever": (
        "Given a claim about climate change, retrieve documents that "
        "support or refute the claim."
    ),
    "dbpedia-entity": (
        "Given a query, retrieve relevant entity descriptions from " "DBPedia."
    ),
    "fever": ("Given a claim, retrieve documents that support or refute the " "claim."),
    "fiqa": (
        "Given a financial question, retrieve user replies that best "
        "answer the question."
    ),
    "hotpotqa": (
        "Given a multi-hop question, retrieve documents that can help "
        "answer the question."
    ),
    "msmarco": (
        "Given a web search query, retrieve relevant passages that answer " "the query."
    ),
    "nfcorpus": (
        "Given a question, retrieve relevant documents that best answer "
        "the question."
    ),
    "nq": (
        "Given a question, retrieve Wikipedia passages that answer the " "question."
    ),
    "quora": (
        "Given a question, retrieve questions that are semantically "
        "equivalent to the given question."
    ),
    "scidocs": (
        "Given a scientific paper title, retrieve paper abstracts that are "
        "cited by the given paper."
    ),
    "scifact": (
        "Given a scientific claim, retrieve documents that support or "
        "refute the claim."
    ),
    "trec-covid": ("Given a query, retrieve documents that answer the query."),
    "webis-touche2020": (
        "Given a question, retrieve detailed and persuasive arguments that "
        "answer the question."
    ),
    "cqadupstack": (
        "Given a question, retrieve detailed question descriptions from "
        "Stackexchange that are duplicates to the given question."
    ),
}

TITLE_DATASETS = ["nfcorpus", "fiqa", "dbpedia-entity", "scidocs"]

# defaults
DEFAULT_EMBEDDING_DIM = 2048
DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_TEMPERATURE = 0.05

# k-means stuff
DEFAULT_KMEANS_ITERATIONS = 20
DEFAULT_KMEANS_MAX_ITERATIONS = 50
KMEANS_CONVERGENCE_THRESHOLD = 0.01

# training
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_STEPS = 1000
DEFAULT_GRADIENT_CLIP_VALUE = 1.0

# eval
NDCG_K_VALUES = [1, 3, 5, 10, 100]
DEFAULT_NDCG_K = 10
