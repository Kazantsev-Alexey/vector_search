# vector_search
Vector search  practical experiments and small projects

Start with a new env:

	uv venv --python=python3

	source .venv/bin/activate

	uv pip install -e ".[hw1]"


and go with all modules:

	python3 -m hw1_how_to_search.checks

	python3 -m hw1_how_to_search.benchmarks

	python3 -m hw1_how_to_search.imdb_glove_search

	python3 -m hw1_how_to_search.visualize

If there will be a problem on the step with imdb_glove_search related to terminated handshake, then it's possible to download the data and put zip files into "data" folder

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"


For hw2

uv venv --python=python3
source .venv/bin/activate

uv pip install -e ".[hw2]"

python3 -m hw2_ann_exploration.annoy_benchmark
python3 -m hw2_ann_exploration.ivfpq_benchmark
python3 -m hw2_ann_exploration.hnsw_benchmark