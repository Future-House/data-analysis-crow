# Data Analysis Crow: A Jupyter Notebook Agent

Data Analysis Crow is an AI agent framework designed to perform complex scientific data analysis tasks by iteratively working through Jupyter notebooks. This agent takes in datasets and prompts, then systematically explores, analyzes, and interprets the data to provide comprehensive answers and insights.

The agent was used to produce the trajectories for the [BixBench benchmark](https://github.com/Future-House/bixbench).

## Key Features

- Accepts datasets and natural language prompts
- Iteratively builds Jupyter notebooks to answer research questions
- Works with Python, R, and Bash code execution
- Specializes in bioinformatics analysis but adaptable to various domains
- Comes with a Docker image including most common bioinformatics packages

## Links

- [Installation](#installation)
- [Using the Agent](#using-the-agent)
- [Advanced Usage](#advanced-usage)
- [BixBench Benchmark](#bixbench-benchmark)

## Installation

```bash
# Clone the repository
git clone https://github.com/Future-House/data-analysis-crow.git
cd data-analysis-crow

# Install dependencies
pip install -e .

# OPTIONAL:pull the docker image with bioinformatics packages
docker pull futurehouse/bixbench:aviary-notebook-env
```

## Prerequisites

### API Keys

We support all LLMs that are supported by [litellm](https://github.com/BerriAI/litellm). Create a `.env` file with the API keys for the LLMs you want to use. For example:

```
OPENAI_API_KEY = "your-openai-api-key"
ANTHROPIC_API_KEY = "your-anthropic-api-key"
```

## Using the Agent

The agent works by taking a dataset and a prompt, then iteratively building a Jupyter notebook to answer the question. Visit the [tutorial](https://github.com/Future-House/data-analysis-crow/blob/main/tutorial/example.ipynb) for a simple step-by-step guide on how to use the agent.

## Advanced Usage
For advanced evaluations, you can configure `server.yaml` and `runner.yaml` in the `src/scripts/bixbench_evaluation` directory and then run the evaluation script:
```bash
bash src/scripts/bixbench_evaluation/run.sh
```

This will:
1. Load the specified dataset
2. Process the prompt to understand the research question
3. Generate a Jupyter notebook with progressive analysis steps
4. Provide a final answer based on the analysis

Results are saved in the output directory specified in your configuration file.

Note that the dataset and environment configuration must be updated appropriately. For an example, see [dataset.py](https://github.com/Future-House/data-analysis-crow/blob/main/src/fhda/dataset.py) which includes the capsule dataset configuration used for the BixBench benchmark.

We also recommend visiting the BixBench repository where we share a full evaluation harness for the agent.

## Hosted Agent
Coming soon!

## BixBench Benchmark

Data Analysis Crow was used to produce the trajectories for the [BixBench benchmark](https://github.com/Future-House/bixbench), which evaluates AI agents on real-world bioinformatics tasks.

BixBench tests AI agents' ability to:

- Explore biological datasets
- Perform long, multi-step computational analyses
- Interpret nuanced results in the context of a research question

You can find the BixBench dataset in [Hugging Face](https://huggingface.co/datasets/futurehouse/BixBench), the paper [here](https://arxiv.org/abs/2503.00096), and the blog post [here](https://www.futurehouse.org/research-announcements/bixbench).

### Running BixBench Evaluations

To use this agent for BixBench evaluations, we recommend visiting the [BixBench repository](https://github.com/Future-House/bixbench) for more details.