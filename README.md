# How to execute the scripts (e.g., command line parameters)

1. Create a ```venv``` and install the required libraries from ```requirements.txt```
2. Locate the PROXY_BASE_URL link in a file (you need to create) called ```.env``` so that:
```PROXY_BASE_URL=""```
3. Run ```python3 scripts/indexer.py``` to generate the text embeddings from the downloaded papers in ```papers/``` folder.
4. Run ```python3 scripts/query.py``` to create Agents and provide questions to the RAG-generated literature review assistant.

# File names

1. papers/2407.01502v1.pdf
2. papers/2510.25445v1.pdf
3. papers/NIPS-2017-attention-is-all-you-need-Paper.pdf

# Questions

Questions can be seen in ```questions.txt```. The list of questions goes as follows:

1.
2.
3.
4.
5.
6.
7.
8.
9.

# Answers

For each of the placed questions, the answers provided by the activated Agent go as follows:

1.

2.

3.

4.

5.

6.

7.

8.

9.


# Any additional comments (optional)

Disclaimer: Officially report that this project is part of the AI Engineering course offered at the University of Oulu (Finland), and that the course allows the use of coding agents. For this project, GitHub Copilot has been used.
