# How to execute the scripts (e.g., command line parameters)

1. Create a ```venv``` and install the required libraries from ```requirements.txt```
2. Locate the PROXY_BASE_URL link in a file (you need to create) called ```.env``` so that:
```PROXY_BASE_URL=""```
3. Run ```python3 scripts/indexer.py``` to generate the text embeddings from the downloaded papers (in PDF format) in ```papers/``` folder.
4. Run ```python3 scripts/query.py``` to create Agents and activate the chat in order to provide questions to the RAG-generated literature review assistant.
5. Provide questions into the terminal, or type "exit" to quit the program.

# File names

1. papers/2407.01502v1.pdf
2. papers/2510.25445v1.pdf
3. papers/NIPS-2017-attention-is-all-you-need-Paper.pdf

# Questions

Questions can be seen in ```questions.txt```. The list of questions goes as follows:

1. What specific design choices (e.g., multi-head attention, positional encodings) are most critical for replacing recurrence without losing sequence-order information?
2. Which part of the experimental setup (training cost, model size, beam search choices, data preprocessing) most strongly influences those reported gains, and how sensitive are the results to those choices?
3. What concrete failure modes or scaling bottlenecks of full self-attention motivated this, and how might those limitations affect applicability beyond MT?
4. How does the evaluation ensure a fair comparison across agents with different numbers of calls, different temperatures, and different underlying models?
5. What does the Pareto frontier analysis reveal that single-metric reporting (accuracy-only) would hide?
6. What threats to validity remain (e.g., benchmark choice, sensitivity to API pricing/time, generalization to other tasks), and which ones most affect the main claims?
7. What were the inclusion/exclusion criteria and coding decisions that most influenced how studies were labeled “symbolic,” “neural,” or “hybrid”?
8. What evidence supports this mapping, and where are the borderline cases that challenge the categorization?
9. What key areas are underexplored (especially for symbolic and hybrid systems), and how might that bias the survey’s conclusions about “the most viable path forward” being hybrid?

# Answers

For each of the placed questions, the answers provided by the activated Agent go as follows:

1. **ANSWER** 

Agent: Methods Analyst

The most critical design choices for replacing recurrence without losing sequence-order information in the Transformer model include:

1. **Multi-Head Attention**: Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. This mechanism is crucial as it replaces the recurrent structure by enabling the model to look at all positions of the sequence simultaneously, capturing dependencies regardless of their distance in the sequence. Specifically, the Transformer uses multiple parallel attention layers ("heads") that operate on different linear projections of the input to capture various aspects of the sequence (NIPS-2017-attention-is-all-you-need-Paper.pdf, p. 5).

2. **Positional Encodings**: Because the Transformer does not have recurrence or convolution, it lacks an inherent notion of sequence order. To address this, positional encodings are added to the input embeddings at the bottoms of the encoder and decoder stacks. These encodings inject information about the relative or absolute position of tokens in the sequence. The original paper uses sinusoidal positional encodings, which enable the model to generalize to longer sequences, but learned positional embeddings were also shown to produce nearly identical results (NIPS-2017-attention-is-all-you-need-Paper.pdf, p. 2, 9).

3. **Masking in the Decoder Attention**: To preserve the auto-regressive property during decoding (generating the next token based only on previous ones), the decoder self-attention is masked so that positions cannot attend to subsequent positions. This masking ensures that predictions for each position can depend only on the known outputs at earlier positions in the sequence (NIPS-2017-attention-is-all-you-need-Paper.pdf, p. 3).

In sum, the critical architectural components that allow for replacing recurrence while preserving sequence order information are multi-head self-attention, positional encodings, and appropriate masking in the decoder self-attention layers (NIPS-2017-attention-is-all-you-need-Paper.pdf, pp. 2-5, 9).


2. **ANSWER**

Agent: Methods Analyst

The reported gains in the experimental results are influenced by several factors in the setup, with specific details found in the literature:

1. **Beam Search Choices**: In the Transformer model experiment, a beam size of 4 and a length penalty α = 0.6 were selected after experimentation on the development set (NIPS-2017-attention-is-all-you-need-Paper.pdf, p. 8). This suggests that beam search parameters were tuned to optimize results, indicating sensitivity to these choices.

2. **Training Cost and Checkpoint Averaging**: The base model used averaging of the last 5 checkpoints, while the big models averaged the last 20 checkpoints (NIPS-2017-attention-is-all-you-need-Paper.pdf, p. 8). The training cost was estimated by floating point operations factoring in training time and hardware, which impacts reproducibility and resource use but indirectly influences results through model quality.

3. **Model Size**: The experimentation included variations of model size (base vs. big models) to evaluate the importance of different components (NIPS-2017-attention-is-all-you-need-Paper.pdf, p. 8). Larger models (big) presumably provided stronger gains but at increased computational cost.

4. **Data Preprocessing**: Data preprocessing specifics were not detailed in the retrieved excerpts for those experiments, so their influence and sensitivity on performance gains are unclear from the available evidence.

5. **Optimization over Prompting and Few-shot Examples**: Another work discussed joint optimization of parameters such as the number of few-shot examples, temperature settings, and formatting instructions to balance accuracy and inference cost (2407.01502v1.pdf, pp. 23-26). This suggests sensitivity to prompt design and example selection that significantly influences observed performance.

In summary, beam search parameters and model size are explicitly highlighted as key factors impacting reported gains, with beam search tuned on dev sets and model size systematically varied. Training cost influences the model quality via checkpoint averaging strategies. Prompt design and few-shot example optimization also play a critical role in a different context. The sensitivity of results to these factors is acknowledged through systematic tuning and optimization reported in the studies. However, data preprocessing impact is not clearly discussed in these excerpts.

Therefore, beam search choices and model size are among the most strongly influential parts of the experimental setup on the gains, with sensitivity demonstrated through parameter tuning and optimization (NIPS-2017-attention-is-all-you-need-Paper.pdf, p. 8; 2407.01502v1.pdf, pp. 23-26).


3. **ANSWER**

Agent: Skeptical Reviewer

The Transformer paper ("Attention is All You Need") highlights some scaling bottlenecks and failure modes of full self-attention that motivated the design:

1. Computational Complexity: The computation of full self-attention scales quadratically with the sequence length n because it considers all pairs of positions. This can become a bottleneck when sequences are very long. Although self-attention is faster than recurrent layers when n is smaller than the representation dimensionality d (which is typical for sentence-level translations), very long sequences pose challenges (NIPS-2017-attention-is-all-you-need, p. 6).

2. Motivation for Restricted Attention: To handle very long inputs and outputs efficiently (like images, audio, video), the authors planned to investigate local or restricted attention mechanisms that limit self-attention to neighborhoods rather than the entire sequence. This suggests that full self-attention may not scale well in modalities or tasks that involve larger input sizes than typical text sequences (NIPS-2017-attention-is-all-you-need, p. 9).

3. Applicability Beyond MT: The authors explicitly plan to extend Transformers to other problems beyond text, including input/output modalities different from text, but acknowledge the need to develop attention variants or modifications to handle large-scale data efficiently (NIPS-2017-attention-is-all-you-need, p. 9).

In summary, the primary limitation is the quadratic scaling of full self-attention with sequence length, which can limit applicability to tasks with very large inputs or outputs. To generalize beyond machine translation, especially to modalities like images or long sequences, adaptations such as local attention or restricted attention may be necessary to maintain computational tractability.

This analysis is based directly on the original Transformer paper.

4. **ANSWER**

Agent: Methods Analyst

The evaluation ensures a fair comparison across agents that might differ in the number of calls, temperatures, and underlying models primarily by focusing on factors that create a level playing field:

1. Cost-normalized evaluation: Rather than relying on proxies for cost such as parameter count or training compute (which can be misleading), the evaluation measures actual inference costs (e.g., token input/output counts and dollar costs). This allows recalculations based on current prices and helps compare models fairly across providers and time, despite differences in model architecture or scale (2407.01502v1.pdf, pp. 6-7).

2. Controlling compute and normalization: They normalize the amount of compute used to train models to isolate improvements due to architecture or data rather than raw compute differences. This helps compare model efficiency and accuracy on an equal footing (2407.01502v1.pdf, p. 6).

3. Use of retries and temperature variation strategies: For handling stochasticity differences (e.g., models with various temperatures or calls), baseline methods include strategies like retrying zero-temperature calls multiple times or warming up temperature gradually, aiming to fairly evaluate the likelihood of success among differently randomized outputs (2407.01502v1.pdf, p. 4).

4. Reporting accuracy versus inference cost/time curves: Results include accuracy plotted against inference cost or time, showing performance across different resource expenditures. This presents a Pareto curve that naturally accommodates differences in calls, temperatures, and model complexity by reflecting trade-offs explicitly (2407.01502v1.pdf, p. 20).

Together, these evaluation design choices help ensure a fair and transparent comparison accounting for practical differences across models, calls, and temperature settings.

References: (2407.01502v1.pdf, pp. 4, 6-7, 20)

5. **ANSWER**

Agent: Skeptical Reviewer

The Pareto frontier analysis reveals trade-offs between accuracy and cost (e.g., inference cost) that single-metric reporting (accuracy-only) would completely hide. While accuracy-only reporting shows only the performance of models, it ignores other critical dimensions like computational expense or monetary cost. Pareto frontiers expose the set of non-dominated models that optimize for both cost and accuracy jointly, highlighting models that provide the best trade-offs between these metrics. This is important because some models may have only marginal accuracy gains but order-of-magnitude higher costs, which single-metric reporting would not reveal. For example, complex AI agents may achieve similar or slightly better accuracy than simpler agents but at vastly higher inference costs, which are shown clearly by the Pareto frontier but remain invisible when only accuracy is considered (2407.01502v1.pdf, pp. 6, 22-23).

Thus, single-metric accuracy reporting risks obscuring important cost-performance trade-offs critical for real-world application and model selection decisions. The Pareto frontier makes clear that there is no free lunch: improvements in one metric often come at a cost in another, and choosing a model requires balancing these factors rather than blindly chasing accuracy alone. This joint consideration is crucial for understanding model value comprehensively (2407.01502v1.pdf, pp. 6, 22-23).

6. **ANSWER**

Agent: Skeptical Reviewer

Several key threats to validity remain that affect the main claims related to benchmark choice, sensitivity to API pricing/time, and generalization:

1. Benchmark Choice and Generalizability:
- Benchmarks used for evaluating agents can be misleading if their design does not fit the agent's intended use case and generality level. For instance, distribution-specific benchmarks only test performance on a narrow task setting and do not account for distribution shifts or domain adaptation, limiting generalization (2407.01502v1, p. 4, 8).
- Different levels of generality require different types of hold-out samples to avoid shortcuts and overfitting to benchmarks. Without proper hold-outs, developers may unintentionally take shortcuts that inflate apparent performance (2407.01502v1, p. 2).
- Lack of consistent, standardized benchmarks across domains and tasks limits the ability to generalize claims from benchmark results to broader or different task settings (2407.01502v1, p. 8).

2. Sensitivity to API Pricing and Time:
- Evaluation of cost is complicated by variability in API pricing models: costs can differ between providers for the same model, prices can change overnight, and charges may depend on developer-decided factors like whether bulk calls are discounted (2407.01502v1, p. 7).
- Proper evaluation of cost-effectiveness requires including input/output token counts alongside dollar costs so that future re-evaluations can adjust for pricing changes. However, this is seldom consistently done, threatening the validity of reported cost-effectiveness (2407.01502v1, p. 7).
- The time and cost of API calls can influence agent evaluation but are often not accounted for sufficiently, leading to overoptimistic conclusions on agent efficiency (inference based on 2407.01502v1, p. 7).

3. Evaluation Design and Reproducibility:
- There are pervasive reproducibility issues in agent evaluations (e.g., with WebArena and HumanEval benchmarks), including errors that inflate accuracy estimates and lead to over-optimistic claims about agent capabilities (2407.01502v1, p. 2).
- Evaluations mix agents with different evaluation protocols, such as using modified benchmarks or adding/removing example test cases, which breaks standardization and comparability of results (2407.01502v1, p. 11).

Among these, the threats from benchmark choice and lack of standardization most critically affect the main claims about agent performance and generality since flawed benchmarks can cause misleading results. Sensitivity to API pricing/time mainly threatens cost evaluation claims. Reproducibility shortcomings undermine confidence in reported accuracy. These remain key challenges that must be addressed for more valid conclusions (2407.01502v1, pp. 2, 4, 7, 8, 11).

In summary, the main threats to validity and their impacts are:
- Benchmark design flaws and overfitting: threaten generalization claims.
- Pricing variability and time cost: threaten cost-effectiveness claims.
- Lack of reproducibility and standardization: threaten accuracy and comparability claims.

7. **ANSWER**

Agent: Methods Analyst

The labeling of studies as "symbolic," "neural," or "hybrid" was primarily influenced by the fundamental operational mechanics and architectural characteristics of the systems under study. Symbolic systems were defined by their use of explicit logic, algorithmic reasoning, and deterministic or probabilistic models. They excel in domains requiring reliability, verifiability, and safety, with a reliance on discrete symbolic knowledge bases or planners (2510.25445v1.pdf, pp. 5-9, 22).

Neural systems were characterized by stochastic orchestration, statistical learning from data, and emergent reasoning capabilities such as pattern recognition and adaptation to unstructured data. They are associated with neural networks and deep reinforcement learning methods (2510.25445v1.pdf, pp. 9, 22).

Hybrid systems represented a strategic integration of the two paradigms, embedding symbolic reasoning modules within neural frameworks to mitigate shortcomings of neural models (e.g., hallucination, lack of verifiability) while preserving their adaptive strengths. The classification thus depended on whether a study's approach and mechanisms aligned predominantly with symbolic logic-based architectures, purely neural network-based architectures, or a combination thereof (2510.25445v1.pdf, p. 22, 33).

The reviewed literature suggests that these distinctions were grounded in the inherent architectural designs and operational paradigms of the systems under study rather than arbitrary criteria, with an emphasis on whether the system's core mechanisms were algorithmic versus stochastic. However, explicit inclusion/exclusion criteria or detailed coding decisions beyond these conceptual divides were not clearly specified in the retrieved text.

Hence, the most influential factors for labeling were the core architectural lineage—symbolic (logic and algorithms), neural (statistical learning and networks), or hybrid (integration of both)—reflecting the underlying mechanism of the AI system’s reasoning and decision-making processes (2510.25445v1.pdf, pp. 5-9, 22, 33).

8. **ANSWER**

Agent: Skeptical Reviewer

The evidence supporting the mapping of paradigms in the reviewed literature includes the creation of a visual taxonomy categorizing core dimensions through symbolic and neural mechanisms, as well as a structured literature map analyzing 90 studies classified by primary architectural paradigm (2510.25445v1, p. 24). This dual-lineage framework was derived from a systematic review and thematic synthesis, combining current research with foundational symbolic context (p. 16, 24).

However, significant challenges to the categorization exist, particularly regarding borderline cases. The authors acknowledge the "paradigm classification challenge," where assigning hybrid or transitional architectures to a single paradigm sometimes requires necessary simplification. This reflects a limitation that some architectures inherently blend features of both symbolic and neural paradigms, making strict classification difficult (p. 18).

Additional threats to validity include methodological heterogeneity across studies, with varied evaluation metrics limiting direct benchmarking. Metrics were also sometimes incomplete or inferred from secondary sources, which may affect the robustness of the paradigm assignments (p. 18).

Domain-specific applications illustrate the paradigm distinctions but also the complementarity and potential overlap. For example, in healthcare, symbolic systems are used for auditable, predictable tasks, while neural frameworks handle more flexible tasks but may be embedded within deterministic pipelines to ensure reliability, indicating some domain-driven hybrid usage that challenges neat categorization (p. 19).

In summary, while there is structured empirical and theoretical evidence supporting the mapping of agentic AI into symbolic and neural paradigms, borderline and hybrid cases pose classification challenges that sometimes necessitate simplification. The heterogeneity of methods and metrics across studies further complicates clear demarcation (2510.25445v1, pp. 16, 18, 19, 24, 36).

9. **ANSWER**

Agent: Skeptical Reviewer

Key underexplored areas for symbolic and hybrid systems include:

1. **Symbolic Systems' Limitations:**
   - Symbolic systems such as BDI and SOAR, while powerful in explicitly modeling internal states and logic-based processes, are brittle and difficult to scale to complex, real-world environments.
   - Their reliance on theorem provers and logic engines leads to scalability and performance constraints sensitive to hardware architecture.
   - Symbolic systems also face challenges such as "perverse instantiations" where rigid goals can be exploited to cause unintended harmful outcomes (2510.25445v1.pdf, pp. 8, 31, 34).

2. **Hybrid Systems and Integration Challenges:**
   - Hybrid or neuro-symbolic integration is seen as promising, aiming to couple neural networks' adaptability with symbolic engines' reliability.
   - However, effective orchestration of hybrid multi-agent ecosystems, where specialized neural and symbolic agents communicate and work together, remains a critical research frontier.
   - Formal methods for symbolic verifiability and statistical methods for neural alignment need concurrent development to govern hybrid systems' safety and reliability (2510.25445v1.pdf, pp. 33, 34, 36).

3. **Bias and Fairness:**
   - Symbolic systems' bias arises from explicitly coded rules, which can be identifiable but hard to eliminate.
   - Neural components introduce subtle, latent biases from training data amplified unpredictably.
   - Hybrid systems need sophisticated mechanisms to detect and mitigate biases from both symbolic and neural parts (2510.25445v1.pdf, p. 31).

The survey's conclusion that hybrid systems represent "the most viable path forward" might be biased by an emphasis on the theoretical advantages of integration without fully addressing these underexplored implementation, scalability, and safety challenges. The complexities and open questions in orchestrating hybrid ecosystems and the potential brittleness or failure modes of symbolic reasoning under real-world conditions might be underrepresented. This may lead to an overly optimistic view of hybrid systems before these significant gaps are resolved (2510.25445v1.pdf, pp. 7, 33-36).

In summary, critical gaps in scalability, integration, bias mitigation, and safety vigilance for symbolic and hybrid approaches could bias the roadmap advocating hybrid intelligence as the paramount direction at this stage. These issues signal the need for caution and emphasize ongoing research requirements before declaring hybrids the definitive path forward.

# Any additional comments (optional)

Disclaimer: Officially report that this project is part of the AI Engineering course offered at the University of Oulu (Finland), and that the course allows the use of coding agents. For this project, GitHub Copilot has been used.
