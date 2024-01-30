

# AnthroScore
This repository contains code to compute AnthroScore. AnthroScore is introduced in the following paper, which is accepted to EACL 2024:
## AnthroScore: A Computational Linguistic Measure of Anthropomorphism
*Myra Cheng, Kristina Gligoric, Tiziano Piccardi, Dan Jurafsky* (Stanford University)
### Abstract:
Anthropomorphism, or the attribution of human-like characteristics to non-human entities, has shaped conversations about the impacts and possibilities of technology. We present ANTHROSCORE, an automatic metric of implicit anthropomorphism in language. We use a masked language model to quantify how non-human entities are implicitly framed as human by the surrounding context. We show that ANTHROSCORE corresponds with human judgments of anthropomorphism and dimensions of anthropomorphism described in social science literature. Motivated by concerns of misleading anthropomorphism in computer science discourse, we use ANTHROSCORE to analyze 15 years of research papers and downstream news articles. In research papers, we find that anthropomorphism has steadily increased over time, and that papers related to natural language processing (NLP) and language models have the most anthropomorphism. Within NLP papers, temporal increases in anthropomorphism are correlated with key neural advancements. Building upon concerns of scientific misinformation in mass media, we identify higher levels of anthropomorphism in news headlines compared to the research papers they cite. Since ANTHROSCORE is lexicon-free, it can be directly applied to a wide range of text sources.



# Setup
`git clone https://github.com/myracheng/anthroscore.git`  
`cd anthroscore-eacl`  
`pip install .`  
Install the spaCy model:
`python -m spacy download en_core_web_sm`
(The specific model used is https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl#sha256=86cc141f63942d4b2c5fcee06630fd6f904788d2f0ab005cce45aadb8fb73889)
# Example Usage
To obtain AnthroScores for the terms "model" and "system" in 
abstracts from examples/acl_50.csv (a subset of ACL Anthology papers)

    python get_anthroscore.py --input_file example/acl_50.csv \
        --text_column_name abstract --entities system model \
        --output_file example/results.csv --text_id_name acl_id
