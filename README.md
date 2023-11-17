Package for computing AnthroScore

# Setup
`git clone https://github.com/myracheng/anthroscore.git`  
`cd anthroscore`  
`pip install .`  
# Example Usage
To obtain AnthroScores for the terms "model" and "system" in 
abstracts from examples/acl_50.csv (a subset of ACL Anthology papers)

    python get_anthroscore.py --input_file example/acl_50.csv \
        --text_column_name abstract --entities system model \
        --output_file example/results.csv --text_id_name acl_id
