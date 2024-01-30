"""
This script computes AnthroScore for a set of entities in a set of texts. It 
results in (1) an output file with AnthroScore for each text, and (2) an 
output file containing all sentences from the texts with AnthroScores.

EXAMPLE USAGE: 
To obtain AnthroScore for the terms "model" and "system" in the text 
    "I love this model. I hate this system.":

    python get_anthroscore.py --input_text "I love this model. I hate this system." --entities system model --output_sentence_file sentence_scores.csv

To obtain AnthroScores for the terms "model" and "system" in 
abstracts from examples/acl_50.csv (a subset of ACL Anthology papers):

    python get_anthroscore.py --input_file example/acl_50.csv \
        --text_column_name abstract --entities system model \

Optionally, you can specify the output file locations, and also the 
identifier for each text

    python get_anthroscore.py --input_file example/acl_50.csv \
        --text_column_name abstract --entities system model \
        --output_sentence_file example/result_sentences.csv \
            --output_file example/results.csv --text_id_name acl_id

You can also list the entities in a separate .txt file instead, 
specified by the argument --entity_filename

    python get_anthroscore.py --input_file example/acl_50.csv \
            --text_column_name abstract --entity_filename example/entities.txt

"""

import re
import pandas as pd
import argparse
import spacy
nlp = spacy.load("en_core_web_sm")
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
import numpy as np
import scipy
import gc
model = RobertaForMaskedLM.from_pretrained('roberta-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("BERT model loaded on %s"%device)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def get_prediction(sent):
    terms = ['he', 'she', 'her', 'him', 'He', 'She', 'Her', 'it', 'its', 'It', 'Its' ]
    target_inds = [tokenizer.get_vocab()[x] for x in terms]
    token_ids = tokenizer.encode(sent,return_tensors='pt').to(device)
    masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()
    try:
        masked_pos = [mask.item() for mask in masked_position][0]
    except IndexError:
        temp = tokenizer.encode(sent, return_tensors='pt').to(device)
        masked_position = (temp.squeeze() == tokenizer.mask_token_id).nonzero()
        try:
            if (int(masked_position[0] + 256)) > len(temp[0]):
                token_ids = torch.reshape(temp[0][-512:], (1, 512))
            else: 
                token_ids = torch.reshape(temp[0][masked_position[0] - 256:masked_position[0]+256], (1, 512))
        except IndexError:
            return np.empty((len(terms),))
        masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()
        masked_pos = [mask.item() for mask in masked_position ][0]
    
    with torch.no_grad():
        output = model(token_ids)

    last_hidden_state = output[0].squeeze()
    mask_hidden_state = last_hidden_state[masked_pos].cpu().numpy()

    probs = scipy.special.softmax(mask_hidden_state)
    scores = np.array([probs[i] for i in target_inds])
    return scores

def get_anthroscores(sentence_filename):
    terms = ['he', 'she', 'her', 'him', 'He', 'She', 'Her', 'it', 'its', 'It', 'Its' ]
    df = pd.read_csv(sentence_filename)
    final =np.empty((len(terms),))
    for i,x in enumerate(df.masked_sentence):
        if i>0 and i%100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            print("Calculating sentence %d"%i)
        newrow = get_prediction(x)
        final = np.vstack([final, newrow])
    
    human_scores = np.sum(final[1:,:7],axis=1)
    nonhuman_scores = np.sum(final[1:,7:],axis=1)
    df['anthroscore'] = np.log(human_scores) - np.log(nonhuman_scores)
    df.to_csv(sentence_filename)

def parse_sentences_from_file(input_filename, entities, text_column_name, id_column_name, output_filename):
    column_names = ['sentence','masked_sentence','text_id','POS','verb','original_term','original_noun']
    pattern_list = ['\\b%s\\b'%s for s in entities] # add boundaries

    if input_filename.endswith('csv'):
        df = pd.read_csv(input_filename).dropna(subset=text_column_name)
    else:
        df = pd.read_json(input_filename).dropna(subset=text_column_name)

    final = []
    for i, k in df.iterrows():
        if i>0 and i%1000==0:
            print("Parsing text #%d, %d sentences found"%(i,len(final)))
        text = k[text_column_name]
        if len(id_column_name)>0:
            text_id = k[id_column_name]
        else:
            text_id = i
        if text.strip():
            doc = nlp(text)
            for _parsed_sentence in doc.sents:
                for _noun_chunk in _parsed_sentence.noun_chunks:
                    if _noun_chunk.root.dep_ == 'nsubj' or _noun_chunk.root.dep_ == 'dobj':
                        for _pattern in pattern_list:
                            if re.findall(_pattern.lower(), _noun_chunk.text.lower()):
                                    _verb = _noun_chunk.root.head.lemma_.lower()
                                    target = str(_parsed_sentence).replace(str(_noun_chunk),'<mask>')
                                    final.append((str(_parsed_sentence), target, text_id, _noun_chunk.root.dep,str(_verb),_pattern.strip('\\b'),_noun_chunk.text.lower()))
    res = pd.DataFrame(final)
    res.columns =column_names
    res.to_csv(output_filename,index=False)
    print('%d sentences containing target entities found'%len(res))

def get_text_score(text,entities,output_filename=''):
    # Mask sentences
    pattern_list = ['\\b%s\\b'%s for s in entities] # add boundaries
    masked_sents = []
    if text.strip():
        doc = nlp(text)
        for _parsed_sentence in doc.sents:
            for _noun_chunk in _parsed_sentence.noun_chunks:
                if _noun_chunk.root.dep_ == 'nsubj' or _noun_chunk.root.dep_ == 'dobj':
                    for _pattern in pattern_list:
                        if re.findall(_pattern.lower(), _noun_chunk.text.lower()):
                                _verb = _noun_chunk.root.head.lemma_.lower()
                                target = str(_parsed_sentence).replace(str(_noun_chunk),'<mask>')
                                masked_sents.append(target)

    print('%d sentences containing target entities found'%len(masked_sents))
    if len(masked_sents)==0:
        print("Stopping calculation, no words found.")
        return np.nan
        
    # Get scores
    terms = ['he', 'she', 'her', 'him', 'He', 'She', 'Her',  'it', 'its', 'It', 'Its' ]
    final =np.empty((len(terms),))
    for i,x in enumerate(masked_sents):
        if i>0 and i%100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            print("Calculating sentence %d"%i)
        newrow = get_prediction(x)
        final = np.vstack([final, newrow])
    human_scores = np.sum(final[1:,:7],axis=1)
    nonhuman_scores = np.sum(final[1:,7:],axis=1)
    final_scores = np.log(human_scores) - np.log(nonhuman_scores)
    df = pd.DataFrame({'sentence':masked_sents,'anthroscore':final_scores})
    if len(output_filename)>0:
        df.to_csv(output_filename)
        print("Scores for individual sentences saved to %s"%output_filename)
    return np.mean(final_scores)



def compute_average_scores(input_file,output_sentence_file,output_file,text_id_name):
    if input_file.endswith('csv'):
        original_df = pd.read_csv(input_file)
    else:
        original_df = pd.read_json(input_file)
        
    final = []
    sentence_df = pd.read_csv(output_sentence_file)

    for i, k in original_df.iterrows():
        if len(text_id_name)>0:
            relevant_sents = sentence_df.loc[sentence_df.text_id==k[text_id_name]]
        else:
            relevant_sents = sentence_df.loc[sentence_df.text_id==i]

        if len(relevant_sents) > 0:
            final.append(np.mean(relevant_sents.anthroscore))
        else:
            final.append(float('nan'))

    original_df['anthroscore'] = final
    original_df.to_csv(output_file)

def main():
    parser = argparse.ArgumentParser(description="Script to compute AnthroScore for a given set of texts",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument("--input_text", type=str, help='Input text to compute AnthroScore on')
    group1.add_argument("--input_file", help="Input CSV or JSON file of text(s) to compute AnthroScore on")
    parser.add_argument("--text_column_name", help="Column of input CSV containing text(s) to compute AnthroScore on.")

    group2= parser.add_mutually_exclusive_group()
    group2.add_argument("--entities",nargs="+", type=str,help="Entities to compute AnthroScore for")
    group2.add_argument('--entity_filename',default='',help=".txt file of entities to compute AnthroScore for")
    
    parser.add_argument("--output_file", default='',help="Location to store output of AnthroScores for every text, optional")
    parser.add_argument("--output_sentence_file", default='',help="Location to store output of parsed sentences with AnthroScores, optional")
    parser.add_argument("--text_id_name",type=str,default='',help="ID for each text, optional -- otherwise defaults to the index in the dataframe")
    
        
    args = parser.parse_args()

    if len(args.entity_filename)>0:
        with open(args.entity_filename) as f:
            entities = [line.rstrip('\n') for line in f]
    else:
        entities = args.entities



    if args.input_text is not None:
        score = get_text_score(args.input_text, entities, args.output_sentence_file)

        print('Average AnthroScore in text: %.3f'%(score))
    else:
        input_file = args.input_file
        output_file = args.output_file
        if len(output_file) == 0:
            output_file = '%s_anthroscores.csv'%(input_file.split('.')[0])
        assert ((input_file[-4:]=='.csv') or (input_file[-5:]=='.json'))
        assert output_file[-4:]=='.csv'

        output_sentence_file = args.output_sentence_file

        if len(output_sentence_file) == 0:
            output_sentence_file = '%s_sentence_scores.csv'%(input_file.split('.')[0])
        
        text_column_name = args.text_column_name
        assert text_column_name is not None

            
        text_id_name = args.text_id_name

        parse_sentences_from_file(input_file,entities,text_column_name, text_id_name, output_sentence_file)
        
        get_anthroscores(output_sentence_file)

        compute_average_scores(input_file,output_sentence_file,output_file,text_id_name)

        print('AnthroScores for each sentence saved in %s'%(output_sentence_file))
        print('AnthroScores for text sentence saved in %s'%(output_file))


if __name__ == '__main__':
    main()

