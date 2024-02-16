from typing import List, Dict, Tuple, Union, Optional
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import json
import numpy as np
from prompts import extract_text, verify_class
from const import *
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
from datasets import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd

nltk.download('stopwords')

# TODO: Extract relevant text from new sample
    
# TODO: Add the lexical (sparse) classifications

class SemanticScanner:

    def __init__(self,
                 condition_classifier_model_path: str = CONDITIONS_CLASSIFIER_MODEL_PATH,
                 importance_classifier_model_path: str = IMPORTANCE_CLASSIFIER_MODEL_PATH,
                 ):
        """
        Args:
        - embeddings (object):  Embeddings object containing embed method for generating embeddings
        - model_data_path (str): Path to KNN model
        - distance (str): Distance metric for KNN model
        - n_neighbors (int): Number of neighbors for KNN model
        """
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.models = {"conditions_classifier":self._load_classifier(model_path=condition_classifier_model_path),
                        "importance_classifier":self._load_classifier(model_path=importance_classifier_model_path)}

    def _load_classifier(self, model_path:str=MODEL_PATH)->None:
        """
        Loads the KNN model from the model_data_path.
        Args:
        - model_data_path (str): Path to KNN model
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        label2id = json.load(open(f"{model_path}"+r"\label2id.json"))
        id2label = {v:k for k,v in label2id.items()}
        model_config = json.load(open(f"{model_path}"+r"\model_config.json"))

        model_dict = {"tokenizer":tokenizer, "model":model, "id2label":id2label, "model_config":model_config}
        return model_dict


    def _chunk_window(self, text:str, w:int=W, s:int=S, words_or_chars:str=WORDS_OR_CHARS)->List[str]:
        """
        Text chunking method using a sliding window.
        Args:
        - text (str): Text to chunk
        - w (int): Size of chunks
        - s (int): Step size for sliding window
        """
        if words_or_chars == "chars":
            words = text.split()
            chunks = []
            i = 0
            while i < len(words):
                chunk = words[i] 
                j = i + 1
                while j < len(words) and len(chunk + ' ' + words[j]) < w:
                    chunk += ' ' + words[j]
                    j += 1
                if chunk:  
                    chunks.append(chunk)
                # increase i by number of len(words) that >= s
                i += max(1, j - i - s + 1)
            return chunks
        elif words_or_chars == "words":
            return [' '.join(text.split()[i:i+w]).strip() for i in range(0, len(text.split()), s)
                    if ' '.join(text.split()[i:i+w]).strip() != '']
        else:
            raise ValueError("words_or_chars must be either 'words' or 'chars'")
    
    def _chunk_character_split(self, text:str, chars:List[str]=SPLIT_CHARS, w:int=W, s:int=S, words_or_chars:str=WORDS_OR_CHARS)->List[str]:
        """
        Text chunking method using splitting on specific characters with optional size control.
        Args:
        - text (str): Text to chunk
        - chars (list): List of characters to split on
        - w (int): Size of chunks (optional)
        - s (int): Step size for sliding window (optional)
        """
        # split the text on any of the characters inside the chars list
        pattern = '[' + re.escape(''.join(chars)) + ']'
        chunks = [x.strip() for x in re.split(pattern, text)]
        
        # if w and s are provided, further chunk the text
        if w is not None and s is not None and words_or_chars is not None:
            chunks = [chunk for chunk in chunks if chunk != '']
            chunks = [self._chunk_window(chunk, w, s, words_or_chars) for chunk in chunks]
            chunks = [chunk for sublist in chunks for chunk in sublist]
            
        # remove empty chunks
        chunks = [chunk for chunk in chunks if chunk != '']
        return chunks
    
    def _preprocess_data(self, examples):
        text = examples["text"]
        encoding = self.models[self.model_type]['tokenizer'](text, padding="max_length", truncation=True, max_length=128)  
        return encoding
    
    def _predict(self, texts:List[str], model_type:str="conditions_classifier")->List[str]:
        """
        Predicts the class of a list of texts.
        Args:
        - texts (list): List of texts to predict
        """
        self.model_type = model_type
        self.models[model_type]['model'].eval()

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.models[model_type]['model'].to(device)

        dataset = Dataset.from_dict({"text": texts})
        dataset = dataset.map(self._preprocess_data, batched=True)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        loader = DataLoader(dataset, batch_size=self.models[model_type]['model_config']['batch_size'], shuffle=False)
        
        if self.models[model_type]['model_config']['classification_type'] == "multi-class":
            act_fn = torch.nn.Softmax(dim=1)
            def post_process(predictions):
                predictions = np.argmax(predictions, axis=1)
                predictions = [self.models[model_type]['id2label'][pred] for pred in predictions]
                return predictions
        elif self.models[model_type]['model_config']['classification_type'] in ["multi-label"]:
            act_fn = torch.nn.Sigmoid()
            def post_process(predictions):
                predictions = np.where(predictions>=self.models[model_type]['model_config']['threshold'], 1, 0)
                predictions = [self.models[model_type]['id2label'][pred] for pred in predictions]
                return predictions
        elif self.models[model_type]['model_config']['classification_type'] in ['binary']:
            act_fn = torch.nn.Sigmoid()
            def post_process(predictions):
                predictions = np.where(predictions>=self.models[model_type]['model_config']['threshold'], 1, 0)
                if len(predictions.shape) != 1:
                    predictions = predictions.reshape(-1)
                return predictions
        
        preds = []
        for batch in tqdm(loader):
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.models[model_type]['model'](**batch)
            logits = outputs.logits
            probs = act_fn(logits)
            preds.append(probs.to('cpu').detach().numpy())

        preds = np.concatenate(preds, axis=0)
        if self.models[model_type]['model_config']['classification_type'] == "multi-class":
            top_3_indices = np.argsort(preds, axis=1)[:,-3:]
            top_3_indices = np.array([[self.models[model_type]['id2label'][idx] for idx in indices] for indices in top_3_indices])
        else:
            top_3_indices = None
        preds = post_process(preds)

        self.models[model_type]['model'].to('cpu')
        return preds, top_3_indices
    
    def _verify_class(self, excerpt, classification):
        """
        Verifies the classification of an excerpt.
        Args:
        - excerpt (str): Excerpt to verify
        - classification (str): Classification of the excerpt
        """
        raise NotImplementedError("This method is not yet implemented.")
    
    def _load_document(self, doc:str)->str:
        """
        Loads a document from a file path or string.
        Args:
        - doc (str): Document to load
        """
        if os.path.exists(doc):
            with open(doc, 'r') as f:
                text = f.read()
        else:
            raise ValueError("doc must be a valid file path to a txt file.")
        return text
    
    def scan_document(self, doc:str, splitting_method:str=SPLITTER, w:int=W, s:int=S, words_or_chars:str=WORDS_OR_CHARS)->List[str]:
        """
        Scans a document for semantic information.
        Args:
        - doc (str): Document to scan
        """

        doc = self._load_document(doc)

        if splitting_method == "window":
            chunks = self._chunk_window(doc, w, s, words_or_chars)
        elif splitting_method == "chars":
            chunks = self._chunk_character_split(doc, w, s, words_or_chars)
        else:
            raise ValueError("splitting_method must be either 'window' or 'chars'")
        
        preds, _ = self._predict(chunks, model_type="importance_classifier")
 
        excerpts_frame = pd.DataFrame({"excerpt":chunks, "importance":preds})
        important = excerpts_frame[excerpts_frame["importance"] == 1].reset_index(drop=True)
        unimportant = excerpts_frame[excerpts_frame["importance"] == 0].reset_index(drop=True)

        preds, top_3_labels = self._predict(important['excerpt'].tolist(), model_type="conditions_classifier")
        important['classification'] = preds
        unimportant['classification'] = ["not_important"]*len(unimportant)

        important.loc[:,'top_1_label'] = top_3_labels[:,0]
        important.loc[:,'top_2_label'] = top_3_labels[:,1]
        important.loc[:,'top_3_label'] = top_3_labels[:,2]
        
        unimportant.loc[:,'top_1_label'] = ["not_important"]*len(unimportant)
        unimportant.loc[:,'top_2_label'] = ["not_important"]*len(unimportant)
        unimportant.loc[:,'top_3_label'] = ["not_important"]*len(unimportant)

        excerpts_frame = pd.concat([important, unimportant], axis=0).reset_index(drop=True)

        # TODO: Verify classification via LLM
        # excerpts_frame["verified"] = excerpts_frame.apply(lambda x: self._verify_class(x["excerpt"], x["classification"]), axis=1)

        return excerpts_frame

    
if __name__=='__main__':

    scanner = SemanticScanner(condition_classifier_model_path = CONDITIONS_CLASSIFIER_MODEL_PATH,
                            importance_classifier_model_path = IMPORTANCE_CLASSIFIER_MODEL_PATH)

    file = r"C:\Users\paulw\Documents\QuantSpark\semantic_scanner\CUAD_v1\full_contract_txt\2ThemartComInc_19990826_10-12G_EX-10.10_6700288_EX-10.10_Co-Branding Agreement_ Agency Agreement.txt"

    excerpt_classifications = scanner.scan_document(doc=file)

    excerpt_classifications.to_csv("test.csv", index=False)