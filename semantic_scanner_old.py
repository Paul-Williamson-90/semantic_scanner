from typing import List, Dict, Tuple, Union, Optional
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from embeddings import Embeddings
import os
from sklearn.neighbors import KNeighborsClassifier
import json
import numpy as np
from prompts import extract_text, verify_class
from const import *
import warnings
from scipy.spatial.distance import cosine, euclidean
import pandas as pd

nltk.download('stopwords')

class SemanticScanner:

    def __init__(self,
                 embeddings: Embeddings,
                 model_data_path: str = "./data/model.json",
                 classes_data_path:str = "./data/classes.json",
                 distance: str = DISTANCE,
                 n_neighbors: int = 5,):
        """
        Args:
        - embeddings (object):  Embeddings object containing embed method for generating embeddings
        - model_data_path (str): Path to KNN model
        - distance (str): Distance metric for KNN model
        - n_neighbors (int): Number of neighbors for KNN model
        """
        self.n_neighbors = n_neighbors
        self.embeddings = embeddings
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.model_data_path = model_data_path
        self.classes_data_path = classes_data_path
        self.knn_model = self._load_knn_model(model_data_path, classes_data_path, distance)
        self.distance = distance

    def _load_knn_model(self, model_data_path: str, classes_data_path: str, distance: str):
        """
        Loads KNN model from path.
        Args:
        - knn_model_path (str): Path to KNN model
        """
        knn_model = KNeighborsClassifier(n_jobs=-1,
                                        n_neighbors=self.n_neighbors,
                                        metric=distance)
        if os.path.exists(model_data_path):
            knn_data = self._load_model_data(model_data_path)
            y_names = [record['label'] for _, record in knn_data.items()]
            self.label_map = self._load_classes(classes_data_path)
            self._label_map_dict = dict()
            for e,l in enumerate(self.label_map.keys()):
                self.label_map[l]["label_int"] = e
                self._label_map_dict[e] = l
            y = np.array([self.label_map[label]["label_int"] for label in y_names])
            x = np.array([record['embedding'] for _, record in knn_data.items()])
            knn_model.fit(x, y)
        else:
            warnings.warn("No samples found, please add samples to the KNN model before classifying or retrieving.")
        return knn_model
    
    def _load_classes(self, classes_data_path: str):
        """
        Loads classes from path.
        Args:
        - classes_data_path (str): Path to classes data
        """
        if os.path.exists(classes_data_path):
            with open(classes_data_path, "r") as f:
                classes_data = json.load(f)
        else:
            warnings.warn("No classes found, please add classes to the KNN model before classifying or retrieving.")
        return classes_data

    def _load_model_data(self, model_data_path: str):
        """
        Loads model data from path.
        Args:
        - model_data_path (str): Path to model data
        """
        if os.path.exists(model_data_path):
            with open(model_data_path, "r") as f:
                knn_data = json.load(f)
        else:
            warnings.warn("No samples found, please add samples to the KNN model before classifying or retrieving.")
        return knn_data

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
        
    def _embed_chunks(self, chunks:List[str]):
        """
        Embeds a list of text chunks.
        Args:
        - chunks (list): List of text chunks
        """
        return self.embeddings.embed(chunks)
    
    def _sparse_embeddings(self, text:str)->Dict[str, List[int]]:
        """
        Generates sparse embeddings for a given text.
        Args:
        - text (str): Text to generate sparse embeddings for
        """
        words = text.split()
        words = set([self.stemmer.stem(word) for word in words if word not in self.stop_words])

        sparse_emb = {word:[1] for word in words}
        return sparse_emb

    def _classify(self, embeddings:np.ndarray)->np.ndarray:
        """
        Classifies a text sample.
        Args:
        - embeddings (np.ndarray): Embeddings for text sample(s)
        - n (int): Number of neighbors to retrieve
        """
        # change number of neighbors
        self.knn_model.n_neighbors = self.n_neighbors

        if len(embeddings.shape)==1:
            embeddings = np.expand_dims(embeddings, axis=0)

        # classify text
        classification = self.knn_model.predict_proba(embeddings)
        return classification
    
    def classify(self, text:str)->np.ndarray:
        """
        Classifies a text sample.
        Args:
        - embeddings (np.ndarray): Embeddings for text sample(s)
        - n (int): Number of neighbors to retrieve
        """
        embeddings = self._embed_chunks(text)

        # change number of neighbors
        self.knn_model.n_neighbors = self.n_neighbors

        if len(embeddings.shape)==1:
            embeddings = np.expand_dims(embeddings, axis=0)

        # classify text
        classification = self.knn_model.predict_proba(embeddings)
        return self._label_map_dict[np.argmax(classification)]
    
    def _retrieve(self, text:str, n:int=N_FEW_SHOTS)->List[Dict[str, Union[str, float]]]:
        """
        Retrieves similar samples from the KNN model.
        Args:
        - text (str): Text to retrieve similar samples for
        - n (int): Number of neighbors to retrieve
        """
        # embed text
        embedding = self._embed_chunks(text)
        embedding = np.expand_dims(embedding, axis=0)

        # change number of neighbors
        knn_n = self.knn_model.n_neighbors
        self.knn_model.n_neighbors = 100

        # retrieve similar samples
        distances, indices = self.knn_model.kneighbors(embedding)
        self.knn_model.n_neighbors = knn_n
        indices = indices[0]
        distances = distances[0]
        # load knn data
        knn_data = self._load_model_data(self.model_data_path)
        retrieved_samples = []
        for e,i in enumerate(indices):
            retrieved_samples.append({'text':list(knn_data.keys())[i],
                                      'label':list(knn_data.values())[i]['label'],
                                      'distance':distances[e]})
        # filter to top n per label
        retrieved_samples = sorted(retrieved_samples, key=lambda x: x['distance'])
        labels = list(set([record['label'] for record in retrieved_samples]))
        retrieved_samples = [[x for x in retrieved_samples if x["label"]==label][:n] for label in labels]
        return retrieved_samples
        
    def refit(self,
              distance: str = DISTANCE,
              n_neighbors: int = 5,):
        """
        Refits the KNN model.
        """
        self.knn_model = self._load_knn_model(self.model_data_path, 
                                              self.classes_data_path,
                                              distance if distance else self.knn_model.metric, 
                                              n_neighbors if n_neighbors else self.knn_model.n_neighbors)
        self.n_neighbors = n_neighbors
        

    def add_sample(self, text:str, label:str, label_metadata:dict=None, refit:bool=False, **kwargs):
        """
        Adds a new sample to the KNN model.
        Args:
        - text (str): Text to add to KNN model
        - label (str): Label for text
        - refit (bool): Whether to refit the KNN model after adding the new sample
        """
        # embed text
        embedding = self._embed_chunks(text)
        # create record
        record = {"text": text, "embedding": embedding.tolist(), "label": label, **kwargs}
        # add record to self.knn_data_path json file
        if os.path.exists(self.model_data_path):
            knn_data = self._load_model_data(self.model_data_path)
        else:
            knn_data = {}
        # load classes data
        if os.path.exists(self.classes_data_path):
            classes_data = self._load_classes(self.classes_data_path)
        else:
            classes_data = {}
        # add label to classes data
        if label not in classes_data.keys():
            if label_metadata:
                # check req keys in label_metadata.keys()
                if not all([key in label_metadata.keys() for key in ["extract_text", "verify_class"]]):
                    raise ValueError("Label metadata must contain the following keys: {}".format(["extract_text", "verify_class"]))
                classes_data[label] = label_metadata
                # save classes data
                with open(self.classes_data_path, "w") as f:
                    json.dump(classes_data, f)
            else:
                raise ValueError("Label metadata not provided. Please provide label metadata when adding new labels.")
        knn_data[text] = record
        with open(self.model_data_path, "w") as f:
            json.dump(knn_data, f)
        # refit knn model
        if refit:
            self.knn_model = self._load_knn_model(self.model_data_path, self.classes_data_path, self.knn_model.metric, self.knn_model.n_neighbors)
        else:
            warnings.warn("Sample added, but KNN model not refit. Please refit the KNN model before classifying or retrieving.")

    def add_samples(self, samples:pd.DataFrame, refit:bool=False):
        """
        Adds a new sample to the KNN model.
        Args:
        - samples (pd.DataFrame): Samples to add to KNN model
        - refit (bool): Whether to refit the KNN model after adding the new sample
        """
        samples["embedding"] = samples["text"].apply(lambda x: self._embed_chunks(x))

        # create record
        for i, row in samples.iterrows():
            record = {"text": row['text'], "embedding": row['embedding'], "label": row['label'], 
                      }
            label_metadata = {k:v for k,v in row.items() if k not in ["text", "embedding", "label"]}
            label = row['label']
            # add record to self.knn_data_path json file
            if os.path.exists(self.model_data_path):
                knn_data = self._load_model_data(self.model_data_path)
            else:
                knn_data = {}
            # load classes data
            if os.path.exists(self.classes_data_path):
                classes_data = self._load_classes(self.classes_data_path)
            else:
                classes_data = {}
            # add label to classes data
            if label not in classes_data.keys():
                if label_metadata:
                    # check req keys in label_metadata.keys()
                    if not all([key in label_metadata.keys() for key in ["extract_text", "verify_class"]]):
                        raise ValueError("Label metadata must contain the following keys: {}".format(["extract_text", "verify_class"]))
                    classes_data[label] = label_metadata
                    # save classes data
                    with open(self.classes_data_path, "w") as f:
                        json.dump(classes_data, f)
                else:
                    raise ValueError("Label metadata not provided. Please provide label metadata when adding new labels.")
            knn_data[row['text']] = record
            with open(self.model_data_path, "w") as f:
                json.dump(knn_data, f)

        if refit:
            self.knn_model = self._load_knn_model(self.model_data_path, self.classes_data_path, self.knn_model.metric, self.knn_model.n_neighbors)
        else:
            warnings.warn("Sample added, but KNN model not refit. Please refit the KNN model before classifying or retrieving.")

    def _distances(self, embeddings:np.ndarray)->np.ndarray:
        """
        Calculates distances between embeddings.
        Args:
        - embeddings (np.ndarray): Embeddings for text sample(s)
        """
        knn_data = self._load_model_data(self.model_data_path)
        # get average embedding per class
        
        # get unique labels
        labels = list(set([record['label'] for _, record in knn_data.items()]))
        # get average embedding per label
        avg_embeddings = {}
        for label in labels:
            avg_embeddings[label] = np.mean([record['embedding'] for _, record in knn_data.items() if record['label'] == label], axis=0)

        # calculate distances
        distances = []
        metric = cosine if self.distance=="cosine" else euclidean
        if len(embeddings.shape)==1:
            distances.append([metric(embeddings, avg_embeddings[label]) for label in labels])
        else:
            for i in embeddings.shape[0]:
                distances.append([metric(embeddings[i,:], avg_embeddings[label]) for label in labels])
        
        return np.array(distances)

    def _prepare_scan(self, text:str, splitter:str=SPLITTER)->Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
        """
        Prepares a document of text for scanning.
        Args:
        - text (str): Text to scan
        """
        # chunk text
        if splitter == "window":
            chunks = self._chunk_window(text)
        elif splitter == "chars":
            chunks = self._chunk_character_split(text)
        # embed chunks
        embeddings = self._embed_chunks(chunks)
        # classify chunks
        classifications = self._classify(embeddings)
        # class distances
        distances = self._distances(embeddings)
 
        return chunks, embeddings, classifications, distances

    def scan(self, text:str, splitter:str=SPLITTER)->dict:
        """
        Scans a document of text for excerpts that match a set of classes.
        
        Args:
        - text (str): Text to scan
        """
        # prepare text for scanning
        chunks, embeddings, classifications, distances = self._prepare_scan(text, splitter)

        return chunks, embeddings, classifications, distances

        # TODO: threshold check on distance to each class (maybe)
    
        # TODO: for each chunk, verify using LLM if belongs to class (need a filtered retrieve function)
    
        # TODO: for verified, extract relevant text




# TODO: Verify classification via LLM

# TODO: Extract relevant text from new sample
    
# TODO: Add the lexical (sparse) classifications