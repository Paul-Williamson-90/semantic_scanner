from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction, EarlyStoppingCallback
import torch
import pandas as pd
from datetime import datetime
import os
import json

class BertClassifierTrain:

    def __init__(self, 
                 train_location:str,
                 val_split:float=0.2,
                 test_location:str=None,
                 model_name:str="bert-base-uncased",
                 save_model_name:str=None,
                 save_location:str=None,
                 logging_location:str='./logs',
                 max_length:int=128,
                 learning_rate:float=2e-5,
                 batch_size:int=8,
                 epochs:int=5,
                 early_stopping_patience:int=2,
                 weight_decay:float=0.01,
                 threshold:float=0.5,
                 _under_sample_ratio:float=1, # for testing number of samples relationship with performance
                 classification_type:str="multi-class"):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.model_name = model_name
        self.max_length = max_length
        self.save_location = save_location
        self.logging_location = logging_location
        self.early_stopping_patience = early_stopping_patience
        self.threshold = threshold
        self.save_model_name = save_model_name
        self.classification_type = classification_type
        self._under_sample_ratio = _under_sample_ratio
        self.prepare_dataset(train_location, test_location, val_split)
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(self.labels) if len(self.labels) > 2 else 1,
                                                           id2label=self.id2label if len(self.labels) > 2 else {k:v for k,v in self.id2label.items() if k==1},
                                                           label2id=self.label2id if len(self.labels) > 2 else {k:v for k,v in self.label2id.items() if v==1})
        self.execute()
        self.save_model()
        if test_location:
            self.evaluate()


    def prepare_dataset(self, train_location:str, test_location:str=None, val_split:float=0.2):
        train_df = pd.read_csv(train_location)

        if self._under_sample_ratio < 1:
            train_df, _ = train_test_split(train_df, test_size=1-self._under_sample_ratio, random_state=42, stratify=train_df['label'])

        if test_location:
            test_df = pd.read_csv(test_location)
        
        assert 'label' in train_df.columns, "Label column not found in train dataset"
        assert 'text' in train_df.columns, "Text column not found in train dataset"
        if test_location:
            assert 'label' in test_df.columns, "Label column not found in test dataset"
            assert 'text' in test_df.columns, "Text column not found in test dataset"

        self.labels = train_df['label'].unique()
        self.id2label = {idx:label for idx, label in enumerate(self.labels)}
        self.label2id = {label:idx for idx, label in enumerate(self.labels)}

        test_df = test_df[test_df['label'].isin(self.labels)]

        train_df, val_df = train_test_split(train_df, test_size=val_split, random_state=42)

        train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
        val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
        test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # turn the three datasets into a single dataset with train, val, and test splits
        if test_location:
            dataset = DatasetDict({
                'train': train_dataset.map(self.preprocess_data, batched=True),
                'val': val_dataset.map(self.preprocess_data, batched=True),
                'test': test_dataset.map(self.preprocess_data, batched=True)
            })
        else:
            dataset = DatasetDict({
                'train': train_dataset.map(self.preprocess_data, batched=True),
                'val': val_dataset.map(self.preprocess_data, batched=True)
            })

        self.dataset = dataset.remove_columns(train_df.columns)
        self.dataset.set_format("torch")

    def preprocess_data(self, examples:pd.DataFrame):
        # take a batch of texts
        text = examples["text"]
        # encode them
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length)
        # add labels
        examples["label_key"] = [self.label2id[label] for label in examples["label"]]
        # create numpy array of shape (batch_size, num_labels)
        if self.classification_type != "binary":
            labels_matrix = np.zeros((len(text), len(self.labels)))
            # fill numpy array
            for idx, label in enumerate(examples["label_key"]):
                labels_matrix[idx][label] = 1 
        else:
            labels_matrix = examples['label_key']

        encoding["labels"] = np.array([float(x) for x in labels_matrix]).reshape(-1, 1)

        return encoding
    
    def execute(self,):
        self.args = TrainingArguments(
                                self.model_name+'_finetuned' if not self.save_model_name else self.save_model_name,
                                evaluation_strategy = "epoch",
                                save_strategy = "epoch",
                                learning_rate=self.learning_rate,
                                per_device_train_batch_size=self.batch_size,
                                per_device_eval_batch_size=self.batch_size,
                                num_train_epochs=self.epochs,
                                weight_decay=self.weight_decay,
                                load_best_model_at_end=True,
                                metric_for_best_model='f1',
                                save_total_limit = 3,
                                use_cpu = False if torch.cuda.is_available() else True,
                                )
        
        trainer = Trainer(
                            self.model,
                            self.args,
                            train_dataset=self.dataset["train"],
                            eval_dataset=self.dataset["val"],
                            tokenizer=self.tokenizer,
                            compute_metrics=self.compute_metrics,
                            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.early_stopping_patience)]
                        )
        
        trainer.train()

        self.model = trainer.model
        
    def save_model(self,):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)
        if self.save_model_name:
            self.model_save_loc = self.save_location+'/'+self.save_model_name+dt_string
        else:
            self.model_save_loc = self.save_location+'/'+self.model_name+'_finetuned_'+dt_string
        self.model.save_pretrained(self.model_save_loc)
        self.tokenizer.save_pretrained(self.model_save_loc)
        if self.save_model_name:
            with open(self.save_location+'/'+self.save_model_name+dt_string+'/label2id.json', 'w') as fp:
                json.dump(self.label2id, fp)
        else:
            with open(self.save_location+'/'+self.model_name+'_finetuned_'+dt_string+'/id2label.json', 'w') as fp:
                json.dump(self.id2label, fp)

        model_config = {'model_name': self.model_name,
                        'save_model_name': self.save_model_name,
                        'save_location': self.save_location,
                        'logging_location': self.logging_location,
                        'max_length': self.max_length,
                        'learning_rate': self.learning_rate,
                        'batch_size': self.batch_size,
                        'epochs': self.epochs,
                        'early_stopping_patience': self.early_stopping_patience,
                        'weight_decay': self.weight_decay,
                        'threshold': self.threshold,
                        'classification_type': self.classification_type,
                        'training_samples': len(self.dataset["train"]),
                        'testing_samples': len(self.dataset["test"]),
                        'classes': len(self.labels),
                        'model_save_loc': self.model_save_loc
                        }
        
        if self.save_model_name:
            with open(self.save_location+'/'+self.save_model_name+dt_string+'/model_config.json', 'w') as fp:
                json.dump(model_config, fp)
        else:
            with open(self.save_location+'/'+self.model_name+'_finetuned_'+dt_string+'/model_config.json', 'w') as fp:
                json.dump(model_config, fp)

    def multi_label_metrics(self, predictions, labels):
        if self.classification_type == "multi-class":
            act_fn = torch.nn.Softmax(dim=1)
            probs = act_fn(torch.Tensor(predictions))
            y_pred = np.zeros(probs.shape)
            y_pred[np.arange(y_pred.shape[0]), np.argmax(probs, axis=1)] = 1
        elif self.classification_type == "multi-label":
            act_fn = torch.nn.Sigmoid()
            probs = act_fn(torch.Tensor(predictions))
            y_pred[np.where(probs >= self.threshold)] = 1
        elif self.classification_type == 'binary':
            act_fn = torch.nn.Sigmoid()
            probs = act_fn(torch.Tensor(predictions)).numpy()
            if probs.shape[1] > 1:
                probs = probs[:,1]
            y_pred = np.zeros(probs.shape)
            y_pred[np.where(probs >= self.threshold)] = 1
            y_true = labels
            f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred)
            roc_auc = roc_auc_score(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
            metrics = {'f1': f1_micro_average,
                       'roc_auc': roc_auc,
                       'accuracy': accuracy}
            return metrics
        else:
            raise ValueError("classification_type must be one of 'multi-class', 'multi-label', or 'binary'")
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
        accuracy = accuracy_score(y_true, y_pred)
        metrics = {'f1': f1_micro_average,
                    'roc_auc': roc_auc,
                    'accuracy': accuracy}
        return metrics

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, 
                tuple) else p.predictions
        result = self.multi_label_metrics(
            predictions=preds, 
            labels=p.label_ids)
        return result
    
    def evaluate(self,):
        trainer = Trainer(
                            self.model,
                            self.args,
                            eval_dataset=self.dataset["test"],
                            tokenizer=self.tokenizer,
                            compute_metrics=self.compute_metrics,
                        )
        ev = trainer.evaluate()
        ev['classification_type'] = self.classification_type
        ev["training_samples"] = len(self.dataset["train"])
        ev['testing_samples'] = len(self.dataset["test"])
        ev["classes"] = len(self.labels)
        ev['model_name'] = self.save_model_name if self.save_model_name else self.model_name
        ev['learning_rate'] = self.learning_rate
        ev['batch_size'] = self.batch_size
        ev['epochs'] = self.epochs
        ev['weight_decay'] = self.weight_decay
        ev['max_length'] = self.max_length
        ev['save_location'] = self.model_save_loc

        ev = pd.DataFrame({k: [v] for k, v in ev.items()})

        if not os.path.exists(self.logging_location):
            os.makedirs(self.logging_location)

        if self.save_model_name:
            model_path = self.save_model_name
        else:
            model_path = self.model_name+'_finetuned'

        if os.path.exists(self.logging_location+'/'+model_path+'_log.csv'):
            prev = pd.read_csv(self.logging_location+'/'+model_path+'_log.csv')
            prev = pd.concat([prev,ev], axis=0).reset_index(drop=True)
            prev.to_csv(self.logging_location+'/'+model_path+'_log.csv', index=False)
        else:
            ev.to_csv(self.logging_location+'/'+model_path+'_log.csv', index=False)


if __name__=='__main__':

    class_type = 'importance'
    classification_type = 'binary'

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    BertClassifierTrain(train_location='C:/Users/paulw/Documents/QuantSpark/semantic_scanner/data/' + f'{class_type}_class_train.csv', 
                        test_location='C:/Users/paulw/Documents/QuantSpark/semantic_scanner/data/' + f'{class_type}_class_test.csv',
                        save_location='./models',
                        save_model_name = f'{class_type}_classification_bert_'+dt_string,
                        epochs=15,
                        classification_type=classification_type)