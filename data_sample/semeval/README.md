This dataset is used from Huggingface Datasets library
```
>>> from datasets import load_dataset
>>> d = load_dataset('sem_eval_2010_task_8')
>>> print(d['train'][0])
{'sentence': 'The system as described above has its greatest application in an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>.', 'relation': 3}
```


Paper can be found at [link](https://aclanthology.org/S10-1006.pdf)
