# Using Pre-Trained Language Models for Producing Counter Narratives Against Hate Speech: a Comparative Study
This repository contains the code to replicate our experiments. The **fine-tuning**, **generation** and **evaluation** folders include the scripts and notebooks necessary to replicate our experiments. The **data** folder includes the data splits to fine-tune the models for both the first set of experiments (`mtconan_splits.csv`) and the Leave One Target Out (LOTO) experiments (`LOTO_splits.csv`): the INDEX column can be used to filter the Multitarget-CONAN dataset, available at https://github.com/marcoguerini/CONAN/tree/master/Multitarget-CONAN.  
Important: for fine-tuning GPT-2 we have used special tags, so for training this model a txt file for each split should be created, containing the data preprocessed as follows  
`<hatespeech> The text of the hate speech. <counternarrative> The text of the counter narrative. <|endoftext|>`

## Reference
Further details can be found in our paper: 

Serra Sinem Tekiroğlu, Helena Bonaldi, Margherita Fanton, Marco Guerini. 2022. <em>Using Pre-Trained Language Models for Producing Counter Narratives Against Hate Speech: a Comparative Study.</em> in Findings of the Association for Computational Linguistics: ACL 2022

```bibtex
@inproceedings{tekiroglu-etal-2022-using,
    title = "Using Pre-Trained Language Models for Producing Counter Narratives Against Hate Speech: a Comparative Study",
    author = "Tekiro{\u{g}}lu, Serra Sinem  and
      Bonaldi, Helena  and
      Fanton, Margherita  and
      Guerini, Marco",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.245",
    doi = "10.18653/v1/2022.findings-acl.245",
    pages = "3099--3114",
}

```
