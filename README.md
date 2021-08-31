# eMLM
Code and Pre-trained models for our ACL 2021 paper entitled: "eMLM: A New Pre-training Objective for Emotion Related Tasks". The pretrained model will be available soon.

## Implementation

The implementation closely follows the Whole Word masking procedure from HuggingFace: https://huggingface.co/transformers/_modules/transformers/data/data_collator.html#DataCollatorForWholeWordMask

## Paper

If you use the model in your research, please consider citing out paper:

```bibtex
@inproceedings{sosea-caragea-2021-emlm,
    title = "e{MLM}: A New Pre-training Objective for Emotion Related Tasks",
    author = "Sosea, Tiberiu  and
      Caragea, Cornelia",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-short.38",
    doi = "10.18653/v1/2021.acl-short.38",
    pages = "286--293",
    abstract = "BERT has been shown to be extremely effective on a wide variety of natural language processing tasks, including sentiment analysis and emotion detection. However, the proposed pretraining objectives of BERT do not induce any sentiment or emotion-specific biases into the model. In this paper, we present Emotion Masked Language Modelling, a variation of Masked Language Modelling aimed at improving the BERT language representation model for emotion detection and sentiment analysis tasks. Using the same pre-training corpora as the original model, Wikipedia and BookCorpus, our BERT variation manages to improve the downstream performance on 4 tasks from emotion detection and sentiment analysis by an average of 1.2{\%} F-1. Moreover, our approach shows an increased performance in our task-specific robustness tests.",
}
```
