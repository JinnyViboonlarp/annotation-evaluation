# annotation-evaluation

This repository stores python codes for evaluating the performance (F1 score) of an NER model (whether spaCy-based or DBpedia-Spotlight-based) when compared to the gold-standard annotation files (which are in the folder `annotations`).

For each NER model, first, `annotate.py` is called to create annotation files in the same format as the gold-standard annotation files. The files annotated with named entities recognized by spaCy-based models are stored in the folder `model-annotations`, while the files annotated with entities recognized by the DBpedia-Spotlight model are stored in the folder `dbpedia-annotations`.

After annotation files are created, then, `evaluate.py` is called to calculate F1, relative to the gold-standard annotations. `evaluate.py` could be configured for different evaluation option, e.g., whether the entities are counted by noun chucks or by tokens, whether a location-typed entity could count as an organization-typed (as in the case of countries' names, where there is some gray area), etc.
