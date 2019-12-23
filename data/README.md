# Preprocess the Dataset

We read the [dataset]( http://data.allenai.org/propara/ ) provided by AI2 and perform the following pre-process procedure.

#### Read the paragraph

Read the topic, the prompt and paragraph text from ``Paragraphs.csv``.

#### Split the train/dev/test sets

This step uses the ``Train_Dev_Test.csv`` file.

#### Read the annotation

Read the human annotations of entities and states from ``State_change_annotations.csv`` file. Create the gold state sequence (CREATE, MOVE, etc.) and the gold location sequence for each entity in each paragraph respectively.

In addition to reads each sentence and records the entities as well as their states at each timestep, we further do the following things:

1. We tokenize the paragraph, the prompt and all entities using ``SpaCy``. All text are cleaned and lower-cased.

2. We extract location candidates from the paragraph. The rules are:

   (1) Noun phrases, with as many adjectives as possible before the central noun.

   (2) Noun phrases, with as many nouns as possible as attributes before the central noun. However, we will retain the central noun as another candidate.

   (3) Two nouns connected by "and" or "or".

   The POS tags are extracted using ``flair``. All candidate locations are lemmatized by ``SpaCy`` after extraction. The recall of gold locations using such methods on train/dev/test sets is **85.7%**.

3. For the train and dev sets, if the gold location is not included in the candidates, we manually add them to the candidate set. This is mainly for expanding the size of trainable instances in location prediction, and for the convenience of calculating cross-entropy loss. For the test set, we do not use such method because we obviously cannot know the gold location while testing.

5. We find the mention positions of the entities, the verbs, as well as all location candidates. This is because we need to compute masks for these words in the model. For verbs, their positions are determined using POS tags extracted by `flair`. For entities and locations, we first lemmatize them as well as the sentence, then perform sub-sequence matching to find their positions.

Every entity in each paragraph corresponds to one instance in the dataset. Therefore, you should acquire 1504, 175 and 236 data instances for train/dev/test respectively.

Finally, the processed data are stored in JSON format, namely `train.json`, `dev.json` and `test.json`.