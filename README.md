Introduction
============

Social media is nowadays a powerful tool not only becoming an
increasingly more important part of our communication with others.
Moreover, it is a commonly used way to influence others by sharing our
opinions and also get influenced by others’ ideas. Twitter is a
prototypical example of such a network: In rather short messages it
transports and repeats feelings and opinions directly and quickly
accessible; an automated analysis of such data is, therefore, capable of
getting an idea of what most people tend to be interested in.

In this project, we intend to find the 10 most common concepts in a
corpus of Twitter tweets. We approached this task by performing a set of
preprocessing steps to remove noise, extracting entities that represent
concepts using a POS-tagger, and vectorizing those using a *word2vec*
approach. Afterwards, we were able to obtain the desired outcome by
first clustering similar concepts together to one utilizing the *DBSCAN*
algorithm and post-process the results. Further effort was carried out
to evaluate the optimal parameters for this approach.

Data
====

The data set used in this project consists of **2880** tweets of mostly
English tweets. An example of a typical sample in the data set can be
found in the following table.

|         Column               |                                  Example value                             |
| :--------------------------: | :------------------------------------------------------------------------: |
|         ID                   |                                    188                                     |
|       TWEETID                |                             497547281650823168                             |
|       TWEET                  | RT @EXAMPLE: My future house will have this. http://t.co/URL               |
|      LANGUAGE                |                                     en                                     |

Each sample consists of four attributes which can be split into metadata
and data. The metadata entails the *ID* of the row, a unique *TWEETID*
and a marker for the *LANGUAGE* describing the actual tweet data labeled
as *TWEET*. For the sake of this project, we only considered tweets with
*LANGUAGE* set to *“en”* to simplify the processing pipeline. After this
filtering, **2861** English tweets remained proving that only an
insignificant fraction of **19** tweets were formulated in other
languages. Considering this observation, the reduction and the resulting
potential loss of information seemed to be reasonable according to the
fact it leads to a far more simple model only accepting English as input
without the need to handle the translation of concepts. Additionally, we
excluded the metadata attributes which did not contribute to the process
of finding concepts in the tweets.

Definition of a “concept”
-------------------------

The actual definition of the term “concept” is a challenging task.
Finding a definition of the borders for the generality and specificity
is hard. In this experiment, we focused in general on nouns phrases and
excluded any other type of words. A foundation for the extraction of
nouns phrases was an entity recognition considering names of persons
among others categories like companies, products, events, and
geographical elements like cities and countries. We did not consider the
effect one noun phrase oppose on another in order of a clear and
unambiguous outcome.

Method
======

Pre-Processing
--------------

In order to gain deeper insight into current Twitter trends, it is
necessary to convert the unstructured Tweets into a format allowing the
appliance of current state-of-the-art algorithms. The following sections
describe the required steps to find the 10 most common concepts in the
data set.

### Noise Removal

Twitter as a medium of our social interaction online is hardly
comparable to the “ordered” type of text usually used in the natural
language processing: With its massive number of abbreviations, ellipsis,
and neologisms a successful parsing gets even more complicated than on
other corpora used for natural language processing. Therefore, to
extract the essential concepts from the corpus of tweets, we had to
perform a number of noise removal steps on the data before we start to
extract the relevant features.

First of all, it is important to strip out all the characters carrying
no meaning for the actual concept extraction. Advanced Unicode
characters and URLs just like any non-printable characters are part of
this category: While emoticons and formatting are useful for the natural
way of communicating, for our algorithms they are just distraction.
Another potential source of such bloat are incomplete words too long for
the 140 char limitation of Twitter, abbreviated by “...”.

While these cases are obvious, specific situations require a more subtle
handling: Twitter uses annotations to mention other users in tweets by
typing their name with the “@”-character in the front. On the one hand,
mentions at the beginning of a tweet are used to address or reply to
another Twitter user and carry no further meaning. On the other hand,
mentions at other places may represent an entity posting on Twitter and
being a concept itself. Therefore, while in preprocessing the former
case was removed, in the second the annotation was converted into a
proper noun by excluding the first character and capitalizing the second
one.

The second Twitter-specific preprocessing hack considers hashtags: Even
if they carry concepts, they do not fit into the standard way we would
parse language. Like the mentions processed before, a position-dependent
way of dealing with them is required: At the end, all hashtags got
extracted and stored in a global hashtag cache, in any other position a
conversion into a proper noun was done again.

After all these cleaning steps, original tweets and tweets were no
longer distinguishable. This fact does not imply any loss of information
as long as the weight of the comment is stored, but allows to trivially
reduce the number of samples being parsed in the next steps.

### Entity Extraction

To finally extract entities representing concepts a number of natural
language processing algorithms were applied to the cleaned tweets. The
utilized technology for the natural language processing was the Python
toolkit *spaCy* [@spacy]. While *NLTK* [@nltk] utilized *spaCy*’s POS
tagger by default which leads to comparable results, in recognition of
entities it performed significant inferior both in term of speed and
performance [@spacyEvaluation].

In the first step, the actual tweet was tokenized into different
entities. These entities represent words, punctuation, and pre-, in- and
suffixes useful in the further processing.

Building on this representation, a POS-tagging algorithm was applied
utilizing the *OntoNotes 5* corpus [@ontoNotes]. In this step, semantic
knowledge about the type of entity was added to the extracted tokens
according to the *Google Universal POS tagset* [@universalPos]. The
crucial role of this function is the reason for the extended
preprocessing due to the importance of the predecessors of a word for
this part of the analysis.

The actual context extraction was performed afterward on the foundation
of this analysis. According to the type of words, noun-based chunks were
extracted as actual contexts.

### Vectorization

One problem remains: Even after an extraction of the entities, a variety
of different words for the same concept remains. While “arturo vidal,”
“arturo vidal football” and “Vidal sportnews” refers to the same idea,
they a syntactically different. At this point, two different strategies
may be usable.

On the one hand, a rule-based approach might be used: By encoding our
semantic knowledge into rules and using a collection of dictionaries,
i.e., to detect common names we would be able to gain high precision.
Nevertheless, such a solution would not be scalable due to in small
general recall; on new data the likelihood of a failure is quite high.
On the other hand, a machine learning algorithm might be used. Utilizing
a statistical approach, the model would probably not have the precision
of a hand-written approach but might work on a far more significant
number of concepts. For using this method and find all the clusters of
interest, it is necessary to convert the entities into a representation
suitable for data mining algorithm.

A first idea may be the occurrences of a term. Foundation of this would
be the theory of similar concepts may occurring with similar
probability. By using a more advanced measure like *TF-IDF*, one may
even be able to handle exceptionally commonly occurring terms like
standard English stopwords appropriately. Nevertheless, this system is
limited due to its simplicity.

A more advanced approach commonly used in modern natural language
processing is therefore *word2vec* [@word2vec]. In this family of
models, the actual entity gets transformed into a vector usually with an
extremely high dimensionality. Based on massive corpora, those vectors
get clustered in a way, that often occurring patterns results in vectors
having a low distance towards each other. In our approach, the *GloVe*
project [@glove] from the University Stanford was used. The pre-computed
300-dimensional word vectors are even capable of mapping syntactic
completely different words like “frog” to Latin names like “Rana” or
“Eleutherodactylus.” On this foundation, the actual vectors of the
extracted entities were generated. In the case of multiple expressions,
the averaged vector over all tokens was used.

Concept Extraction
------------------

Clustering is a commonly used type of task in the field of machine
learning. Usually this kind of algorithm it is used in a unsupervised
environment: To predict an unknown sample its vector is compared to a
fixed number of clusters ordered according to their class label. In most
cases, the group with the minimal distance corresponds to the class the
sample fits. Such kinds of setup are not applicable in the context of
concept extraction due to the unknown number of groups. Moreover, an
algorithm is needed capable of handling the weights extracted as the
number of re-tweets. One of the few candidates matching all criteria is
*Density-based spatial clustering of applications with noise* [@dbscan
*DBSCAN*]. Beside the used similarity metrics, the most important
parameter of this algorithm is called epsilon: It defines the range of
the neighborhood around a point which is considered as a cluster.

After the actual clustering, some post processing was applied to gain
optimal results. The clusters were scored according to their frequency
divided by the square root of the number of concepts in the cluster.
Prior experiments have shown that huge clusters of incoherent words may
appear biasing the results - by penalizing a high number of words,
smaller and more meaningful groups are rated higher. Clusters containing
only one item were considered as noise and removed from the results if
they were mentioned or re-tweeted less than **20** times. It is likely
that such a parameter might be set to a higher value on bigger data sets
because the the meaning of the term “importance” scales with the number
of the available samples.

For a selection of the main concept of each cluster, the entity with the
highest individual number of occurrences was chosen. The first ten of
those concepts sorted according to the score of their clusters are,
therefore, the final result of the concept extraction process.

Results & Discussion
====================

During experimentation, by changing the epsilon value and the distance
metric of the *DBSCAN* algorithm, we achieved very different concepts
output as the most important ones.

By choosing the epsilon to be too small we would get very similar
concepts clustered to one while being in risk of clustering concepts
that should be in one cluster into several clusters. “Arturo Vidal” and
“DeadlineDayLive Arturo Vidal” should be regarded as the same concept
and, hence, also be in the same cluster.

On the other hand, with a large epsilon we would cluster concepts that
are less similar together, but we would risk clustering too different
concepts together which is unfavourable. An example can be different
football clubs seeming to be different concepts according to our working
definition. The vectors produced by *word2vec* were similar enough to be
placed in the same clusters if the epsilon was large enough.

The similarity of two concepts is defined by the similarity or distance
metric. Since we clustered concepts together if they were within an
epsilon range defined by this metric, the metric selected could decide
whether two concepts end up in the same cluster or not. There are
primarily two types of similarity metrics, Euclidean and non-euclidean,
each of which measures similarity of two vectors in different ways. The
Euclidean similarity could be defined as the spatial distance between
the two vectors, whereas non-euclidean similarity could be defined as
the similarity in the properties of the vectors but not their spatial
distance. In the experiments, we tested both Euclidean similarity and
cosine similarity, a non-euclidean measurement, and use both to cluster
the concepts together.

In addition to the epsilon and the similarity metric, the *DBSCAN*
algorithm also makes use of a *minPts* parameter deciding how many
neighbouring elements an element should at least have. If an element has
fewer neighbours than this threshold it is disregarded as noise. We set
this parameter to be **0** since we do not want to disregard frequent
concepts even though they have too few neighbouring concepts. Noise is
instead something we defined as less frequent concepts, and it is
removed if they are below some threshold we set.

The following experiments show the different output of the 10 most
common concepts with the use of different parameters. The best selection
of parameters depends entirely on the corpus; defining a measurement is
rather hard. The output should at least be diverse regarding concepts
and should also contain highly frequent concepts.

DBSCAN with Euclidean
---------------------

In this experiment we used an **epsilon=0.1**. As we see from the
output, we do not get 10 concepts. We got too few clusters because we
filter out all clusters with concepts with frequencies less than
**100**; this fact is an indication that our epsilon was too small.

``` {.python language="Python"}
    [(manchester united, 1208),
     (arturo vidals, 1017),
     (new zealand, 682),
     (united, 69),
     (sportupdate paul scholes, 19),
     (conservative, 5),
     (newzealand redstag, 2),
     (lucy johnston, 1)]
```

However, by analyzing the frequencies of the concepts further, we set
the lower boundary of frequencies to **20**. This was no longer a
problem that could occur since we had afterwards more than 10 clusters
with only one concept in them. A frequency limit of **100** was too high
and not enough concepts met this criterion as we can see above.

We experimented with increasing the epsilon (**epsilon=0.5**). As we see
from the output, most of the concepts did not seem to be among the most
frequent concepts. Some of the most frequent concepts were placed in the
same clusters which contain even more frequent concepts. They were
therefore not present in the top 10 since the more frequent concepts in
the same clusters were elected to represent the concept. Unless these
concepts were in fact a part of the same real concept, they were
misplaced. Less frequent that reside in other clusters will, therefore,
be chosen instead. Some clusters were just too big and this indicates
that our epsilon was too large.

``` {.python language="Python"}
    [(manchester united, 699),
     (arturo vidals, 166),
     (australian, 67),
     (manchester, 55),
     (tuvalu, 43),
     (napoli, 37),
     (subaru, 34),
     (la stampa, 29),
     (hammer, 28),
     (sydney, 24)]
```

We performed a grid search according to the score and its variance to
determine the best epsilon for clustering, and we found out that the
clustering did best with as small as possible epsilon. Although
**epsilon=0.005** in the particular run produced the output below, this
result remained the same for smaller epsilon and up to some epsilon
where the result was displayed concepts less frequent than it should.
This occurred because the most frequent concepts reside in their
clusters when the Euclidean distance is used. This was, however, a
special case.

``` {.python language="Python"}
    [(manchester united, 1208),
     (arturo vidals, 1017),
     (new zealand, 682),
     (united, 98),
     (the worlds first climate change refugees, 96),
     (paul scholes, 90),
     (australian, 67),
     (manchester, 55),
     (james wilson, 46),
     (tuvalu, 43)]
```

DBSCAN with Non-Euclidean
-------------------------

A larger epsilon, i.e., **epsilon=0.5**, produced too large clusters
indicating that non-similar concepts were placed in the same cluster.
The result can be seen below which show very few concepts.

``` {.python language="Python"}
    [(manchester united, 230),
     (newzealand redstag, 39),
     (lvg, 21),
     (scuttlebutt sailing, 3)]
```

A too small epsilon produced a set of too many similar concepts that
should have been clustered together but was not due to the small epsilon
range. This was, however, expected behaviour.

The best output produced with cosine similarity as metric was with
**epsilon=0.2**, and it is shown below. It was diverse and contains only
frequent concepts.

``` {.python language="Python"}
    [(manchester united, 699),
     (new zealand, 245),
     (arturo vidals, 165),
     (australian, 67),
     (paul scholes, 57),
     (united, 47),
     (the worlds first climate change refugees, 44),
     (tuvalu, 43),
     (newzealand redstag, 39),
     (napoli, 37)]
```

The best results produced by either similarity metric were satisfying as
they were diverse in concepts and contained the concepts occurring with
the highest frequency in the tweets. However, whether or not we should
use the Euclidean distance or non-euclidean distance to determine that
two concepts are the same depends on what we want to achieve. Do we want
the difference in the words or the difference in the semantics of the
words? In this case, we would go with cosine distance over the Euclidean
distance since it seems to produce better clusters for somewhat similar
concepts. If we take Manchester and Manchester United as an example, we
can see that the best results for the two metrics make more sense with
cosine distance than euclidean since it clusters the concepts together.
In general, Cosine similarity is also usually the preferred metric to
measure the similarity of two words in natural language processing.

Conclusion & Further Work
=========================

It is stunning in which way even rather simple approaches on a small
data set is capable of extracting the inner concepts behind the noisy
words of our digital life on Twitter. Even without utilizing modern
multithreading architectures the computational costs of the extraction
of higher semantic clusters were that low that an appliance on far
bigger data sets would be no problem.

Probably, the biggest problem in the field of concept extraction is the
incredibly high ambiguity in the term “concept”. Finding concepts is a
task mostly suitable for humans. Algorithms may clearly assist us,
however, with limitations due to the rather constrained areas of
knowledge modern artificial intelligence is capable of handling. This
fact implies another problem: The completely missing ground truth. An
actual evaluation of the results cannot utilize hard measurements like
precision or F-values. It requires a human being on the other end of the
pipeline for detecting useful parameters for producing optimal results.
A clear definition of what a concept is would make the performed grid
search significantly more useful. At the moment it seems that only the
actual understanding of the semantic relations is the remaining topic
which might be further improved, all the other parts seem to be
satisfying. On the other hand, we have to ask ourselves again the
question: What are actually important concepts; where exactly do we set
the cuts for the generality of our model and the specificity of it? Do
we really want to classify “Arturo Vidal”, “Manchester United” and
“Louis van Gaal” as different concepts? Or would “Football” be the
better categorization? Finding the golden path between a model general
enough to exclude unimportant rumors and specialized enough not to find
only general knowledge is probably not a science, but more of an an art.
Our approach might be a good starting point for further extractions and
a useful tool to deal with such tasks, but it is clearly not a general
out-of-the-box solution applicable for every situation. The old rule of
data science remains: There is no silver bullet for everything.
