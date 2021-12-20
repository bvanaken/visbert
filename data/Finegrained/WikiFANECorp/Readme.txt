===========================================================================
WikiFANECorp: Wikipedia based Fine-grained Arabic Named Entity Corpus
===========================================================================

Those are corpora have been automatically compiled from the Arabic Wikipedia. The technical methodology behind this work is fully described in the paper:

F. Alotaibi and M. Lee, "Automatically Developing a Fine-grained Arabic Named Entity Corpus and Gazetteer by utilising Wikipedia", 2013 (awaited decision).


Meanwhile, you can use the dataset by citing the following paper instead:

F. Alotaibi and M. Lee, "Mapping Arabic Wikipedia into the Named Entities Taxonomy", In Proceedings of COLING 2012: Posters, p43-52, IIT, Mumbai, India, December 8-15. 2012.

and can be downloaded at:
http://www.cs.bham.ac.uk/~fsa081/

** Please use the details of paper mentioned above for citation.

This dataset is licensed under a Creative Commons Attribution-ShareAlike 3.0 Unported License. (see License.txt)


The produced WikiFANECorp is compiled in two alternatives as follows:

1) WikiFANEWhole: All sentences of the Arabic Wikipedia articles were retrieved to compile to corpus. 76821 sentences. 2,023,496 tokens.

2) WikiFANESelective: Sentences which have at least one NE phrase were retrieved to compile the corpus. (tags density is high). 57126 sentences. 2,021,177 tokens

* The whole tagged Wikipedia is available upon requested.

The format of the corpus is presented using CoNLL representation, i.e. two columns. The first column is the word and the second column is the tag. The encoding scheme used is BIO. The tags are presented to keep both levels of granularities as (BIO-Coarse_Fine-grainedTag).

For example:

...
تأسست    O
الفرقة    O
في    O
لندن    B-GPE_Population-Center
عام    O
1958    O
من    O
طرف    O
كليف    B-PER_Artist
ريتشارد    I-PER_Artist
وكانت    O
...
 


Authors: Fahd Alotaibi* [fsa081@cs.bham.ac.uk] and Mark Lee [m.g.lee@cs.bham.ac.uk].

* for correspondence

----------------------------------
** More about the licence is found in License.txt
----------------------------------

===========================================================================