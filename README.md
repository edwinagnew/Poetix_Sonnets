# Poetix_Sonnets

## Duke Poetry Team

Welcome!

### To get a poem

```
import scenery
s = scenery.Scenery_Gen()
```
(NB you can call it anything from s to the_thing_to_write_sonnets_and_stuff as long as no spaces, not already used and you are consistent when referring back to it)
Parameters (all optional):
- templates_file = “path_to_file” default is “poems/jordan_templates.txt”
- words_file = “path_to_file” (default is saved_objects/tagged_words.p -- could be saved_objects/misc_tagged_words.p for archaic words)
- Some others that arent important

`s.write_poem_revised(theme="forest")`
Parameters (all optional):
- theme (= “love” by default). What type of words to write for scNN etc. Can include spaces
- verbose (False by default). Whether or not to print output as it generates
- rhyme_lines (True by default). Whether or not to follow ABAB rhyme scheme 
- all_verbs (=False by default) allows gpt to replace a verb with any type of verb (e.g. a VBD in the template could become a VBG if it wanted it to)
- theme_lines (0 by default). Chooses whether or not to seed generation with sample lines
  -   If theme_lines is an int, it chooses that many lines from a corpus of poems which we deem relevant to the theme
  -   If theme_lines is a string, it has to be either "poem" or "stanza" and then generates one of our own with the same theme to be used as a seed
- k (=5 by default) Chooses the best out of k samples for each line. **most important parameter**
- alliteration (=1 by default) average number of lines to alliterate per stanza
- theme_threshold (0.5 by default) - if ratio between the score of best token and a theme token is above this value, the score of the theme token is increased
- no_meter (=False by default) whether or not to use meter (note: no_meter requires rhyme_lines=False)
- theme_choice (= "and" by default) "and" chooses words which are relevant to all theme words, "or" chooses words which are relevant to any. Assumes that the theme contains more than one word.
- theme_progression (False by default) assuming that the theme has more than one word, write half the poem using one theme and the second half with the other (and some interpolation)
- theme_cutoff (0.35) - words below this similarity to the theme words (accroding to glove and fasttext) are removed from the vocabulary
- weight_repetition (True by default). Whether or not to discourage repetition of words
- gpt_size. Chooses which model to generate with. Normal options are "gpt2", "gpt2-large
, "gpt2-xl"
- tense (="rand" by default) which tense of templates to use
- internal_rhyme (=1 by default) number of previous lines to attempt to internal rhyme with
- dynamik - (off by defualt). Whether to generate lines until they get a certain score instead of a fixed number of times
- random_selection (off by default). Whether to select the top token deterministically or randomly sampled in proportion to their score
- b (1 by default) - the number of beams
- b_inc (1 by default) - the number of tokens to generate before beaming
- beam_score - how to compare different possible hypotheses. Could be "token", "line"
- phi_score - how to evaluate different stuff

To get related words
`s.get_diff_pos(theme, pos)`
- theme = word(s) to get related words to
- pos = POS of desired words (only works for NN, JJ, RB so far)



