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

`s.write_poem_flex(theme="forest")`
Parameters (all optional):
- theme (= “love” by default). What type of words to write for scNN etc. Can include spaces
- random_templates (= True by default). If False, uses the first 14 templates in the file in order. If True, randomly picks a template for each line, conforming to punctuation rules and not starting stanza with AND etc.
- rhyme_lines (= True by default). Whether or not to follow rhyme scheme. If templates arent working, try without rhyme. It also runs quite a lot quicker without rhyming
- verbose (= False by default) whether or not to output loads of info along the way. Quite entertaining to watch while you’re waiting for it to finish generating
- all_verbs (=False by default) allows gpt to replace a verb with any type of verb (e.g. a VBD in the template could become a VBG if it wanted it to)
- k (=5 by default) Chooses the best out of k samples for each line
- theme_choice (= "and" by default) "and" chooses words which are relevant to all theme words, "or" chooses words which are relevant to any

To get related words
`s.get_diff_pos(theme, pos)`
- theme = word(s) to get related words to
- pos = POS of desired words (only works for NN, JJ, RB so far)



