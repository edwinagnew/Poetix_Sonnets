import scenery
s = scenery.Scenery_Gen()

s.write_poem_revised()
s.beam_manager.generate(template=['WHICH VBD NN UVBZ TO VB?'],meter=['0_10_10_101_0_1'],num_beams=1)
# s.beam_manager.reset_to_new_line(rhyme_word=None, theme_words=None, alliteration=None, seed=None, internal_rhymes=None)
#
# s.write_poem_revised()
# repeat
