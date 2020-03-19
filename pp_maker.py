import pickle
postag_file='saved_objects/postag_dict_all.p'
with open(postag_file, 'rb') as f:
    postag_dict = pickle.load(f)

print(postag_dict[1].keys())
#print(postag_dict[1]['VB'])

postag_dict[1]['VBN'] = []

import nltk

for word in postag_dict[1]['VBD']:
    pos = nltk.pos_tag([word])[0]
    #print(pos)
    if len(pos) > 1 and pos[1] == 'VBN':
        print(word)
        postag_dict[1]['VBN'].append(word)
        if word in postag_dict[2]:
            postag_dict[2][word].append('VBN')

with open('saved_objects/postag_dict_all+VBN.p', "wb") as pickle_in:
    pickle.dump(postag_dict, pickle_in)