
import random
import torch
import json
import numpy as np
import string
import pandas as pd


from py_files import helper

import theme_word_file
import gpt_2
import fasttext_simfinder
from difflib import SequenceMatcher

from collections import Counter



#from transformers import BertTokenizer, BertForMaskedLM
#from transformers import RobertaTokenizer, RobertaForMaskedLM

from nltk.corpus import wordnet as wn

import poem_core



class Scenery_Gen(poem_core.Poem):
    def __init__(self, model=None, words_file="saved_objects/tagged_words.p",
                 syllables_file='saved_objects/cmudict-0.7b.txt',
                 extra_stress_file='saved_objects/edwins_extra_stresses.txt',
                 top_file='saved_objects/words/top_words.txt',
                 templates_file=('poems/jordan_templates.txt', "poems/rhetorical_templates.txt"),
                 #templates_file='poems/number_templates.txt',
                 mistakes_file=None):

        #self.templates = [("FROM scJJS scNNS PRP VBZ NN", "0_10_10_1_01_01"),
         #                 ("THAT scJJ scNN PRP VBD MIGHT RB VB", "0_10_10_1_0_10_1"),
          #                ("WHERE ALL THE scNNS OF PRP$ JJ NNS", "0_1_0_10_1_0_10_1"),
           #               ("AND THAT JJ WHICH RB VBZ NN", "0_1_01_0_10_1_01")]

        poem_core.Poem.__init__(self, words_file=words_file, templates_file=templates_file,
                                syllables_file=syllables_file, extra_stress_file=extra_stress_file, top_file=top_file, mistakes_file=mistakes_file)
        self.vocab_orig = self.pos_to_words.copy()

        if model == "bert":
            self.lang_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.lang_vocab = list(self.tokenizer.vocab.keys())
            self.lang_model.eval()
            self.vocab_to_num = self.tokenizer.vocab

        elif model == "roberta":
            self.lang_model = RobertaForMaskedLM.from_pretrained('roberta-base') # 'roberta-base'
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base') # 'roberta-large'
            with open("saved_objects/roberta/vocab.json") as json_file:
                j = json.load(json_file)
            self.lang_vocab = list(j.keys())
            self.lang_model.eval()
            self.vocab_to_num = {self.lang_vocab[x]: x for x in range(len(self.lang_vocab))}


        else:
            self.lang_model = None

        #with open('poems/kaggle_poem_dataset.csv', newline='') as csvfile:
         #   self.poems = csv.DictReader(csvfile)
        self.poems = list(pd.read_csv('poems/kaggle_poem_dataset.csv')['Content'])
        self.surrounding_words = {}

        #self.gender = random.choice([["he", "him", "his", "himself"], ["she", "her", "hers", "herself"]])

        self.theme_gen = theme_word_file.Theme()

        self.fasttext = fasttext_simfinder.Sim_finder()

        self.theme = ""

    #override
    def get_pos_words(self,pos, meter=None, rhyme=None, phrase=()):
        """
        Gets all the words of a given POS
        Parameters
        ----------
        pos - the POS you want
        meter - (optional) returns only words which fit the given meter, e.g. 101
        phrase (optional) - returns only words which have a phrase in the dataset. in format ([word1, word2, word3], i) where i is the index of the word to change since the length can be 2 or 3
        """
        #print("oi," , pos, meter, phrase)
        #punctuation management
        punc = [".", ",", ";", "?", ">"]
        #print("here1", pos, meter)
        #if pos[-1] in punc:
        #    p = pos[-1]
        #    if p == ">":
        #        p = random.choice(pos.split("<")[-1].strip(">").split("/"))
        #        pos = pos.split("<")[0] + p
        #    return [word + p for word in self.get_pos_words(pos[:-1], meter=meter, rhyme=rhyme)]
        #print("here", pos, meter, rhyme)
        #similar/repeated word management
        if "*VB" in pos:
            ps = []
            for po in ["VB", "VBZ", "VBG", "VBD", "VBN", "VBP"]:
                ps += self.get_pos_words(po, meter=meter, rhyme=rhyme, phrase=phrase)
            return ps
        if pos not in self.pos_to_words and "_" in pos:
            sub_pos = pos.split("_")[0]
            word = self.weighted_choice(sub_pos, meter=meter, rhyme=rhyme)
            if not word: input("rhyme broke " + sub_pos + " " + str(meter) + " " + str(rhyme))
            #word = random.choice(poss)
            if pos.split("_")[1] in string.ascii_lowercase:
                #print("maybe breaking on", pos, word, sub_pos)
                self.pos_to_words[pos] = {word: self.pos_to_words[sub_pos][word]}
            else:
                num = pos.split("_")[1]
                if num not in self.pos_to_words:
                    #self.pos_to_words[pos] = {word:1}
                    self.pos_to_words[num] = word
                else:
                    poss = self.get_pos_words(sub_pos, meter)
                    word = self.pos_to_words[num]
                    self.pos_to_words[pos] = {w: helper.get_spacy_similarity(w, word) for w in poss}
                    return poss

            return [word]
        #if rhyme: return [w for w in self.get_pos_words(pos, meter=meter) if self.rhymes(w, rhyme)]
        if len(phrase) == 0 or len(phrase[0]) == 0: return super().get_pos_words(pos, meter=meter, rhyme=rhyme)
        else:
            if type(meter) == str: meter = [meter]
            ret = [word for word in self.pos_to_words[pos] if word in self.dict_meters and any(m in self.dict_meters[word] for m in meter)]
            phrases = []
            for word in ret:
                phrase[0][phrase[1]] = word
                phrases.append(" ".join(phrase[0]))
            #print(phrases, ret)
            ret = [ret[i] for i in range(len(ret)) if self.phrase_in_poem_fast(phrases[i], include_syns=True)]
            return ret

    #@override
    def last_word_dict(self, rhyme_dict):
        """
        Given the rhyme sets, extract all possible last words from the rhyme set
        dictionaries.

        Parameters
        ----------
        rhyme_dict: dictionary
            Format is   {'A': {tone1 : {similar word: [rhyming words], similar word: [rhyming words], etc.}}, {tone2:{...}}},
                        'B': {tone1 : {similar word: [rhyming words], similar word: [rhyming words], etc.}}, {tone2:{...}}}
                        etc
        Returns
        -------
        dictionary
            Format is {1: ['apple', 'orange'], 2: ['apple', orange] ... }

        """
        scheme = {1: 'A', 2: 'B', 3: 'A', 4: 'B'}
        last_word_dict = {}

        first_rhymes = []
        for i in range(1, len(scheme) + 1):
            if i in [1, 2]:  # lines with a new rhyme -> pick a random key
                last_word_dict[i] = [random.choice(
                    list(rhyme_dict[scheme[i]].keys()))]  # NB ensure it doesnt pick the same as another one
                j = 0
                while not self.suitable_last_word(last_word_dict[i][0], i - 1) or last_word_dict[i][0] in first_rhymes:
                    # or any(rhyme_dict['A'][last_word_dict[i][0]] in rhyme_dict['A'][word] for word in first_rhymes):
                    last_word_dict[i] = [random.choice(list(rhyme_dict[scheme[i]].keys()))]
                    if not any(self.templates[i - 1][1].split("_")[-1] in self.dict_meters[w] for w in
                               rhyme_dict[scheme[i]]):
                        word = last_word_dict[i][0]
                        if self.templates[i - 1][0].split()[-1] in self.get_word_pos(word) and len(
                                self.dict_meters[word][0]) == len(self.templates[i - 1][1].split("_")[-1]) and any(
                                self.suitable_last_word(r, i + 1) for r in rhyme_dict[scheme[i]][word]):
                            self.dict_meters[word].append(self.templates[i - 1][1].split("_")[-1])
                            print("cheated with ", word, " ", self.dict_meters[word],
                                  self.suitable_last_word(word, i - 1))
                    j += 1
                    if j > len(rhyme_dict[scheme[i]]) * 2: input(str(scheme[i]) + " " + str(rhyme_dict[scheme[i]]))
                first_rhymes.append(last_word_dict[i][0])

            if i in [3, 4]:  # lines with an old rhyme -> pick a random value corresponding to key of rhyming couplet
                letter = scheme[i]
                pair = last_word_dict[i - 2][0]
                last_word_dict[i] = [word for word in rhyme_dict[letter][pair] if self.suitable_last_word(word, i - 1)]
                if len(last_word_dict[i]) == 0:
                    print("fuck me", last_word_dict, i, self.templates[i])
                    print(1 / 0)
        return last_word_dict

    #@ovveride
    def suitable_last_word(self, word, line):
        pos = self.templates[line][0].split()[-1].split("sc")[-1]
        meter = self.templates[line][1].split("_")[-1]
        return pos in self.get_word_pos(word) and meter in self.dict_meters[word]


    def write_poem_flex(self, theme="love", verbose=False, random_templates=True, rhyme_lines=True, all_verbs=False, theme_lines=0, k=5, alliteration=True, theme_threshold=0.5, theme_choice="and"):
        if not self.gpt:
            if verbose: print("getting gpt")
            self.gpt = gpt_2.gpt_gen(sonnet_object=self, model="gpt2")
            #self.gpt = gpt_2.gpt_gen(sonnet_object=self, model="gpt2-large")
        self.reset_gender()

        self.pos_to_words = self.vocab_orig.copy()
        self.theme = theme

        if theme_lines > 0: self.update_theme_words(theme=theme)
        theme_contexts = self.theme_gen.get_cases(theme) if theme_lines > 0 else [""]
        if verbose and theme_lines: print("total lines", len(theme_contexts), "e.g.", random.sample(theme_contexts, min(len(theme_contexts), theme_lines)))

        if theme:
            theme_words = {}
            theme_words[theme] = {}

            for pos in ['NN', 'JJ', 'RB']:
                if pos not in theme_words[theme]: theme_words[theme][pos] = []
                if theme_choice == "and":
                    theme_words[theme][pos] += self.get_diff_pos(theme, pos, 10)
                else:
                    for t in theme.split():
                        theme_words[theme][pos] += self.get_diff_pos(t, pos, 10)
                if verbose: print("theme words, ", pos, ": ", len(theme_words[theme][pos]), theme_words[theme][pos])
            rhymes = [] #i think??
            if verbose: print("\n")
            """elif theme:
            #rhymes = list(self.getRhymes(theme, words=self.words_to_pos.keys()).keys())
            n = 25
            rhymes = [theme]
            #while len(set(rhymes)) < 100 or not any(len(self.get_meter(w)) == 1 for w in rhymes):
            #    n += 25
            #    rhymes += [x for x in self.fasttext.get_close_words(random.choice(rhymes), n=n) if len(x) > 3 and x in self.words_to_pos and any(m in ["1", "01", "101", "0101", "10101"] for m in self.get_meter(x))]
            #    #nouns
            #    rhymes = [r for r in rhymes if r in self.pos_to_words['NN'] or r in self.pos_to_words["NNS"] or r in self.pos_to_words['ABNN']]

            #adjectives
            if theme == "forest":
                jjs = 'dense dark thick primeval deep whole open virgin green tropical vast wild black ancient gloomy distant tangled silent unbroken beautiful heavy mighty entire impenetrable tall trackless dead extensive endless native primitive immense high dim nearby pathless strange fine wide national magnificent huge african noble original northern royal natural miniature cool deeper thin DESCRIBING WORDS CONTINUE AFTER ADVERTISEMENT shadowy adjacent lofty petrified narrow primary quiet western veritable shady lush boundless empty interminable sparse true perfect mysterious dry dismal terrible sombre moonlit fairy rough low free german rich splendid brazilian wet white canadian fair unknown grim swampy leafy verdant long blue solitary dusky deepest snowy southern secondary savage untouched soft broad fine open wonderful golden somber brown solemn dark green gigantic majestic main sacred peaceful certain untamed towering dreary wintry ordinary grand old desolate uninhabited cold primaeval equatorial alien continuous glorious nocturnal almost impenetrable magical purple dangerous eastern upland larger nearest more open fresh gray sodden large flat densest undulating orbital enormous temperate simple weird familiar upper grand solid poor lone short beloved normal great flaming darker dense tropical sal lower coastal secret fragrant remote autumnal regular faroff misty usual famous eternal private crystal clear steep dreadful artificial dank bleak outer primordial common metallic humid almost unbroken thickest deep green french red darkest sweet eerie actual unfamiliar amazonian general unexplored naked boreal massive old various fossil oldgrowth bare metal darkling major yellow moist different deep dark particular dense green murky important spacious healthy public nighttime wondrous impervious valuable snowcovered central australian ragged great dark shaggy vast primeval picturesque grey delightful primal bad thicker warm european gorgeous muddy treacherous grassy untracked former thick dark undisturbed monotonous less impassable unchanging awful clean charming unburned nighted stony evil benighted rude dim old limitless cruel lifeless british intricate rainy inhospitable romantic deep dark mountainous pure horrible present central national siberian hostile wooded entangled terrestrial aboriginal principal thick green uncontrolled seemingly endless dappled damned gentle uncut rugged precious handsome dark and tangled finest indonesian mature single uninterrupted sunken edible barren hushed inner great northern plain central national older great black tremendous genuine mere dense primeval difficult fearful sentient faint shy rocky fierce dear chief imperial complete prim�val total seeming vertical dark and gloomy charred great green pale frequent wildest bizarre bavarian opposite destructive mid flush scented darkgreen celestial smaller crystalline spectral biggest tranquil horizontal sunny thorny modern next innumerable aromatic fine old bloody concrete deepest darkest stupendous oldest occasional bright heathen hideous inverted full icy taller feathery seventh sandy dense and tangled unending parklike younger local nearer universal semitropical great primeval dense dark exuberant burnedout underwater damn antique happy fantastic innocent everlasting considerable otherwise unbroken miserable nightmarish lowlying proper prostrate superb wellknown mournful strong ancestral bluegreen rather dense predawn sylvan bushy cool green vacant vast green obscure alternate richest sharp myriad extreme sinister stupid raw proud absolute marvelous metal shallow dark tangled dreamy prehistoric dark dense average stalwart unnatural gnarled thick and gloomy sizable endless primeval typical lonesome scanty inaccessible motionless ashen troublesome dense unbroken powerful formidable abundant slender best semitropical pristine thick dark frightful timbered soggy foggy great southern angular similar indoor shaggy old characterless apparently endless decayed immeasurable venerable inscrutable disastrous gay curved unseen electronic fullfledged sullen further uncanny wretched human sparkling special calm most mexican socalled dark and unknown almost boundless great equatorial excellent coral himalayan scientific vigorous comparatively open norwegian contiguous green and brown brushy rare festival thick tropical cavernous small breathless emerald fabled craggy immediate subtropical greatest remarkable continental unprotected curious vast and vacant gaunt dark and endless dense dark dense untouched mythical experimental pillared stormy federal great african unsettled sufficient hot higher outlying sudden somber primitive teutonic awesome great amazonian monstrous great gloomy scorched gently undulating urban ample fertile dark and eerie amazing extraordinary neverending elfin lefthand deep and dense shady dense dense and dark good open old german almost impervious dense and beautiful minor timid pleasant glassy virtual dark and mysterious drab colossal lush green horned strange uncanny thick impenetrable demented dark and treacherous invisible dark gloomy sad dark silent untidy gray and empty horrid spicy rich green windblown old lesser wild and tangled bold perilous far distant delicious delicate magnificent primeval whole green moderately clear luxurious peruvian sweetsmelling just undisturbed brownish onrushing whitish whole little nice young mundane thick and thin hazy immemorial sooty lush wild farthest smooth crazy wise invincible dull circular thickly wooded impenetrable primeval wild and thick supposedly wild and thick supposedly wild whole big great brazilian sleepy vast tropical uneven temporary wellkept big tallest nearly virgin tall thick elder vocal divine fungal windy highest skilled great silent whole wide severe barbarous luminous dark primeval infernal brilliant gloomy virgin heaviest untenanted hard diminutive dark dank thorough belgian softwood looser plain old reedy tribal slowmoving endless empty primeval tropical poisonous unspoiled heavier coloured perfumed regular little unimproved tortuous large and spacious distant ancient cooler colored rusted great old vast and trackless gloomy old haitian seaside spongy eternally moonlit noisy claustrophobic bizarre claustrophobic unknown and trackless cool silent brittle impromptu old old horrible lofty dense angular uncouth infinite mineral painful solemn and beautiful effective sturdy uncomplaining splendid open fenced continual many other great shady dear old unfenced oldtime dumb sorry stumpy green old legendary green upper broad green triangular resistant ablaze pleasant open slim harpy deep quiet unoccupied highaltitude crimson grotesque woolen unknown and untracked tropical african distant african unhealthy miraculous little vast monotonous silent inner thick tangled mangled skeletal woodland nearly impenetrable dark primeval unique deep silent quaint incredible northeastern obscene rich black certain distant ravenous grandest virgin tropical omnipresent dark tangled fullgrown sweltering weary windswept restless pale green unbounded social phantom once beautiful dense and gloomy cool green scenic flat open faraway illusive lilliputian enchanting inextricable druidical dark dense wild free heavy primitive tolerably open fatal asiatic watery nether dark and unbroken same open same open undulating chaste mighty titanic suitable placid pleasant open lighter rutted resurgent warm highaltitude dark mysterious glossy black impenetrable grim and hostile hairy frightening lithuanian great lithuanian dense and silent dark and evil inimical great white frosty dense gloomy ageold lurid large unimproved knotted old deep odd protective true tropical mobile vast impenetrable undersea apparently impenetrable great dark more durable durable cruel and savage dark wild cool dark spacious and shady gothic inexhaustible seemingly impenetrable tame basic more ordinary improbable brandnew luridly flaming little nearby past interior coal ethereal midsized tawny mad visible unexplained dark and alien greedy deep high less and less virginal tall old completely petrified decayed and prostrate drunken fabulous possible infinitesimal transparent loose little wild almost tropical beautiful open almost virgin old primeval excellent open dense virgin almost continuous graceful distinct fairest almost pathless wild old lofty virgin vast and dismal dark inscrutable classic surprising youthful open unfenced silent vast largest unbroken open but pathless uninviting more uninviting thick primeval unceasing stern tall dark oaken littoral cheerful loveliest withered extinct serene now sufficient stiff deep and pleasant seductive deepest darkest still undulating dainty polar reddish austrian otherwise unoccupied inevitable abysmal tight nondescript charred empty long green dangerous impregnable deep impenetrable fitting open and parklike less open colossal and grotesque primeval african bad white huge and hairy dense lush nasty moderately dense comparable temperate almost universal vast great unfathomable straight memorial large but ordinary ancient dead green and golden heavily wooded airy diseased dull green remote coastal otherwise impassable everpresent mystical impossible uncaring cutoff savage and solitary gloomy and intricate dangerous mysterious directionless open grassy symmetrical fake huge tangled darker heavier striped almost primeval unusual spectacular mosscovered truly ancient deep and secret fruitful pristine oldgrowth truly unique sturdy northern shorter widespread'
            elif theme == "love":
                jjs = 'true passionate pure dear human perfect divine mutual deep ardent romantic best maternal sweet natural genuine infinite conjugial eternal intense dearest filial hopeless strong unrequited free undying faithful mere universal unselfish sincere secret paternal parental boundless happy sheer spiritual sexual disinterested everlasting fervent honest former conjugal unhappy innate dead DESCRIBING WORDS CONTINUE AFTER ADVERTISEMENT personal violent common greatest simple special deepest youthful constant abiding warm ideal own true gentle inordinate poor wonderful beautiful illicit fraternal fond noble instinctive excessive equal physical innocent sacred generous selfish selfsacrificing fierce false purest guilty profound boyish unbounded deeper enthusiastic childish wild mad great particular mighty platonic less loyal blind strange foolish charming domestic general grateful adulterous ancient unfortunate truest long kindest sensual desperate fondest immortal virtuous sentimental deathless wondrous utmost fair unconditional jealous fatal overwhelming highest inherent hearty absolute mortal anxious hot intellectual chaste affectionate certain sudden unlawful modern serious higher eager tragic steadfast silent complete slighted unchanging whole endless celestial reverent humble french past impassioned inborn lawless despairing carnal reciprocal ineffable present unspeakable extraordinary insatiable chivalrous thy true peculiar national sweetest ordinary successful heroic extreme abstract precious own dear earliest immense religious reverential powerful unconquerable hapless lifelong active least quiet immoderate sinful entire stronger patriotic real unalterable truer mysterious sure sad public selfless actual girlish oldfashioned utter open fine keen native cruel exclusive single pious savage inspiring heartfelt terrible inexpressible trusting unfailing exalted connubial fiery morbid normal fresh slow feminine social usual undivided strongest old immeasurable protective healthy respectful different obvious wholesome tough warmer especial insane watchful sordid helpless spontaneous vulgar virgin popular restless cordial impossible honorable gracious unspoken overpowering triumphant extravagant high soft singular clandestine socalled double proper evil profane exquisite german glorious unchangeable amazing unsatisfied italian unnatural feigned spanish apparent individual intelligent full erotic homosexual childlike conscious exciting possible sensuous lawful wanton poetic incestuous worthy late unconscious frantic proud evident lavish compassionate famous unsuccessful abundant oldtime furious curious undue rare thoughtful golden joyous timid sympathetic vain practical absent animal rich new oriental vivid vehement lifelong royal hungry unworthy thorough latent brief disastrous own great brotherly unfathomable primitive sufficient sublime holiest weak latest transcendent inexhaustible much more deep and abiding distant cold vast wicked metaphysical invincible wise unswerving newfound intimate incipient sovereign irresistible wellknown bitter pathetic idyllic unreasoning painful private masculine impatient young female larger manifest refined ridiculous japanese mystic criminal hereditary joyful kindred perpetual original reckless various dark responsive illfated persistent thy sweet illstarred imperfect sincerest rational fruitless unknown more unlimited ambitious incredible matchless ecstatic indomitable careless feeble partial hopeful primal zealous total obsessive idle noblest tranquil mediaeval mystical sorrowful newfound delicate habitual british lofty steady inveterate vague mature brave impartial nobler pitiful legitimate remarkable bright positive artificial impure delightful familiar irrepressible unchanged dear dear pentup unreasonable bad poor little priceless allpowerful sacred and profane white thy great plain overweening finest willing thankful fearless irish uncompromising dainty lesser impersonal unabated medi�val unceasing little more blissful characteristic second such great flaming imaginary unaffected doubtful imperishable old old angelic harmonious dangerous similar unusual radiant barbaric such passionate dumb sensitive deeprooted deep abiding good traditional chinese marital previous petty sensible supernatural further more ardent superior miserable irrational dearer onesided tremendous own sweet creative lyric almost passionate alien emotional seeming prosperous wretched pure and noble pure and perfect condescending imaginative unmistakable thine own superb indulgent delicious final due righteous marvelous grand chief numerous reluctant casual satisfying recent misguided thy dear dutiful impetuous fortunate familial veritable allconsuming perilous uninterrupted short undeserved joint nascent loose genial dormant longsuffering essential ingrained own possessive calm fantastic wider speechless spotless superhuman remorseful guileless unending twofold stupid fanatical shameful additional greedy degrading nuptial lusty dear little feverish weird subtle devotional purely spiritual peaceful vital hateful passive necessary abnormal frank conventional western deepseated obstinate transient nearer tempestuous potential promising evergrowing much less artless wistful awful such true hallowed tangled sole futile easy unparalleled unexpected devout incarnate helpful victorious loving primeval temporary hard odd inarticulate newborn sacrificial fickle next such perfect courageous fascinating rustic incomparable uncontrollable undiminished more happy happy happy heroical infernal zeal theological stern absurd incurable enlightened irregular wide charming idyllic early literary torrid decent intensive exacting pastoral careful ethereal plaintive uncritical permanent lustful low modest pure unselfish owne true own own languorous rough haunting fleeting venal own passionate little real deep true sweeter splendid potent counterfeit crazy wholehearted worshipful poetical clean limitless moral compelling terrestrial christlike regular immutable olden little true noisy matrimonial obedient fearful impious bygone thy faithful presumptuous threefold direct useless cheerful swift neverending impulsive heterosexual considerable exotic deep and passionate such deep avowed harmless imperious shameless unaltered longcherished glad vicious heartiest engrossing ceaseless quick languid broad current virginal dishonest poignant parental and filial mad passionate frenzied lazy true and perfect credulous brightest wayward clear broader cunning dutch energetic gay shy illegitimate pure and disinterested reasonable unqualified solid separate unique fairy indefatigable constitutional moderate great great mute godlike internal unrestrained unshakable cosmic interesting thrilling scant unreal proverbial fundamental pagan difficult impotent odious ferocious voluptuous vigorous greater adventurous crafty dramatic preferred artistic fairest unbroken ignorant submissive such faithful appealing welcome true and honest naive dreamy implacable immediate solemn senseless enormous endearing more dear unuttered human and divine monogamous unwise wiser primary thy constant quaint old true lower philosophical exuberant slightest untold shallow unlucky simultaneous just plain hasty tenacious loveliest elizabethan fruitful unclouded comfortable deep passionate mournful unbridled magical ready brutal inflexible customary promiscuous celtic allpervading inviolable confident everincreasing ill independent perverse red own personal classic playful stormy momentary liquid maddening finer empty true human clever charitable tireless dearest dearest extensive initial egyptian warm human older austere precocious pure and true mere human heavenly ignoble apostolic leal such wondrous maternal interested worthless happier dark secret horrible shortlived bashful sterile newer genuine romantic withered much true exceptional singleminded teutonic everyday real true more perfect troubled frivolous aspiring queer poor dear stainless pitiless admirable material fictitious grave consummate outgoing bold corresponding inner male other prodigious pathological graceful own little real true same ardent magnificent discerning shadowy barren interior sturdy coy worthier younger fonder closer frail wisest continuous safe unabashed sweet sweet notorious wrong continual true pure dull solitary diseased benevolent uncertain anguished catlike onenight profoundest dreadful indestructible deserving worth synthetic consequent lukewarm everpresent intuitive superstitious almost maternal mercenary fresh young godgiven visionary deceitful predominant servile courteous important lost suitable idealistic sticky redemptive incomprehensible fullest treacherous secondary dark and doubtful ultimate slow sweet great and pure ungovernable undisguised own deep inconceivable altruistic fanciful riotous fuller more passionate such ardent divine and human meek humorous pure disinterested liberal providential rosy piteous little human scandalous extra onetime subsequent supernal scrupulous deep and infinite heathen true true deliberate outrageous unquestionable average'
            jjs = jjs.split()
                #got from https://describingwords.io/for/tree - work on webscraping
            rhymes = [x for x in jjs if x in self.pos_to_words["JJ"]][:300]
            if verbose: print("rhymes", len(rhymes), rhymes)
            #for pos in ["NN", "ABNN", "NNS"]:
            for pos in ["JJ"]:
                self.pos_to_words[pos] = {r:1 for r in rhymes if r in self.pos_to_words[pos]}
            if len(theme.split()) > 1: rhymes.remove(theme)
            #c = Counter(rhymes)
            #sample = [k[0] for k in c.most_common(10)]
            #rhymes = helper.get_similar_word_henry(theme.lower().split(), n_return=50, word_set=set(self.words_to_pos.keys()))
            """
        else:
            rhymes = []
            theme_words = []
        #random.shuffle(rhymes)

        for p in ["NN", "NNS", "ABNN"]:
            self.pos_to_words[p] = {word:s for (word,s) in self.pos_to_words[p].items() if self.fasttext.word_similarity(word, self.theme.split()) > 0.3}
            if verbose: print("deleted", set(self.vocab_orig[p]) - set(self.pos_to_words[p]))
        self.set_meter_pos_dict()


        samples = ["\n".join(random.sample(theme_contexts, theme_lines)) if theme_lines else "" for i in range(4)] #one for each stanza
        if verbose: print("samples, ", samples)
        #rhymes = []
        #theme = None

        lines = []
        used_templates = []
        choices = []
        # first three stanzas

        self.gpt_past = ""
        line_number = 0
        while line_number < 14:
            if line_number % 4 == 0:
                if verbose: print("\n\nwriting stanza", 1 + line_number/4)
                else:
                    if line_number > 0: print("done")
                    print("\nwriting stanza", 1 + line_number/4, end=" ...")
                alliterated = not alliteration
            lines = lines[:line_number]
            used_templates = used_templates[:line_number]
            if rhyme_lines and line_number % 4 >= 2:
                r = helper.remove_punc(lines[line_number-2].split()[-1]) #last word in rhyming couplet
            elif rhyme_lines and line_number == 13:
                r = helper.remove_punc(lines[12].split()[-1])
            elif rhyme_lines and theme:
                #r = "__" + random.choice(rhymes)
                r = None #r = set(rhymes)
            else:
                r = None

            if random_templates:
                template, meter = self.get_next_template(used_templates, end=r)
                if not template:
                    if verbose: print("didnt work out for", used_templates, r)
                    continue
            else:
                template, meter = self.templates[line_number]

            #if r and len()
            alliterating = "_" not in template and not alliterated and random.random() < 0.3
            if alliterating:
                if random.random() < 0.85:
                    letters = string.ascii_lowercase
                else:
                    letters = "s"
                    #letters = string.ascii_lowercase
            else:
                letters = None


            #self.gpt_past = str(theme_lines and theme.upper() + "\n") + "\n".join(lines) #bit weird but begins with prompt if trying to be themey
            #self.gpt_past = " ".join(theme_words) + "\n" + "\n".join(lines)
            self.gpt_past = samples[0] + "\n"
            for i in range(len(lines)):
                if i % 4 == 0: self.gpt_past += samples[i//4] + "\n"
                self.gpt_past += lines[i] + "\n"
            self.reset_letter_words()
            if verbose:
                print("\nwriting line", line_number)
                print("alliterating", alliterating, letters)
                print(template, meter, r)
            line = self.write_line_gpt(template, meter, rhyme_word=r, flex_meter=True, verbose=verbose, all_verbs=all_verbs, alliteration=letters, theme_words=theme_words[theme], theme_threshold=theme_threshold)
            if line: line_arr = line.split()
            if line and rhyme_lines and not random_templates and line_number % 4 < 2:
                rhyme_pos = self.templates[min(line_number+2, 13)][0].split()[-1]
                #if any(self.rhymes(line.split()[-1], w) for w in self.get_pos_words(rhyme_pos)):
                if len(self.get_pos_words(rhyme_pos, rhyme=line.split()[-1])) > 0.001 * len(self.get_pos_words(rhyme_pos)):
                    if "a" in line_arr and line_arr[line_arr.index("a") + 1][0] in "aeiou": line = line.replace("a ", "an ")
                    if len(lines) % 4 == 0 or lines[-1][-1] in ".?!": line = line.capitalize()
                    if verbose: print("wrote line which rhymes with", rhyme_pos, ":", line)
                    #score = self.gpt.score_line("\n".join(random.sample(theme_contexts, min(len(theme_contexts), theme_lines))) + line)
                    score = self.gpt.score_line(line)
                    choices.append((score, line, template)) #scores with similarity to a random other line talking about it
                    if len(choices) == k:
                        best = min(choices)
                        if verbose: print("out of", len(choices), "chose", best)
                        lines.append(best[1])
                        used_templates.append(best[2])
                        line_number += 1
                        choices = []
                        if best[3]: alliterated = True
                else:
                    if verbose: print(line_number, "probably wasnt going to get a rhyme with", rhyme_pos)
                    #self.pos_to_words[template.split()[-1]][line.split()[-1]] /= 2
            elif line:
                if "a" in line_arr and line_arr[line_arr.index("a") + 1][0] in "aeiou": line = line.replace("a ", "an ")
                if len(lines) % 4 == 0 or lines[-1][-1] in ".?!": line = line.capitalize()
                line = line.replace(" i ", " I ")
                if verbose: print("wrote line", line)
                if len(lines) % 4 == 0:
                    choices.append((self.gpt.score_line(samples[len(lines)//4] + line), line, template, alliterating))
                else:
                    curr_stanza = "\n".join(lines[4 * len(lines)//4:])
                    choices.append((self.gpt.score_line(curr_stanza + "\n" + line), line, template, alliterating))
                if len(choices) == k:
                    best = min(choices)
                    if verbose:
                        print(choices)
                        print(line_number, ":out of", len(choices), "chose", best)
                    lines.append(best[1])
                    used_templates.append(best[2])
                    line_number += 1
                    choices = []
                    if best[3]: alliterated = True
                    last = helper.remove_punc(lines[-1].split()[-1])
                    if last in rhymes: rhymes = [r for r in rhymes if r != last]
            else:
                if verbose: print("no line", template, r)
                if random.random() < (1 / len(self.templates) * 2) * (1/k):
                    if verbose: print("so resetting randomly")
                    if line_number == 13: line_number = 12
                    else: line_number -= 2

        if not verbose: print("done")
        ret = ("         ---" + theme.upper() + "---       \n") if theme else ""
        for cand in range(len(lines)):
            ret += str(lines[cand]) + "\n"
            if ((cand + 1) % 4 == 0): ret+=("\n")
        print(ret)

        return ret


    def update_bert(self, line, meter, template, iterations, rhyme_words=[], filter_meter=True, verbose=False, choice = "min"):
        if iterations <= 0: return " ".join(line) #base case
        #TODO deal with tags like ### (which are responsible for actually cool words)
        input_ids = torch.tensor(self.tokenizer.encode(" ".join(line), add_special_tokens=False)).unsqueeze(0) #tokenizes
        tokens = [self.lang_vocab[x] for x in input_ids[0]]
        loss, outputs = self.lang_model(input_ids, masked_lm_labels=input_ids) #masks each token and gives probability for all tokens in each word. Shape num_words * vocab_size
        if verbose: print("loss = ", loss)
        softmax = torch.nn.Softmax(dim=1) #normalizes the probabilites to be between 0 and 1
        outputs = softmax(outputs[0])
        extra_token = ""
        #for word_number in range(0,len(line)-1): #ignore  last word to keep rhyme
        k = tokens.index(self.tokenizer.tokenize(line[-1])[0])  # where the last word begins

        if choice == "rand":
            word_number = out_number = random.choice(np.arange(k))

        elif choice == "min":
            probs = np.array([outputs[i][self.vocab_to_num[tokens[i]]] for i in range(k)])
            word_number = out_number = np.argmin(probs)
            while tokens[out_number].upper() in self.special_words or any(x in self.get_word_pos(tokens[out_number]) for x in ["PRP", "PRP$"]):
                if verbose: print("least likely is unchangable ", tokens, out_number, outputs[out_number][input_ids[0][out_number]])
                probs[out_number] *= 10
                word_number = out_number = np.argmin(probs)

        if len(outputs) > len(line):
            if verbose: print("before: predicting", self.lang_vocab[input_ids[0][out_number]], tokens)
            if tokens[out_number] in line:
                word_number = line.index(tokens[out_number])
                t = 1
                while "##" in self.lang_vocab[input_ids[0][out_number + t]]:
                    extra_token += self.lang_vocab[input_ids[0][out_number + 1]].split("#")[-1]
                    t += 1
                    if out_number + t >= len(input_ids[0]):
                        if verbose: print("last word chosen --> restarting", 1/0)
                        return self.update_bert(line, meter, template, iterations, rhyme_words=rhyme_words, verbose=verbose)
            else:
                sub_tokens = [self.tokenizer.tokenize(w)[0] for w in line]
                while self.lang_vocab[input_ids[0][out_number]] not in sub_tokens: out_number -= 1
                word_number = sub_tokens.index(self.lang_vocab[input_ids[0][out_number]])
                t = 1
                while "##" in self.lang_vocab[input_ids[0][out_number + t]]:
                    extra_token += self.lang_vocab[input_ids[0][word_number + t]].split("#")[-1]
                    t += 1
                    if out_number + t >= len(input_ids[0]):
                        if verbose: print("last word chosen --> restarting", 1/0)
                        return self.update_bert(line, meter, template, iterations, rhyme_words=rhyme_words, verbose=verbose)

            if verbose: print("after: ", out_number, word_number, line, " '", extra_token, "' ")

        #if verbose: print("word number ", word_number, line[word_number], template[word_number], "outnumber:", out_number)

        temp = template[word_number].split("sc")[-1]
        if len(self.get_pos_words(temp)) > 1 and temp not in ['PRP', 'PRP$']: #only change one word each time?
            filt = np.array([int( temp in self.get_word_pos(word) or temp in self.get_word_pos(word + extra_token)) for word in self.lang_vocab])
            if filter_meter and meter: filt *= np.array([int(meter[word_number] in self.get_meter(word) or meter[word_number] in self.get_meter(word + extra_token)) for word in self.lang_vocab])
            predictions = outputs[out_number].detach().numpy() * filt #filters non-words and words which dont fit meter and template

            for p in range(len(predictions)):
                if predictions[p] > 0.001 and self.lang_vocab[p] in rhyme_words:
                    print("weighting internal rhyme '", self.lang_vocab[p], "', orig: ", predictions[p], ", now: ", predictions[p]*5/sum(predictions))
                    predictions[p] *= 5
                """if predictions[p] > 0.001 and self.lang_vocab[p] in theme_words and "sc" in template[word_number]:
                    b = predictions[p]
                    input("change here and for the print")
                    predictions[p] *= theme_words[self.lang_vocab[p]]**2
                    if verbose: print("weighting thematic '", self.lang_vocab[p], "' by ", theme_words[self.lang_vocab[p]], ", now: ", predictions[p]/sum(predictions), ", was: ", b)
"""
            predictions /= sum(predictions)
            if verbose: print("predicting a ", template[word_number], meter[word_number], " for ", word_number, ". min: ", min(predictions), " max: ", max(predictions), "sum: ", sum(predictions), ", ", {self.lang_vocab[p]: predictions[p] for p in range(len(predictions)) if predictions[p] > 0})

            if iterations > 1:
                line[word_number] = np.random.choice(self.lang_vocab, p=predictions)
            else: #greedy for last iteration
                line[word_number] = self.lang_vocab[np.argmax(predictions)]

            print("word now ", line[word_number], "prob: ", predictions[self.lang_vocab.index(line[word_number])])

            if temp not in self.get_word_pos(line[word_number]):
                line[word_number] += extra_token
                if temp not in self.get_word_pos(line[word_number]): #still not valid
                    print("Extra token didnt help ", template[word_number], line[word_number], extra_token)
                    print(1/0)


            if verbose: print("line now", line)
        else:
            if verbose: print("picked ", line[word_number], "which is bad word")
            iterations += 1
        return self.update_bert(line, meter, template, iterations-1, rhyme_words=rhyme_words, verbose=verbose)

    def update_theme_words(self, word_dict={}, theme=None):
        if theme: word_dict = self.theme_gen.get_theme_words(theme)
        for pos in word_dict:
            self.pos_to_words["sc" + pos] = word_dict[pos]


    def phrase_in_poem_fast(self, words, include_syns=False):
        if type(words) == list:
            if len(words) <= 1: return True
        else:
            words = words.split()
        if len(words) > 2: return self.phrase_in_poem_fast(words[:2], include_syns=include_syns) and self.phrase_in_poem_fast(words[1:], include_syns=include_syns)
        if words[0] == words[1]: return True
        if words[0][-1] in ",.?;>" or words[1][-1] in ",.?;>": return self.phrase_in_poem_fast((words[0] + " " + words[1]).translate(str.maketrans('', '', string.punctuation)), include_syns=include_syns)
        if words[0] in self.gender: return True  # ?
        # words = " "+ words + " "

        #print("evaluating", words)

        if include_syns:
            syns = []
            for j in words:
                syns.append([l.name() for s in wn.synsets(j) for l in s.lemmas() if l.name() in self.dict_meters])
            contenders = set(words[0] + " " + w for w in syns[1])
            contenders.update([w + " " + words[1] for w in syns[0]])
            #print(words, ": " , contenders)
            return any(self.phrase_in_poem_fast(c) for c in contenders)



        if words[0] in self.surrounding_words:
            return words[1] in self.surrounding_words[words[0]]
        elif words[1] in self.surrounding_words:
            return words[0] in self.surrounding_words[words[1]]
        else:
            self.surrounding_words[words[0]] = set()
            self.surrounding_words[words[1]] = set()
            translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
            for poem in self.poems:
                poem = " " + poem.lower() + " "
                for word in words:
                    a = poem.find(word)
                    if a != -1 and poem[a-1] not in string.ascii_lowercase and poem[a+len(word)] not in string.ascii_lowercase:
                        #print(a, poem[a:a+len(word)])
                        p_words = poem.translate(translator).split() #remove punctuation and split
                        if word in p_words: #not ideal but eg a line which ends with a '-' confuses it
                            a = p_words.index(word)
                            if a - 1 >= 0 and a - 1 < len(p_words): self.surrounding_words[word].add(p_words[a-1])
                            if a + 1 >= 1 and a + 1 < len(p_words): self.surrounding_words[word].add(p_words[a+1])
            return self.phrase_in_poem_fast(words, include_syns=False)

    def get_backup_words(self, pos, meter, words_file="saved_objects/tagged_words.p"):
        if not self.backup_words:
            pc = poem_core.Poem()
            self.backup_words = pc.get_pos_words

        return [p for p in self.backup_words(pos) if meter in self.get_meter(p)]



    def close_adv(self, input, num=5, model_topn=50):
        if type(input) == str:
            positive = input.split() + ['happily']
        else:
            positive = input + ["happily"]
        negative = [       'happy']
        all_similar = self.fasttext.model.most_similar(positive, negative, topn=model_topn)

        def score(candidate):
            ratio = SequenceMatcher(None, candidate, input).ratio()
            looks_like_adv = 1.0 if candidate.endswith('ly') else 0.0
            return ratio + looks_like_adv

        close = sorted([(word, score(word)) for word, _ in all_similar], key=lambda x: -x[1])
        return [word[0] for word in close[:num]]

    def close_jj(self, input, num=5, model_topn=50):
        #positive = [input, 'dark']
        negative = [       'darkness']
        if type(input) == str:
            positive = input.split() + ['dark']
        else:
            positive = input + ["dark"]
        all_similar = self.fasttext.model.most_similar(positive, negative, topn=model_topn)
        close = [word[0] for word in all_similar if word[0] in self.pos_to_words["JJ"]]

        return close

    def close_nn(self, input, num=5, model_topn=50):
        negative = ['dark']
        if type(input) == str:
            positive = input.split() + ['darkness']
        else:
            positive = input + ["darkness"]
        all_similar = self.fasttext.model.most_similar(positive, negative, topn=model_topn)
        close = [word[0] for word in all_similar if word[0] in self.pos_to_words["NN"] or word[0] in self.pos_to_words["NNS"] or word[0] in self.pos_to_words["ABNN"]]

        return close

    def get_diff_pos(self, word, desired_pos, n=10):
        closest_words = [noun for noun in self.fasttext.get_close_words(word) if (noun in self.pos_to_words["NN"] or noun in self.pos_to_words["NNS"])]
        if desired_pos == "JJ":
            index = 0
            words = set(self.close_jj(word))
            while(len(words) < n and index < 5):
                words.update(self.close_jj(closest_words[index]))
                index += 1
            return list(words)

        if desired_pos =="RB":
            index = 0
            words = set(self.close_adv(word))
            while(len(words) < n and index < 5):
                words.update(self.close_adv(closest_words[index]))
                index += 1
            return list(words)

        if "NN" in desired_pos:
            index = 0
            words = set(self.close_nn(word))
            while(len(words) < n and index < 5):
                words.update(self.close_nn(closest_words[index]))
                index += 1
            return list(words)

        return [w for w in closest_words if desired_pos in self.get_word_pos(w)]