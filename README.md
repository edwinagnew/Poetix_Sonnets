# Poetix_Sonnets

## Duke Poetry Team

Welcome!

**Setup**
Some things you should do to get this working: (if you get stuck on anything please ask me for help)
1. Github 
  - Make a github account (if you don't have one)
  - Assuming you're on mac, open Terminal and navigate to where you want to store the project
    - Enter `ls` to see where you can navigate to and `cd some_folder` to navigate to some_folder (`cd ..` to go back)
  - Enter `git clone https://github.com/edwinagnew/Poetix_Sonnets.git`
2. [Download and Install Anaconda](https://docs.anaconda.com/anaconda/install/mac-os/)
3. Download an IDE, [PyCharm](https://www.jetbrains.com/pycharm/promo/anaconda/) recommended (free version of course)
4. After everything has installed, restart terminal and it should say (base) at the beginning of the line. To install the necessary packages:
  - Navigate back to where you cloned the project
  - Type `conda create -n poetry python=3.7` to create an environmnet name poetry
  - Type `conda activate poetry`. It should now say (poetry) at the beginning of the line
  - Type `conda install --file requirements.txt` and feel like a hacker while the screen goes crazy
  - Type `python -m spacy download en_core_web_lg`. This might take a few minutes

**Run Code**
1. Open the project in your IDE
2. If you're using pycharm, make sure you have selected your conda environment (poetry in example) and it says Python 3.7 (poetry) in the bottom right.
3. For a 'typical' sonnet:
  - Click on Python Console at the bottom
  - Type `import sonnet_basic`
  - `sonnet = sonnet_basic.Sonnet_Gen()` If you want to use your own templates write `sonnet = sonnet_basic.Sonnet_Gen(templates_file="path_to_file.txt")`
  - `sonnet.gen_poem_edwin('love')` you can replace love with any prompt word you like. The first time you use any prompt word will take longer since it finds and stores related words for future use.
  - It will print a poem and some info along the way
 4. For a jabberwocky -ish poem:
  - Click on Python Console at the bottom
  - `import sonnet_jabber`
  - `jabber = sonnet_jabber.Sonnet_Gen()`
  - `poem = jabber.gen_poem_jabber()`
  - To then add a narrative:
    - `jabber.get_heros_theme(poem)` for Hero's journey
    - OR `jabber.get_summary_theme(poem, x)` where x is any number from 0 to 37159503 and corresponds to which book summary you want.
