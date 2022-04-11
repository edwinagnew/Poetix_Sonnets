import scenery
import poem_core

from time import time

import logging
import os
from queue import Queue
from threading import Thread


#sc = scenery.Scenery_Gen(words_file="saved_objects/tagged_words.p")
sc = poem_core.Poem()

k = 5

logger = logging.getLogger(__name__)


class LineThread(Thread):

    def __init__(self, queue, sonnet_object, lines): #queue - all k templates we want to generate
        Thread.__init__(self)
        self.queue = queue

        self.sonnet_object = sonnet_object

        self.lines = lines

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            theme = self.queue.get()
            try:
                first_let = theme[0] #break when empty
                poem = self.sonnet_object.write_line_random()
                print(poem)
                self.lines.append(poem)
            finally:
                self.queue.task_done()


def main():
    ts = time()


    themes = ["love", "death", "cheese", "war", "forest"]
    # Create a queue to communicate with the worker threads
    queue = Queue()
    lines = []
    # Create 5 worker threads
    for x in range(5):
        worker = LineThread(queue, sc, lines)

        worker.start()
    # Put the tasks into the queue as a tuple
    for t in themes:
        logger.info('Queueing {}'.format(t))
        queue.put(t)
    # Causes the main thread to wait for the queue to finish processing all the tasks
    queue.join()
    logging.info('Took %s', time() - ts)
    print('Took %s', time() - ts)

    print(lines)

if __name__ == '__main__':
    main()