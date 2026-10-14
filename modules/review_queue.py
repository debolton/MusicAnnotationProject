class ReviewQueue:

    def __init__(self, cfg):
        self.cfg = cfg

    def reviewed_hashes(self):
        ...

    def next_page_number(self):
        ...

    def build_queue(self):
        ...

    def refresh(self):
        ...
