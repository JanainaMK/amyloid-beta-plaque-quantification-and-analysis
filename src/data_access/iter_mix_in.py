class IterMixIn:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        try:
            res = self[self.i]
            self.i += 1
            return res
        except IndexError:
            raise StopIteration
            