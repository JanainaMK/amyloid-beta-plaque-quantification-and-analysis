class ConfusionMatrix:
    def __init__(self, TP=0, FP=0, TN=0, FN=0):
        self.TP = TP
        self.FP = FP
        self.TN = TN
        self.FN = FN

    def __add__(self, other):
        return ConfusionMatrix(
            self.TP + other.TP,
            self.FP + other.FP,
            self.TN + other.TN,
            self.FN + other.FN,
        )

    def __len__(self):
        return self.TP + self.FP + self.TN + self.FN

    def __str__(self):
        return f"TP: {self.TP} FN: {self.FN}\n" f"FP: {self.FP} TN: {self.TN}"

    def precision(self):
        if self.TP == 0:
            return 1 if self.FP == 0 else 0
        else:
            return self.TP / (self.TP + self.FP)

    def recall(self):
        if self.TP == 0:
            return 1 if self.FN == 0 else 0
        else:
            return self.TP / (self.TP + self.FN)
