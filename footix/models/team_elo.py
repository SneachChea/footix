class team:
    def __init__(self, name: str):
        self.name_ = name
        self.rank_ = 0.0

    @property
    def name(self):
        return self.name_

    @property
    def rank(self):
        return self.rank_

    @rank.setter
    def rank(self, new_rank):
        if isinstance(new_rank, float):
            self.rank_ = new_rank
        else:
            print("Enter a valid rank, i.e. a float")

    def __str__(self):
        return f"team {self.name}"

    def __repr__(self):
        return str(self)
