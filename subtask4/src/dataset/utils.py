class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class TagMap(metaclass=Singleton):
    def __init__(self, unique_tags):
        print("creating TagMap")
        self.tag2id = {tag: id for id, tag in enumerate(unique_tags)}
        self.id2tag = {id: tag for tag, id in self.tag2id.items()}

    def save(self, filepath):
        tags = sorted(self.tag2id.keys())
        with open(filepath, 'w') as f:
            for t in tags:
                f.write(f"{t}\n")
