from io import BytesIO
import pickle


def bytes_to_file(bytes_img):
    return BytesIO(bytes_img)


class PickledImageProvider(object):
    def __init__(self, obj_path, labels=None):
        self.obj_path = obj_path
        self.labels = labels
        self.examples = self.load_pickled_examples()

    def load_pickled_examples(self):
        with open(self.obj_path, "rb") as of:
            examples = list()
            labelSet = None
            if self.labels is not None: labelSet = set(self.labels)
            while True:
                try:
                    e = pickle.load(of)
                    if labelSet is None: examples.append(e)
                    else:
                        if e[0] in labelSet: examples.append(e)
                    if len(examples) % 100000 == 0:
                        print("processed %d examples" % len(examples))
                except EOFError:
                    break
                except Exception:
                    pass
            print("unpickled total %d examples" % len(examples))
            return examples
