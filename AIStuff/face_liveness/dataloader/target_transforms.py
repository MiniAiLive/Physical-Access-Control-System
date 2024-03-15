class Compose():

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, target):
        dst = []
        for t in self.transforms:
            dst.append(t(target))
        return dst

    def test(self):
        pass

class ClassLabel():

    def __call__(self, target):
        return target['label']

    def test(self):
        pass

class VideoID():

    def __call__(self, target):
        return target['video_id']

    def test(self):
        pass
