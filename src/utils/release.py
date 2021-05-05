
class ReleaseModel():
    def __init__(self, name, labels, extractor, model, thresholds, category_dict, color_dict):
        self.name = name
        self.labels = labels
        self.extractor = extractor
        self.model = model
        self.thresholds = thresholds
        self.category_dict = category_dict
        self.color_dict = color_dict

    def get_id(self, emotion):
        if getattr(self, 'emotion_ids', None) is None:
            self.emotion_ids = {label: i for i, label in enumerate(self.labels)}
        return self.emotion_ids[emotion]
