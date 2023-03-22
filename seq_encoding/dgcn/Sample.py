class Sample:
    """
    The text data contains a list of utterances, which has been tokenized and converted to word embeddings of size 100
    """

    def __init__(self, vid, speaker, label, text, audio, visual, sentence):
        self.vid = vid
        self.speaker = speaker
        self.label = label
        self.text = text
        self.audio = audio
        self.visual = visual
        self.sentence = sentence
