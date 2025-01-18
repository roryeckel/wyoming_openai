from io import BytesIO

class NamedBytesIO(BytesIO):
    def __init__(self, *args, **kwargs):
        self._name = kwargs.pop('name', 'audio.wav')
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return self._name