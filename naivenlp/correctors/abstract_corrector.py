import abc


class AbstractCorrector(abc.ABC):

    def correct(self, text, **kwargs):
        raise NotImplementedError()
