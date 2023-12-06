from .augment import Format, Compose


class OMFormat(Format):
    def _format_img(self, img):
        '''
        do nothing
        '''
        return img
