class Global:
    pipe = None

    def __init__(self):
        Global.pipe = None

    @classmethod
    def set_pipe(self, p):
        Global.pipe = p

    @classmethod
    def get_pipe(self):
        return Global.pipe
