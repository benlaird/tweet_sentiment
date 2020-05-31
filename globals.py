class Global:
    pipe = None
    kaggle_submission = False
    input_dir = None
    output_dir = None

    def __init__(self):
        Global.pipe = None
        Global.kaggle_submission = False
        if Global.kaggle_submission:
            Global.input_dir = "/kaggle/input/tweet-sentiment-extraction/"
            Global.output_dir = "/kaggle/working/"
        else:
            Global.input_dir = "./data/tweet-sentiment-extraction/"
            Global.output_dir = "./"

    @classmethod
    def set_pipe(self, p):
        Global.pipe = p

    @classmethod
    def get_pipe(self):
        return Global.pipe
