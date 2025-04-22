import util
from .base_arg_parser import BaseArgParser


class TestArgParser(BaseArgParser):
    def __init__(self):
        super(TestArgParser, self).__init__()
        self.is_training = False
