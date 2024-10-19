from .modeling_unimo import UnimoForMaskedLM


class UnimoKGC(UnimoForMaskedLM):
    @staticmethod
    def add_to_argparse(parser):
        #print("function = add_to_argparse , class = UnimoKGC , file = model.py")
        parser.add_argument("--pretrain", type=int, default=0, help="")
        return parser
