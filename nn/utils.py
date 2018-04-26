
def import_model_type(model_type):
    if args.model_type == 'ctc':
        from nn.ctc.hparams import hparams
        from nn.ctc.model import G2PModel
    elif args.model_type == 'attention':
        from nn.attention.hparams import hparams
        from nn.attention.model import G2PModel
    elif args.model_type == 'transformer':
        from nn.transformer.hparams import hparams
        from nn.transformer.model import G2PModel
    return G2PModel, hparams
