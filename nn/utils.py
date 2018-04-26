
import time

from read_utils import *
from nn.encode_utils import *


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


def compute_wer(model, batched_dict, i2p, model_type):
    wer = 0.0
    stressless_wer = 0.0
    words_num = 0
    eval_start = time.time()
    for batch in batched_dict:
        output = sess.run([model.decoded_best],
                          feed_dict=model.create_feed_dict(batch))

        orig = decode_pron(i2p, batch[2],
                           is_sparse=model_type == 'ctc',
                           with_stop_symbol=model_type != 'ctc')
        predicted = decode_pron(i2p, output,
                                is_sparse=model_type == 'ctc',
                                with_stop_symbol=model_type != 'ctc')

        words_num += len(orig)
        for o, p in zip(orig, predicted):
            oo = ' '.join(o)
            pp = ' '.join(p)
            if oo != pp:
                wer += 1
            oo = ' '.join(drop_stress(o))
            pp = ' '.join(drop_stress(p))
            if oo != pp:
                stressless_wer += 1

    wer /= float(words_num)
    stressless_wer /= float(words_num)
    eval_took = time.time() - eval_start
    return wer, stressless_wer, eval_took