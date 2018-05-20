
import io
import argparse
import falcon
import json
import os

from libg2p import PyG2P

html_body = '''<html><title>G2P</title>
<style>
body {padding: 16px; font-family: sans-serif; font-size: 14px; color: #444}
div {
    background: white;
    position: relative;
    width: 100%;
    height: 100%;
}
input {font-size: 14px; padding: 8px 12px; outline: none; border: 1px solid #ddd}
p {padding: 12px}
button {background: #28d; padding: 9px 14px; border: none; outline: none;
        color: #fff; font-size: 14px; border-radius: 4px; cursor: pointer}
button:hover {box-shadow: 0 1px 2px rgba(0,0,0,.15); opacity: 0.9;}
button:active {background: #29f;}
button[disabled] {opacity: 0.4; cursor: default}
footer {
background-color: #FFF;
position:fixed;
bottom: 0px;
width: 100%;
text-align: center;
}
</style>
<body>
<div>

<p id="languages">
<input type="radio" name="lang" id="en_us" value="en_us" checked>
<label for="en_us">en_us</label>
<input type="radio" name="lang" id="en_gb" value="en_gb" style="margin-left:20px">
<label for="en_gb">en_gb</label>
<input type="radio" name="lang" id="ru" value="ru" style="margin-left:20px">
<label for="other">ru</label>
</p>

<p style="margin-top:10px">
<button id="button_graphemes" name="graphemes">Graphemes</button>
<button id="button_phonemes" name="phonemes">Phonemes</button>
</p>

<p style="margin-left:60px">Submit word list:</p>
<input type="file" id="word_list" accept="text/*" size="30">
<p style="margin-left:100px">or</p>
<input id="text" type="text" size="30" placeholder="Enter word" style="margin-top:5px">
<p><button id="button" name="phonetisize" style="margin-left:60px" style="margin-top:10px">Phonetisize</button></p>
<p id="message"></p>
</div>
<footer align="center">
  <p>bicuser470@gmail.com <a href="https://github.com/bic-user/g2p">github</a></p>
</footer>
<script>
function q(selector) {return document.querySelector(selector)}

q('#button_graphemes').addEventListener('click', function(e) {
    lang = get_lang()
    window.location = '/graphemes?lang=' + lang
})
q('#button_phonemes').addEventListener('click', function(e) {
    lang = get_lang()
    window.location = '/phonemes?lang=' + lang
})
q('#button').addEventListener('click', function(e) {
  text = q('#text').value.trim()
  path = q('#word_list').value.trim()
  lang = get_lang()
  if (path) {
    q('#message').textContent = 'Generating pronunciation for word list...'
    q('#button').disabled = true
    file = q('#word_list').files[0]
    if(file.size > 1048576) {
        q('#message').textContent = 'Provided file is too big'
        q('#word_list').value = ''
        q('#button').disabled = false
    } else {
        phonetisize_batch(file, lang)
    }
  }
  if (text) {
    q('#message').textContent = 'Generating pronunciation for word...'
    q('#button').disabled = true
    phonetisize(text, lang)
  }
  e.preventDefault()
  return false
})
function get_lang() {
  lang = ""
  if (q('#en_us').checked) {
    lang = "en_us"
  } else if (q('#en_gb').checked) {
    lang = "en_gb"
  } else if (q('#ru').checked) {
    lang = "ru"
  }
  return lang
}
function phonetisize(text, lang) {
  fetch('/phonetisize?lang=' + lang + '&text=' + encodeURIComponent(text), {cache: 'no-cache'})
    .then(function(res) {
      if (!res.ok) throw Error(res.statusText)
      return res.text()
    }).then(function(pron) {
      q('#message').textContent = q('#text').value.trim() + ' ' + pron
      q('#text').value = ''
      q('#button').disabled = false
    }).catch(function(err) {
      q('#message').textContent = 'Error: ' + err.message
      q('#button').disabled = false
      q('#text').value = ''
    })
}
function phonetisize_batch(file, lang) {
  fetch('/phonetisize?lang=' + lang, { method: 'POST', body:file }, { responseType: 'blob' })
    .then(function(res){
      res.blob().then(function(blob) {
        console.log(blob)
        let a = document.createElement("a");
        a.style = "display: none";
        document.body.appendChild(a);
        let url = window.URL.createObjectURL(blob);
        console.log(url)
        a.href = url;
        a.download = 'dict';
        a.click();
        window.URL.revokeObjectURL(url);
      });
      q('#button').disabled = false
      q('#word_list').value = ''
      q('#message').textContent = ''
    }).catch(function(err) {
      q('#message').textContent = 'Error: ' + err.message
      q('#button').disabled = false
    })
}
</script></body></html>
'''


class UIResource:
  def on_get(self, req, res):
    res.content_type = 'text/html'
    res.body = html_body


class PhonetisizeResource:

  def on_get(self, req, res):
    if not req.params.get('lang'):
      raise falcon.HTTPBadRequest('**Error! Language code is not provided')
    if not req.params.get('text'):
      raise falcon.HTTPBadRequest('**Error! Word for phonetisation is not provided')
    lang = req.params.get('lang')
    if lang not in g2p:
      raise falcon.HTTPBadRequest('**Error! Language is not supported')
    pron = g2p[lang].Phonetisize([req.params.get('text')])[0]
    if not pron:
      raise falcon.HTTPBadRequest('**Error! Invalid input or empty pronunciation')
    else:
      res.body = ' '.join(pron)

  def on_post(self, req, res):
      if not req.params.get('lang'):
        raise falcon.HTTPBadRequest('Language code is not provided')
      lang = req.params.get('lang')
      if lang not in g2p:
        raise falcon.HTTPBadRequest('Language is not supported')
      words = req.stream.read().decode().split('\n')
      if not words[-1]:
          words = words[:-1]
      outfp = io.BytesIO()
      pron = g2p[lang].Phonetisize(words)
      for w, p in zip(words, pron):
          outfp.write(('%s\t%s\n' % (w, ' '.join(p))).encode())
      res.data = outfp.getvalue()


class GraphemesResource():
    def on_get(self, req, res):
        if not req.params.get('lang'):
            raise falcon.HTTPBadRequest('Language code is not provided')
        lang = req.params.get('lang')
        if lang not in g2p:
            raise falcon.HTTPBadRequest('Language is not supported')

        res.content_type = 'text/html'
        line = '</br>'.join(g2p[lang].GetGraphemes())
        res.body = '<html><title>G2P graphemes</title><meta charset="utf-8"/><body></body>%s</html>' % line


class PhonemesResource():
    def on_get(self, req, res):
        if not req.params.get('lang'):
            raise falcon.HTTPBadRequest('Language code is not provided')
        lang = req.params.get('lang')
        if lang not in g2p:
            raise falcon.HTTPBadRequest('Language is not supported')

        res.content_type = 'text/html'
        line = '</br>'.join(g2p[lang].GetPhonemes())
        res.body = '<html><title>G2P phonemes</title><meta charset="utf-8"/><body></body>%s</html>' % line


g2p = {}
api = falcon.API()
api.add_route('/phonetisize', PhonetisizeResource())
api.add_route('/graphemes', GraphemesResource())
api.add_route('/phonemes', PhonemesResource())
api.add_route('/', UIResource())


def parse_args():
    arg_parser = argparse.ArgumentParser(description='Parses args for g2p server')
    arg_parser.add_argument('--config', required=True, help='Path to json with models config')
    arg_parser.add_argument('--port', default=80, type=int)
    args = arg_parser.parse_args()
    if not os.path.isfile(args.config):
        raise RuntimeError('**Error! Failed to open %s' % args.config)
    return args


if __name__ == '__main__':
    from wsgiref import simple_server
    args = parse_args()
    with open(args.config, 'r') as infp:
        config = json.load(infp)
    for lang in config.keys():
        py_g2p_args = []
        for name in ['nn', 'nn_meta', 'fst', 'dict']:
            if config[lang][name]:
                if not os.path.isfile(config[lang][name]):
                    raise RuntimeError('**Error! Cant open %s' % config[lang][name])
            py_g2p_args.append(config[lang][name])
        g2p[lang] = PyG2P(*py_g2p_args)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    simple_server.make_server('0.0.0.0', args.port, api).serve_forever()
