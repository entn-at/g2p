
import io
import argparse
import falcon
import os

from libg2p import PyG2P
from g2p_app import parse_args

html_body = '''<html><title>G2P</title>
<style>
body {padding: 16px; font-family: sans-serif; font-size: 14px; color: #444}
div {
    background: white;
    position: fixed;
    top: 50%;
    left: 50%;
    margin-top: -10%;
    margin-left: -5%;
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
<p align="center">Submit word list:</p>
<input type="file" id="word_list" accept="text/*" size="30">
<p align="center">or</p>
<input id="text" type="text" size="30" placeholder="Enter word" style="margin-top:5px">
<p align="center"><button id="button" name="phonetisize" style="margin-top:10px">Phonetisize</button></p>
<p align="center" id="message"></p>
</div>
<footer align="center">
  <p>bicuser470@gmail.com <a href="https://github.com/bic-user/g2p">github</a></p>
</footer>
<script>
function q(selector) {return document.querySelector(selector)}
q('#button').addEventListener('click', function(e) {
  text = q('#text').value.trim()
  path = q('#word_list').value.trim()
  console.log(text)
  console.log(path)
  if (path) {
    q('#message').textContent = 'Generating pronunciation for word list...'
    q('#button').disabled = true
    phonetisize_batch(q('#word_list').files[0])
  }
  if (text) {
    q('#message').textContent = 'Generating pronunciation for word...'
    q('#button').disabled = true
    phonetisize(text)
  }
  e.preventDefault()
  return false
})
function phonetisize(text) {
  fetch('/phonetisize?text=' + encodeURIComponent(text), {cache: 'no-cache'})
    .then(function(res) {
      if (!res.ok) throw Error(res.statusText)
      return res.text()
    }).then(function(pron) {
      q('#message').textContent = pron
      q('#button').disabled = false
    }).catch(function(err) {
      q('#message').textContent = 'Error: ' + err.message
      q('#button').disabled = false
    })
}
function phonetisize_batch(file) {
  fetch('/phonetisize', { method: 'POST', body:file }, { responseType: 'blob' })
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
    if not req.params.get('text'):
      raise falcon.HTTPBadRequest()
    pron = g2p.Phonetisize(req.params.get('text'))
    if not pron:
      res.body = '**Error! Invalid input or empty pronunciation'
    else:
      res.body = ' '.join(pron)
  def on_post(self, req, res):
      words = req.stream.read().decode().split('\n')
      if not words[-1]:
          words = words[:-1]
      outfp = io.BytesIO()
      for w in words:
          pron = g2p.Phonetisize(w)
          outfp.write(('%s\t%s\n' % (w, ' '.join(pron))).encode())
      res.data = outfp.getvalue()


g2p = None
api = falcon.API()
api.add_route('/phonetisize', PhonetisizeResource())
api.add_route('/', UIResource())


if __name__ == '__main__':
    from wsgiref import simple_server
    args = parse_args()
    g2p = PyG2P(args.nn, args.nn_meta, args.fst, args.dict)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    simple_server.make_server('0.0.0.0', 80, api).serve_forever()
