
import argparse
import falcon
import os

from libg2p import PyG2P
from g2p_app import parse_args

html_body = '''<html><title>G2P</title>
<style>
body {padding: 16px; font-family: sans-serif; font-size: 14px; color: #444}
input {font-size: 14px; padding: 8px 12px; outline: none; border: 1px solid #ddd}
input:focus {box-shadow: 0 1px 2px rgba(0,0,0,.15)}
p {padding: 12px}
button {background: #28d; padding: 9px 14px; margin-left: 8px; border: none; outline: none;
        color: #fff; font-size: 14px; border-radius: 4px; cursor: pointer;}
button:hover {box-shadow: 0 1px 2px rgba(0,0,0,.15); opacity: 0.9;}
button:active {background: #29f;}
button[disabled] {opacity: 0.4; cursor: default}
</style>
<body>
<form>
  <input id="text" type="text" size="40" placeholder="Enter Text">
  <button id="button" name="phonetisize">Phonetisize</button>
</form>
<p id="message"></p>
<script>
function q(selector) {return document.querySelector(selector)}
q('#text').focus()
q('#button').addEventListener('click', function(e) {
  text = q('#text').value.trim()
  if (text) {
    q('#message').textContent = 'Generating...'
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
      res.body = '**Error! Invalid input'
    else:
      res.body = ' '.join(pron)

g2p = None
api = falcon.API()
api.add_route('/phonetisize', PhonetisizeResource())
api.add_route('/', UIResource())


if __name__ == '__main__':
    from wsgiref import simple_server
    args = parse_args()
    g2p = PyG2P(args.nn, args.nn_meta, args.fst, args.dict)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    simple_server.make_server('0.0.0.0', 9000, api).serve_forever()
