"""Regression checks for binary QAT, feature parity, and candidate exports."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from model import LexerConfig, NeuralLexer
from quant import QuantLinear, quantize, row_scale
import tokenizer
import export
from reference_large import Reference
from smoke_mps import CANDIDATES, mmap_member
from wgsl_large import generate

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    'x', '(){}[]', 'const MAX = 3.14; // hello\nreturn MAX;',
    '\tdef f(x):\r\n    return "escaped \\" quote"\n',
    'const 🚀🚀 = `hello ${世界}`; /* 😀 */\n',
    '/* open\n' + 'a(b[3]);\n' * 70 + '*/',
]


class ArchitectureTests(unittest.TestCase):
    def test_training_defaults_and_overrides(self):
        self.assertEqual(NeuralLexer().size_report()['packed_bytes'], 110352)
        self.assertEqual(NeuralLexer(LexerConfig()).size_report()['packed_bytes'], 102368)
        # Capture wrapper arguments before the real trainer can load a corpus.
        script = r'''
import importlib.util, importlib.abc, json, runpy, sys
sys.path.insert(0, 'train_large')
import model
original = importlib.util.spec_from_file_location
class Capture(importlib.abc.Loader):
    def create_module(self, spec): return None
    def exec_module(self, module):
        module.main = lambda: print(json.dumps(sys.argv[1:]))
def spec(name, path, *args, **kwargs):
    if name == '_lex_base_trainer':
        return importlib.util.spec_from_loader(name, Capture())
    return original(name, path, *args, **kwargs)
importlib.util.spec_from_file_location = spec
target = sys.argv[1]
sys.argv = [target, *sys.argv[2:]]
runpy.run_path(target, run_name='__main__')
'''
        for entry, overrides in [('train.py', []), ('train_teacher.py', []),
                                 ('train.py', ['--kernel-size', '5', '--no-binary-ste'])]:
            output = subprocess.check_output([sys.executable, '-c', script,
                        f'train_large/{entry}', *overrides], cwd=ROOT, text=True)
            args = json.loads(output)
            get = lambda flag: args[args.index(flag) + 1]
            self.assertEqual(get('--kernel-size'), '5' if overrides else '7')
            self.assertEqual(get('--erase-rank'), '32')
            self.assertEqual(get('--n-layers'), '4')
            if entry == 'train_teacher.py':
                self.assertIn('--full-precision', args)
                self.assertEqual(get('--dim'), '192')
                self.assertEqual(get('--weight-budget'), '0')
                self.assertEqual(get('--teacher-logits'), '')
            else:
                self.assertEqual([get(f) for f in ('--input-bits', '--head-bits', '--output-bits')], ['3', '2', '3'])
                self.assertEqual(get('--weight-budget'), '111000')
                self.assertEqual('--binary-ste' in args, not bool(overrides))

    def test_binary_forward_and_gradient(self):
        w = torch.tensor([[0.2, -0.4, 0.6, -0.8, 1.2]], requires_grad=True)
        upstream = torch.tensor([[1., 2., 3., 4., 5.]])
        scale = row_scale(w, 1)
        fixed = quantize(w, 1, scale)
        legacy = quantize(w, 1, scale, binary_ste=False)
        torch.testing.assert_close(fixed, legacy, rtol=0, atol=0)
        (fixed * upstream).sum().backward()
        torch.testing.assert_close(w.grad, torch.tensor([[1., 2., 3., 4., 0.]]))

    def test_zero_initialized_binary_projection_can_learn(self):
        layer = QuantLinear(4, 2)
        torch.nn.init.zeros_(layer.weight)
        layer(torch.tensor([[1., 2., 3., 4.]])).sum().backward()
        torch.testing.assert_close(layer.weight.grad, torch.tensor([[1., 2., 3., 4.]]).expand(2, 4))

    def test_tokenizer_matches_python(self):
        script = r'''
import { tokenize } from './lex-large/src/tokenizer.js';
let input=''; for await (const chunk of process.stdin) input+=chunk;
const result=JSON.parse(input).map(code=>{
  const t=tokenize(code), fields=[];
  for(let i=0;i<t.count;i++) {
    const [a,b,c]=t.packed.slice(i*3,i*3+3);
    fields.push({kind:a&3,len_bucket:(a>>>2)&7,first_char:(a>>>5)&127,
      last_char:(a>>>12)&127,flags:(a>>>19)&255,sym_next:a>>>27,
      hash1:b&1023,hash2:(b>>>10)&255,trans_prev:(b>>>18)&15,
      trans_next:(b>>>22)&15,sym_prev:(b>>>26)&31,
      paren_depth:c&7,brace_depth:(c>>>3)&7,bracket_depth:(c>>>6)&7,
      line_pos:(c>>>9)&7,indent_bucket:(c>>>12)&7,quote_state:(c>>>15)&3});
  }
  return {text:Array.from(t.starts,(s,i)=>code.slice(s,t.ends[i])),fields};
}); console.log(JSON.stringify(result));
'''
        result = subprocess.run(['node', '--input-type=module', '-e', script],
                                input=json.dumps(CASES), text=True, capture_output=True,
                                cwd=ROOT, check=True)
        for code, js in zip(CASES, json.loads(result.stdout)):
            tokens = tokenizer.tokenize(code)
            feats = tokenizer.tokens_to_arrays(tokens)
            self.assertEqual(js['text'], [t.text for t in tokens])
            for i, fields in enumerate(js['fields']):
                self.assertEqual(fields, {k: int(v[i]) for k, v in feats.items()})

    def test_candidate_export_and_numpy_forward(self):
        expected = dict(legacy=102368, baseline=102368, head2=107192,
                        mixed=108944, mixed_erase=109968, mixed_kernel=109328,
                        context=110352, row_head2=110550, film128=110304)
        torch.set_num_threads(2)
        for name, changes in CANDIDATES.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(20260912)
                cfg = LexerConfig(**changes)
                model = NeuralLexer(cfg).eval()
                # Exercise nontrivial FiLM, which initializes as an identity.
                with torch.no_grad():
                    model.film.strength.fill_(0.2)
                report = export.export_model(model, directory)
                self.assertEqual(report['bytes'], expected[name])
                self.assertEqual(Path(directory, 'weights.bin').stat().st_size, expected[name])
                self.assertEqual(report['meta']['config']['embedding_scale'], cfg.embedding_scale)
                restored = NeuralLexer(LexerConfig(**asdict(cfg)))
                restored.load_state_dict(model.state_dict())
                shader = generate(report['meta'])
                self.assertNotIn('__FILM_UP_DOT__', shader)
                self.assertIn('array<f32, KSIZE>', shader)
                reference = Reference(directory)
                for code in CASES[:5]:
                    arrays = tokenizer.tokens_to_arrays(tokenizer.tokenize(code))
                    features = {k: torch.from_numpy(v).unsqueeze(0) for k, v in arrays.items()}
                    with torch.no_grad():
                        actual = model(features)[0].numpy()
                    np.testing.assert_allclose(actual, reference.forward(arrays), atol=0.025, rtol=0.01)

    def test_npz_mmap_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.npz'
            expected = np.arange(32, dtype=np.int64).reshape(4, 8)
            np.savez(path, values=expected)
            np.testing.assert_array_equal(mmap_member(path, 'values'), expected)


if __name__ == '__main__':
    unittest.main()
