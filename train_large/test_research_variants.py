"""Small correctness checks for the isolated research experiment graph."""
from dataclasses import replace
import tempfile
import unittest
import torch
from model import NeuralLexer, student_config
import export
from research_variants import ResearchLexer
from quant import QuantLinear, QuantEmbedding
import tokenizer


class ResearchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(2)
        arrays = tokenizer.tokens_to_arrays(tokenizer.tokenize('const x = "hello"; // test\n'))
        self.feats = {k: torch.from_numpy(v).unsqueeze(0) for k, v in arrays.items()}

    def test_single_pass_matches_production(self):
        production = NeuralLexer()
        research = ResearchLexer(student_config())
        research.load_state_dict(production.state_dict())
        torch.testing.assert_close(research(self.feats), production(self.feats), rtol=0, atol=0)

    def test_repeated_depth_costs_no_extra_weights_and_learns(self):
        model = ResearchLexer(student_config(), repeats=2)
        self.assertEqual(model.size_report()['packed_bytes'], 110352)
        model(self.feats).square().mean().backward()
        self.assertGreater(float(model.layers[0].proj_in.weight.grad.abs().sum()), 0)

    def test_float_diagnostic_stays_unquantized(self):
        model = ResearchLexer(student_config(), full_precision=True)
        model.set_quant(True)
        self.assertTrue(all(not m.quant_enabled for m in model.modules()
                            if isinstance(m, (QuantLinear, QuantEmbedding))))

    def test_rms_clipping_export_and_precision_budget(self):
        model = ResearchLexer(student_config(), embedding_rms_clip=True)
        with tempfile.TemporaryDirectory() as directory:
            report = export.export_model(model, directory)
            self.assertEqual(report['bytes'], 110352)
        narrow = ResearchLexer(replace(student_config(), dim=64, proj_bits=2,
                                       embed_bits=4, film_rank=32, erase_rank=16))
        self.assertEqual(narrow.size_report()['packed_bytes'], 108496)
        self.assertTrue(torch.isfinite(narrow(self.feats)).all())


if __name__ == '__main__':
    unittest.main()
