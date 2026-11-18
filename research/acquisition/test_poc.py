import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from poc import ROOT, Ridge, bayes_batch, features, validate_response

class Tests(unittest.TestCase):
    def test_fit_and_periodicity(self):
        a = np.linspace(-np.pi, np.pi, 80).reshape(-1,1)
        x = features(a)
        target = np.arange(x.shape[1])*.2
        m = Ridge(x, x@target, alpha=1e-8)
        np.testing.assert_allclose(m.predict(x), x@target, atol=1e-7)
        np.testing.assert_allclose(x, features(a+2*np.pi), atol=1e-12)

    def test_bayes_sequential_covariance(self):
        x = features(np.linspace(-np.pi,np.pi,32).reshape(-1,1))
        m = Ridge(x[:3], np.zeros(3))
        chosen = bayes_batch(m,x,list(range(3,32)),4)
        self.assertEqual(len(set(chosen)),4)
        cov = m.cov.copy()
        for i in chosen:
            v = cov@x[i]; cov -= np.outer(v,v)/(m.noise**2+x[i]@v)
        self.assertLess(np.trace(cov),np.trace(m.cov))
        refit = Ridge(x[list(range(3))+chosen],np.zeros(7))
        np.testing.assert_allclose(cov,refit.cov,atol=1e-10)

    def test_invalid_llm_actions(self):
        for ids in [['a','a'],['a','test_secret'],['a']]:
            with self.assertRaises(ValueError):
                validate_response({'candidate_ids':ids},['a','b'],2)

    def test_end_to_end_and_prompt_boundary(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            subprocess.run([sys.executable,str(ROOT/'poc.py'),'synthetic','--out',str(root/'data')],check=True)
            out = root/'run'
            subprocess.run([sys.executable,str(ROOT/'poc.py'),'run','--data',str(root/'data/butane.json'),
                            '--out',str(out),'--policies','random','bayes','llm','--seeds','7',
                            '--budget','16','--llm-command',json.dumps([sys.executable,str(ROOT/'mock_adapter.py')])],
                           check=True,capture_output=True)
            selected = [json.loads((out/f'{p}_7/selected_ids.json').read_text()) for p in ['random','bayes','llm']]
            self.assertTrue(all(s[:8]==selected[0][:8] for s in selected))
            state = json.loads((out/'llm_7/000_request.json').read_text())
            self.assertEqual(len(state['observations']),8)
            self.assertNotIn('test',state)
            self.assertFalse(any('qm_energy_kcal_mol' in r or 'correction_kcal_mol' in r for r in state['candidates']))
            self.assertTrue(all(r['id'].startswith('pool_') for r in state['candidates']))

if __name__=='__main__':
    unittest.main()
