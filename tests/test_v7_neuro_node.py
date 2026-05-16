import math

from ctnet_neuro.v7_neuro.atlas import NDimensionalAtlas


def _body():
    return {"energy":0.7,"stress":0.2,"pain":0.1,"interoception":0.1,"proprioception":0.1,"autonomic_load":0.2}

def _tissue():
    return {"excitation":0.2,"inhibition":0.1,"glial_modulation":0.2,"local_plasticity":0.2,"energy":0.7,"structural_signal":0.1,"patch_A":0.1,"patch_B":0.1}

def _context():
    return {"hypotheses":4,"compatibility":0.4,"incompatibility":0.1,"grammar":0.2,"return_signal":0.2,"active_regime":"orient"}


def test_global_reconstruction_shape():
    atlas = NDimensionalAtlas(4)
    x = atlas.reconstruct_global()
    assert len(x) == atlas.nodes[0].DYNAMIC_DIM


def test_local_closure_errors_finite():
    atlas = NDimensionalAtlas(4)
    errs = atlas.local_closure_errors()
    assert all(math.isfinite(x) for x in errs)


def test_coherence_bounds():
    atlas = NDimensionalAtlas(4)
    atlas.step([0.1, 0.2, 0.1, 0.0], 0.1, _body(), _tissue(), _context(), False)
    for s in atlas.states:
        assert 0.0 <= s.local_metrics["C_total"] <= 1.0


def test_remote_perturbation_changes_global():
    atlas = NDimensionalAtlas(4)
    atlas.step([0.1, 0.2, 0.1, 0.0], 0.1, _body(), _tissue(), _context(), False)
    before = atlas.reconstruct_global()[:]
    atlas.states[-1].dynamic = [x + 0.9 for x in atlas.states[-1].dynamic]
    atlas.step([0.1, 0.2, 0.1, 0.0], 0.1, _body(), _tissue(), _context(), False)
    after = atlas.reconstruct_global()
    assert any(abs(a-b) > 1e-6 for a,b in zip(before, after))
