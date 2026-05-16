import math

from ctnet_neuro.v7_neuro.atlas import NDimensionalAtlas


def test_global_reconstruction_shape():
    atlas = NDimensionalAtlas(4)
    x = atlas.reconstruct_global()
    assert len(x) == 4


def test_local_closure_errors_finite():
    atlas = NDimensionalAtlas(4)
    errs = atlas.local_closure_errors()
    assert all(math.isfinite(x) for x in errs)


def test_coherence_bounds():
    atlas = NDimensionalAtlas(4)
    atlas.step([0.1, 0.2, 0.1, 0.0], 0.1, {"energy":0.7,"stress":0.2,"pain":0.1,"interoception":0.1,"proprioception":0.1,"autonomic_load":0.2}, {"excitation":0.2,"inhibition":0.1,"glial_modulation":0.2,"local_plasticity":0.2,"energy":0.7,"structural_signal":0.1,"patch_A":0.1,"patch_B":0.1}, {"hypotheses":4,"compatibility":0.4,"incompatibility":0.1,"grammar":0.2,"return_signal":0.2,"active_regime":"orient"}, False)
    for s in atlas.states:
        assert 0.0 <= s.local_metrics["C_total"] <= 1.0
