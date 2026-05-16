import math

from ctnet_neuro.v7_neuro.atlas import NDimensionalAtlas


def test_global_reconstruction_shape():
    atlas = NDimensionalAtlas(4, dim=6)
    x = atlas.reconstruct_global()
    assert len(x) == 6


def test_local_closure_errors_finite():
    atlas = NDimensionalAtlas(4, dim=6)
    errs = atlas.local_closure_errors()
    assert all(math.isfinite(x) for x in errs)


def test_coherence_bounds():
    atlas = NDimensionalAtlas(4, dim=6)
    atlas.step([0.1, 0.2, 0.1, 0.0], 0.1, {"energy":0.7,"stress":0.2,"pain":0.1,"interoception":0.1,"proprioception":0.1,"autonomic_load":0.2}, {"excitation":0.2,"inhibition":0.1,"glial_modulation":0.2,"local_plasticity":0.2,"energy":0.7,"structural_signal":0.1,"patch_A":0.1,"patch_B":0.1}, {"hypotheses":4,"compatibility":0.4,"incompatibility":0.1,"grammar":0.2,"return_signal":0.2,"active_regime":"orient"}, False)
    for s in atlas.states:
        assert 0.0 <= s.local_metrics["C_total"] <= 1.0


def test_remote_node_perturbation_changes_global_state():
    atlas = NDimensionalAtlas(4, dim=6)
    body={"energy":0.7,"stress":0.2,"pain":0.1,"interoception":0.1,"proprioception":0.1,"autonomic_load":0.2}
    tissue={"excitation":0.2,"inhibition":0.1,"glial_modulation":0.2,"local_plasticity":0.2,"energy":0.7,"structural_signal":0.1,"patch_A":0.1,"patch_B":0.1}
    ctx={"hypotheses":4,"compatibility":0.4,"incompatibility":0.1,"grammar":0.2,"return_signal":0.2,"active_regime":"orient"}
    atlas.step([0.1,0.1,0.1,0.1],0.2,body,tissue,ctx,False)
    x1=atlas.reconstruct_global()
    atlas.states[3].dynamic=[v+1.5 for v in atlas.states[3].dynamic]
    atlas.step([0.1,0.1,0.1,0.1],0.2,body,tissue,ctx,False)
    x2=atlas.reconstruct_global()
    assert any(abs(a-b)>1e-6 for a,b in zip(x1,x2))
