package graphics

import (
	"math"
	"testing"
)

// The fixture from the issue: one mesh, two nodes, a rotated parent. The
// whole file is loaded and the instances are checked where they land, both
// through the loader and through the same projection the viewers use.
func TestGLBSceneTreeFixtureSharedMeshUnderRotatedParent(t *testing.T) {
	f := newGLTFFixture()
	// A unit triangle lying in the XY plane with an explicit +Z normal, so
	// it can be rotated about Y and stay inspectable.
	pos := f.addFloatAccessor("VEC3", []float64{
		0, 0, 0,
		1, 0, 0,
		0, 1, 0,
	})
	normal := f.addFloatAccessor("VEC3", []float64{0, 0, 1, 0, 0, 1, 0, 0, 1})
	uv := f.addFloatAccessor("VEC2", []float64{0, 0, 1, 0, 0, 1})
	idx := f.addIndexAccessor([]int{0, 1, 2})
	mesh := f.addMesh(pos, normal, uv, idx, nil)

	// Two children of a parent turned 90° about Y. The children are offset
	// along X, so the parent's turn moves them onto the Z axis: the same
	// shared mesh must appear at (0, 0, -3) and (0, 0, +3), not both at the
	// origin and not both on X.
	left := f.addNode(gltfNode{Mesh: intPtr(mesh), Translation: []float64{-3, 0, 0}})
	right := f.addNode(gltfNode{Mesh: intPtr(mesh), Translation: []float64{3, 0, 0}})
	half := math.Sqrt2 / 2
	parent := f.addNode(gltfNode{Children: []int{left, right}, Rotation: []float64{0, half, 0, half}})
	f.scene(parent)

	model, err := ParseGLB(f.glb(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(model.Vertices) != 6 || len(model.Faces) != 2 {
		t.Fatalf("%d vertices and %d faces from two instances, want 6 and 2", len(model.Vertices), len(model.Faces))
	}
	if len(model.VertexNormals) != 6 {
		t.Fatalf("%d normal, want 6", len(model.VertexNormals))
	}

	// 90° about Y sends -3·X to +3·Z (this rotation's convention) and
	// +3·X to -3·Z.
	assertVertex(t, "left instance", model.Vertices[0], Vertex3D{0, 0, 3})
	assertVertex(t, "right instance", model.Vertices[3], Vertex3D{0, 0, -3})
	if model.Vertices[0].X == model.Vertices[3].X && model.Vertices[0].Z == model.Vertices[3].Z {
		t.Fatal("both instances landed on the same point: the shared mesh was not placed per node")
	}

	// The +Z normal of each instance turns to +X after the parent's 90° Y
	// rotation, and the shared mesh therefore has the same normal at both.
	for i, n := range model.VertexNormals {
		if math.Abs(n.X-1) > 1e-9 || math.Abs(n.Y) > 1e-9 || math.Abs(n.Z) > 1e-9 {
			t.Errorf("normal %d = %+v, want (1, 0, 0) after the parent's rotation", i, n)
		}
	}

	// The projection the viewers use puts them at different depths. A 90°
	// turn about Y lays the triangle in the YZ plane, so both instances see
	// the same X column, but one is nearer the camera than the other and
	// projects larger. Landing at the same Z was the bug.
	depth := func(first int) (z float64) {
		for _, v := range model.Vertices[first : first+3] {
			z += v.Z
		}
		return z / 3
	}
	height := func(first int) float64 {
		minY, maxY := math.Inf(1), math.Inf(-1)
		for _, v := range model.Vertices[first : first+3] {
			_, y, visible := Project(v, 80, 40, 6, 60)
			if !visible {
				continue
			}
			minY, maxY = math.Min(minY, y), math.Max(maxY, y)
		}
		return maxY - minY
	}
	if math.Abs(depth(0)-depth(3)) < 1e-6 {
		t.Fatalf("both instances sit at z=%.3f: they were not placed apart", depth(0))
	}
	// Each instance projects to its own vertical line, and the nearer one
	// is the taller on screen.
	if math.Abs(height(0)-height(3)) < 1e-6 {
		t.Errorf("both instances project %.3f columns tall; they are not at different depths", height(0))
	}
}

// The other GLB assets in the repository have no scene transform worth
// moving (limoni.glb has a single node with no TRS). Loading them must
// reproduce the same vertex count and the same extent as the raw accessors,
// so the node layer changes nothing for untransformed files.
func TestGLBUntransformedAssetsAreUnchanged(t *testing.T) {
	model, err := LoadGLB("../examples/demo/limoni.glb")
	if err != nil {
		t.Fatal(err)
	}
	if len(model.Vertices) != 78459 {
		t.Fatalf("%d vertices, want the accessor's 78459", len(model.Vertices))
	}
	minX, maxX := model.Vertices[0].X, model.Vertices[0].X
	minY, maxY := model.Vertices[0].Y, model.Vertices[0].Y
	minZ, maxZ := model.Vertices[0].Z, model.Vertices[0].Z
	for _, v := range model.Vertices {
		minX, maxX = math.Min(minX, v.X), math.Max(maxX, v.X)
		minY, maxY = math.Min(minY, v.Y), math.Max(maxY, v.Y)
		minZ, maxZ = math.Min(minZ, v.Z), math.Max(maxZ, v.Z)
	}
	// The accessor's own min/max, to float32 precision.
	for _, c := range []struct {
		name             string
		gotMin, gotMax   float64
		wantMin, wantMax float64
	}{
		{"X", minX, maxX, -0.4131726026535034, 0.4154924750328064},
		{"Y", minY, maxY, -0.44089555740356445, 0.4407694339752197},
		{"Z", minZ, maxZ, -0.5001043081283569, 0.5022504329681396},
	} {
		if math.Abs(c.gotMin-c.wantMin) > 1e-5 || math.Abs(c.gotMax-c.wantMax) > 1e-5 {
			t.Errorf("%s extent %.7f..%.7f, want %.7f..%.7f", c.name, c.gotMin, c.gotMax, c.wantMin, c.wantMax)
		}
	}
}
