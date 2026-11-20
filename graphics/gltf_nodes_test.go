package graphics

import (
	"math"
	"strings"
	"testing"
)

// oneTriGLB is a GLB whose single mesh is mounted at one node with the given
// transform; hasTransform false leaves the node with no TRS at all.
func oneTriGLB(t *testing.T, node gltfNode) []byte {
	t.Helper()
	f := newGLTFFixture()
	pos, _, _, idx := f.addUnitTriangle(false)
	mesh := f.addMesh(pos, -1, -1, idx, nil)
	node.Mesh = intPtr(mesh)
	f.addNode(node)
	f.scene(0)
	return f.glb(t)
}

func verticesOf(model Model3D) []Vertex3D { return model.Vertices }

func assertVertex(t *testing.T, label string, got, want Vertex3D) {
	t.Helper()
	// The scene graph is described in float64, so a chain of transforms
	// accumulates a little float noise; 1e-6 relative to the units used is
	// far tighter than any drawing decision.
	if math.Abs(got.X-want.X) > 1e-6 || math.Abs(got.Y-want.Y) > 1e-6 || math.Abs(got.Z-want.Z) > 1e-6 {
		t.Errorf("%s: got (%g, %g, %g), want (%g, %g, %g)", label, got.X, got.Y, got.Z, want.X, want.Y, want.Z)
	}
}

// A model with no node transforms is drawn exactly as its accessors store
// it: the node layer must not shift, scale or reorient anything.
func TestGLBNodeWithoutTransformStaysAtOrigin(t *testing.T) {
	model, err := ParseGLB(oneTriGLB(t, gltfNode{}))
	if err != nil {
		t.Fatal(err)
	}
	want := []Vertex3D{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}
	if len(model.Vertices) != 3 {
		t.Fatalf("%d vertices, want 3", len(model.Vertices))
	}
	for i, v := range model.Vertices {
		assertVertex(t, "origin", v, want[i])
	}
}

// A child's translation is added to the parent's, down the chain: the
// composition is parent * child, not either alone.
func TestGLBNodeTranslationComposesDownTheHierarchy(t *testing.T) {
	f := newGLTFFixture()
	pos, _, _, idx := f.addUnitTriangle(false)
	mesh := f.addMesh(pos, -1, -1, idx, nil)

	child := f.addNode(gltfNode{Mesh: intPtr(mesh), Translation: []float64{0, 2, 0}})
	parent := f.addNode(gltfNode{Children: []int{child}, Translation: []float64{3, 0, 0}})
	f.scene(parent)

	model, err := ParseGLB(f.glb(t))
	if err != nil {
		t.Fatal(err)
	}
	assertVertex(t, "origin", model.Vertices[0], Vertex3D{3, 2, 0})
	assertVertex(t, "unit x", model.Vertices[1], Vertex3D{4, 2, 0})
	assertVertex(t, "unit y", model.Vertices[2], Vertex3D{3, 3, 0})
}

// A parent's rotation turns its child's offset as well as the child's mesh.
func TestGLBRotatedParentRotatesChildTranslation(t *testing.T) {
	f := newGLTFFixture()
	pos, _, _, idx := f.addUnitTriangle(false)
	mesh := f.addMesh(pos, -1, -1, idx, nil)

	// 90° about Z: (1, 0, 0) becomes (0, 1, 0). Half angle quaternion.
	half := math.Sqrt2 / 2
	child := f.addNode(gltfNode{Mesh: intPtr(mesh), Translation: []float64{2, 0, 0}})
	parent := f.addNode(gltfNode{Children: []int{child}, Rotation: []float64{0, 0, half, half}})
	f.scene(parent)

	model, err := ParseGLB(f.glb(t))
	if err != nil {
		t.Fatal(err)
	}
	// The child sits at (0, 2, 0) after the parent's turn; its own unit
	// triangle now points along +Y, so the X edge becomes a Y edge.
	assertVertex(t, "origin", model.Vertices[0], Vertex3D{0, 2, 0})
	assertVertex(t, "unit x", model.Vertices[1], Vertex3D{0, 3, 0})
	assertVertex(t, "unit y", model.Vertices[2], Vertex3D{-1, 2, 0})
}

// A scene-tree file with two nodes sharing one mesh produces two copies at
// their own positions. This is the bug the issue reports: both instances
// used to land on the origin because the mesh was emitted once, untransformed.
func TestGLBSharedMeshIsDrawnOncePerNode(t *testing.T) {
	f := newGLTFFixture()
	pos, normal, _, idx := f.addUnitTriangle(true)
	mesh := f.addMesh(pos, normal, -1, idx, nil)

	left := f.addNode(gltfNode{Mesh: intPtr(mesh), Translation: []float64{-4, 0, 0}})
	right := f.addNode(gltfNode{Mesh: intPtr(mesh), Translation: []float64{4, 0, 0}})
	root := f.addNode(gltfNode{Children: []int{left, right}})
	f.scene(root)

	model, err := ParseGLB(f.glb(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(model.Vertices) != 6 || len(model.Faces) != 2 {
		t.Fatalf("%d vertices, %d faces; want 6 and 2 (two instances)", len(model.Vertices), len(model.Faces))
	}
	if len(model.VertexNormals) != 6 {
		t.Fatalf("%d vertex normals, want 6", len(model.VertexNormals))
	}
	assertVertex(t, "left origin", model.Vertices[0], Vertex3D{-4, 0, 0})
	assertVertex(t, "right origin", model.Vertices[3], Vertex3D{4, 0, 0})
	if model.Faces[0][0] != 0 || model.Faces[1][0] != 3 {
		t.Errorf("faces point at vertices %v and %v; each instance needs its own", model.Faces[0], model.Faces[1])
	}
}

// Non-uniform and negative scales are applied as they stand. A mirroring
// transform reverses triangle winding, so the loader swaps two corners to
// keep front faces facing the viewer. Normals use the inverse transpose, so
// a normal stays perpendicular to its surface under a squashed scale.
func TestGLBNegativeAndNonUniformScale(t *testing.T) {
	t.Run("non-uniform", func(t *testing.T) {
		model, err := ParseGLB(oneTriGLB(t, gltfNode{Scale: []float64{2, 3, 1}}))
		if err != nil {
			t.Fatal(err)
		}
		assertVertex(t, "unit x", model.Vertices[1], Vertex3D{2, 0, 0})
		assertVertex(t, "unit y", model.Vertices[2], Vertex3D{0, 3, 0})
	})

	t.Run("negative mirrors winding", func(t *testing.T) {
		f := newGLTFFixture()
		pos, normal, _, idx := f.addUnitTriangle(true)
		mesh := f.addMesh(pos, normal, -1, idx, nil)
		f.addNode(gltfNode{Mesh: intPtr(mesh), Scale: []float64{-1, 1, 1}})
		f.scene(0)

		model, err := ParseGLB(f.glb(t))
		if err != nil {
			t.Fatal(err)
		}
		assertVertex(t, "unit x", model.Vertices[1], Vertex3D{-1, 0, 0})
		// Original winding (0,1,2) is counter-clockwise; mirroring across X
		// makes it clockwise, so the loader must emit (0,2,1).
		if len(model.Faces) != 1 {
			t.Fatalf("%d faces, want 1", len(model.Faces))
		}
		x0, x1, x2 := model.Vertices[model.Faces[0][0]], model.Vertices[model.Faces[0][1]], model.Vertices[model.Faces[0][2]]
		a := Vector3D{X: x1.X - x0.X, Y: x1.Y - x0.Y, Z: x1.Z - x0.Z}
		b := Vector3D{X: x2.X - x0.X, Y: x2.Y - x0.Y, Z: x2.Z - x0.Z}
		if cross := a.Cross(b); cross.Z <= 0 {
			t.Errorf("mirrored face still winds counter-clockwise (cross.Z=%g); it would be culled", cross.Z)
		}
	})
}

// A normal on a slanted surface must stay perpendicular after a
// non-uniform scale, which only the inverse transpose guarantees. Scaling by
// (2, 1, 1) takes the plane x+y=0 to x/2+y=0, whose normal is (1/2, 1, 0);
// transforming the normal like a point would give (2, 1, 0) — a different
// direction, and visibly wrong shading.
func TestGLBNormalUsesInverseTranspose(t *testing.T) {
	f := newGLTFFixture()
	// A slanted quad along the line x = -y in the XY plane.
	pos := f.addFloatAccessor("VEC3", []float64{
		0, 0, 0,
		1, -1, 0,
		0, 0, 1,
	})
	inv2 := 1 / math.Sqrt2
	normal := f.addFloatAccessor("VEC3", []float64{
		inv2, inv2, 0,
		inv2, inv2, 0,
		inv2, inv2, 0,
	})
	idx := f.addIndexAccessor([]int{0, 1, 2})
	mesh := f.addMesh(pos, normal, -1, idx, nil)
	f.addNode(gltfNode{Mesh: intPtr(mesh), Scale: []float64{2, 1, 1}})
	f.scene(0)

	model, err := ParseGLB(f.glb(t))
	if err != nil {
		t.Fatal(err)
	}
	want := Vector3D{X: 0.5, Y: 1, Z: 0}.Normalize()
	for i, n := range model.VertexNormals {
		if math.Abs(n.X-want.X) > 1e-6 || math.Abs(n.Y-want.Y) > 1e-6 || math.Abs(n.Z-want.Z) > 1e-6 {
			t.Fatalf("normal %d = %+v, want %+v (the inverse transpose, not the linear map (2,1,0))", i, n, want)
		}
	}
	// And the transformed normal is still perpendicular to the face.
	p0, p1, p2 := model.Vertices[0], model.Vertices[1], model.Vertices[2]
	face := Vector3D{X: p1.X - p0.X, Y: p1.Y - p0.Y, Z: p1.Z - p0.Z}
	edge := Vector3D{X: p2.X - p0.X, Y: p2.Y - p0.Y, Z: p2.Z - p0.Z}
	if dot := model.VertexNormals[0].Dot(face.Cross(edge).Normalize()); math.Abs(math.Abs(dot)-1) > 1e-6 {
		t.Errorf("normal is not the face's normal after scaling (unit dot=%.2g, want ±1)", dot)
	}
}

// A glTF may store its transform as a column-major 16-value matrix instead
// of TRS; it must be applied with the same point convention.
func TestGLBNodeMatrixTransform(t *testing.T) {
	// Column-major: translating by (5, 6, 7) writes the translation in the
	// last column, elements 12, 13, 14.
	matrix := make([]float64, 16)
	matrix[0], matrix[5], matrix[10], matrix[15] = 1, 1, 1, 1
	matrix[12], matrix[13], matrix[14] = 5, 6, 7

	model, err := ParseGLB(oneTriGLB(t, gltfNode{Matrix: matrix}))
	if err != nil {
		t.Fatal(err)
	}
	assertVertex(t, "origin", model.Vertices[0], Vertex3D{5, 6, 7})
	assertVertex(t, "unit x", model.Vertices[1], Vertex3D{6, 6, 7})
	assertVertex(t, "unit y", model.Vertices[2], Vertex3D{5, 7, 7})
}

// With no scenes the nodes that are nobody's child are the roots, so a
// hand-written file still draws through its whole hierarchy.
func TestGLBWithoutScenesFallsBackToRootNodes(t *testing.T) {
	f := newGLTFFixture()
	pos, _, _, idx := f.addUnitTriangle(false)
	mesh := f.addMesh(pos, -1, -1, idx, nil)
	child := f.addNode(gltfNode{Mesh: intPtr(mesh), Translation: []float64{1, 1, 0}})
	f.addNode(gltfNode{Children: []int{child}})

	model, err := ParseGLB(f.glb(t))
	if err != nil {
		t.Fatal(err)
	}
	assertVertex(t, "origin", model.Vertices[0], Vertex3D{1, 1, 0})
}

// The selected scene is the one drawn; nodes outside it do not appear.
func TestGLBOnlyTheSelectedSceneIsDrawn(t *testing.T) {
	f := newGLTFFixture()
	pos, _, _, idx := f.addUnitTriangle(false)
	mesh := f.addMesh(pos, -1, -1, idx, nil)
	inScene := f.addNode(gltfNode{Mesh: intPtr(mesh), Translation: []float64{2, 0, 0}})
	outOfScene := f.addNode(gltfNode{Mesh: intPtr(mesh), Translation: []float64{-2, 0, 0}})
	f.doc.Scenes = []gltfScene{{Nodes: []int{inScene}}, {Nodes: []int{outOfScene}}}
	one := 1
	f.doc.Scene = &one

	model, err := ParseGLB(f.glb(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(model.Vertices) != 3 {
		t.Fatalf("%d vertices, want the 3 of scene 1 only", len(model.Vertices))
	}
	assertVertex(t, "origin", model.Vertices[0], Vertex3D{-2, 0, 0})
}

func TestGLBNodeErrors(t *testing.T) {
	t.Run("child cycle", func(t *testing.T) {
		f := newGLTFFixture()
		a := f.addNode(gltfNode{})
		b := f.addNode(gltfNode{Children: []int{a}})
		f.doc.Nodes[a].Children = []int{b}
		f.scene(b)
		_, err := ParseGLB(f.glb(t))
		if err == nil || !strings.Contains(err.Error(), "cycle") {
			t.Fatalf("cycle error: %v", err)
		}
	})

	t.Run("self cycle", func(t *testing.T) {
		f := newGLTFFixture()
		a := f.addNode(gltfNode{Children: []int{0}})
		f.scene(a)
		_, err := ParseGLB(f.glb(t))
		if err == nil || !strings.Contains(err.Error(), "cycle") {
			t.Fatalf("self cycle error: %v", err)
		}
	})

	t.Run("dangling child index", func(t *testing.T) {
		f := newGLTFFixture()
		a := f.addNode(gltfNode{Children: []int{7}})
		f.scene(a)
		_, err := ParseGLB(f.glb(t))
		if err == nil || !strings.Contains(err.Error(), "child index") {
			t.Fatalf("dangling child error: %v", err)
		}
	})

	t.Run("dangling mesh index", func(t *testing.T) {
		f := newGLTFFixture()
		f.addNode(gltfNode{Mesh: intPtr(9)})
		f.scene(0)
		_, err := ParseGLB(f.glb(t))
		if err == nil || !strings.Contains(err.Error(), "mesh index") {
			t.Fatalf("dangling mesh error: %v", err)
		}
	})

	t.Run("dangling scene index", func(t *testing.T) {
		f := newGLTFFixture()
		pos, _, _, idx := f.addUnitTriangle(false)
		mesh := f.addMesh(pos, -1, -1, idx, nil)
		f.addNode(gltfNode{Mesh: intPtr(mesh)})
		f.doc.Scenes = []gltfScene{{Nodes: []int{0}}}
		bad := 4
		f.doc.Scene = &bad
		_, err := ParseGLB(f.glb(t))
		if err == nil || !strings.Contains(err.Error(), "scene index") {
			t.Fatalf("dangling scene error: %v", err)
		}
	})

	t.Run("scene lists a dangling node", func(t *testing.T) {
		f := newGLTFFixture()
		f.doc.Scenes = []gltfScene{{Nodes: []int{3}}}
		zero := 0
		f.doc.Scene = &zero
		f.addNode(gltfNode{})
		_, err := ParseGLB(f.glb(t))
		if err == nil || !strings.Contains(err.Error(), "node index") {
			t.Fatalf("dangling scene node error: %v", err)
		}
	})

	t.Run("zero scale", func(t *testing.T) {
		_, err := ParseGLB(oneTriGLB(t, gltfNode{Scale: []float64{0, 1, 1}}))
		if err == nil || !strings.Contains(err.Error(), "zero") {
			t.Fatalf("zero scale error: %v", err)
		}
	})

	t.Run("zero scale inherited from a parent", func(t *testing.T) {
		f := newGLTFFixture()
		pos, _, _, idx := f.addUnitTriangle(false)
		mesh := f.addMesh(pos, -1, -1, idx, nil)
		child := f.addNode(gltfNode{Mesh: intPtr(mesh)})
		parent := f.addNode(gltfNode{Children: []int{child}, Scale: []float64{1, 0, 1}})
		f.scene(parent)
		_, err := ParseGLB(f.glb(t))
		if err == nil || !strings.Contains(err.Error(), "zero") {
			t.Fatalf("inherited zero scale error: %v", err)
		}
	})

	t.Run("non-finite matrix", func(t *testing.T) {
		// JSON cannot carry NaN or Inf as a float64 here, so the check is
		// driven directly: a matrix that reaches it non-finite is refused.
		matrix := make([]float64, 16)
		matrix[0], matrix[15] = math.NaN(), 1
		if _, err := gltfNodeTransform(gltfNode{Matrix: matrix}); err == nil || !strings.Contains(err.Error(), "non-finite") {
			t.Fatalf("NaN matrix error: %v", err)
		}
		matrix[0] = math.Inf(1)
		if _, err := gltfNodeTransform(gltfNode{Matrix: matrix}); err == nil || !strings.Contains(err.Error(), "non-finite") {
			t.Fatalf("Inf matrix error: %v", err)
		}
	})

	t.Run("non-finite translation", func(t *testing.T) {
		_, err := gltfNodeTransform(gltfNode{Translation: []float64{math.Inf(1), 0, 0}})
		if err == nil || !strings.Contains(err.Error(), "non-finite") {
			t.Fatalf("Inf translation error: %v", err)
		}
		_, err = gltfNodeTransform(gltfNode{Rotation: []float64{math.NaN(), 0, 0, 1}})
		if err == nil || !strings.Contains(err.Error(), "non-finite") {
			t.Fatalf("NaN rotation error: %v", err)
		}
		_, err = gltfNodeTransform(gltfNode{Scale: []float64{1, math.Inf(-1), 1}})
		if err == nil || !strings.Contains(err.Error(), "non-finite") {
			t.Fatalf("Inf scale error: %v", err)
		}
	})

	t.Run("wrong length transform", func(t *testing.T) {
		_, err := ParseGLB(oneTriGLB(t, gltfNode{Matrix: []float64{1, 0, 0}}))
		if err == nil || !strings.Contains(err.Error(), "16") {
			t.Fatalf("short matrix error: %v", err)
		}
		_, err = ParseGLB(oneTriGLB(t, gltfNode{Translation: []float64{1, 2}}))
		if err == nil || !strings.Contains(err.Error(), "3") {
			t.Fatalf("short translation error: %v", err)
		}
	})

	t.Run("zero quaternion", func(t *testing.T) {
		_, err := ParseGLB(oneTriGLB(t, gltfNode{Rotation: []float64{0, 0, 0, 0}}))
		if err == nil || !strings.Contains(err.Error(), "quaternion") {
			t.Fatalf("zero quaternion error: %v", err)
		}
	})
}

// A hierarchy deeper than the limit is reported, not recursed into.
func TestGLBNodeDepthLimit(t *testing.T) {
	f := newGLTFFixture()
	pos, _, _, idx := f.addUnitTriangle(false)
	mesh := f.addMesh(pos, -1, -1, idx, nil)

	// A chain of maxGLTFNodeDepth+2 nodes: the deepest one exceeds the cap.
	var child = -1
	for i := 0; i <= maxGLTFNodeDepth+1; i++ {
		node := gltfNode{Mesh: intPtr(mesh), Translation: []float64{0, 0, 0}}
		if child >= 0 {
			node.Children = []int{child}
		}
		child = f.addNode(node)
	}
	f.scene(child)

	_, err := ParseGLB(f.glb(t))
	if err == nil || !strings.Contains(err.Error(), "deeper") {
		t.Fatalf("depth error: %v", err)
	}
}

// The limit is a limit, not an off-by-one: a chain right at it loads.
func TestGLBNodeDepthAtTheLimitLoads(t *testing.T) {
	f := newGLTFFixture()
	pos, _, _, idx := f.addUnitTriangle(false)
	mesh := f.addMesh(pos, -1, -1, idx, nil)

	var child = -1
	for i := 0; i < maxGLTFNodeDepth; i++ { // depth 0 .. max-1: allowed
		node := gltfNode{Translation: []float64{1, 0, 0}}
		if child >= 0 {
			node.Children = []int{child}
		}
		child = f.addNode(node)
	}
	// The deepest node is the first one created; scene(child) roots the last.
	f.doc.Nodes[0].Mesh = intPtr(mesh)
	f.scene(child)

	model, err := ParseGLB(f.glb(t))
	if err != nil {
		t.Fatalf("nesting at the limit failed: %v", err)
	}
	if len(model.Vertices) != 3 {
		t.Fatalf("%d vertices", len(model.Vertices))
	}
	// Each level adds its own translation, so the origin ends at the count.
	assertVertex(t, "origin", model.Vertices[0], Vertex3D{maxGLTFNodeDepth, 0, 0})
}

// An indexed primitive whose indices point past its vertices skips those
// triangles rather than emitting faces that read out of range at draw time.
func TestGLBIndicesOutOfRangeAreSkipped(t *testing.T) {
	f := newGLTFFixture()
	pos := f.addFloatAccessor("VEC3", []float64{0, 0, 0, 1, 0, 0, 0, 1, 0})
	idx := f.addIndexAccessor([]int{0, 1, 2, 0, 1, 99})
	mesh := f.addMesh(pos, -1, -1, idx, nil)
	f.addNode(gltfNode{Mesh: intPtr(mesh)})
	f.scene(0)

	model, err := ParseGLB(f.glb(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(model.Faces) != 1 {
		t.Fatalf("%d faces, want only the in-range triangle", len(model.Faces))
	}
}

// A mesh with one texture, mounted twice, keeps its own UVs: both instances
// sample the image, so both are red on the left and blue on the right.
func TestGLBSharedTexturedMeshKeepsUVsPerInstance(t *testing.T) {
	f := newGLTFFixture()
	pos := f.addFloatAccessor("VEC3", []float64{
		0, 0, 0,
		1, 0, 0,
		0, 1, 0,
	})
	uv := f.addFloatAccessor("VEC2", []float64{0, 0, 0, 0, 0, 0}) // all sample the red texel
	idx := f.addIndexAccessor([]int{0, 1, 2})
	mat := f.addTexturedMaterial(t)
	mesh := f.addMesh(pos, -1, uv, idx, &mat)

	a := f.addNode(gltfNode{Mesh: intPtr(mesh), Translation: []float64{-2, 0, 0}})
	b := f.addNode(gltfNode{Mesh: intPtr(mesh), Translation: []float64{2, 0, 0}})
	f.scene(a, b)

	model, err := ParseGLB(f.glb(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(model.Faces) != 2 || len(model.FaceUVs) != 2 {
		t.Fatalf("%d faces, %d face UVs; want 2 each", len(model.Faces), len(model.FaceUVs))
	}
	if model.Texture == nil {
		t.Fatal("texture missing")
	}
	// Each FaceUVs entry indexes this instance's own copy of the UVs.
	if got := model.FaceUVs[0][0]; got != 0 {
		t.Errorf("first instance's UV index %d, want 0", got)
	}
	if got := model.FaceUVs[1][0]; got != 3 {
		t.Errorf("second instance's UV index %d, want 3 (its own copy)", got)
	}
	for i, c := range model.FaceColors {
		r, g, bl := c.RGB()
		if r < 200 || bl > 40 {
			t.Errorf("face %d colour = (%d,%d,%d), want red", i, r, g, bl)
		}
	}
}

// Reading the transforms must accept the real duck: its root carries a
// 0.01 matrix and the mesh hangs off a child, so the geometry lands far
// from the raw accessor values.
func TestGLBDuckUsesItsRootMatrix(t *testing.T) {
	model, err := LoadGLB("../examples/ascii3d/duck.glb")
	if err != nil {
		t.Fatal(err)
	}
	if len(model.Vertices) != 2399 {
		t.Fatalf("%d vertices, want 2399", len(model.Vertices))
	}
	// The raw positions span ~165 units; the 0.01 root matrix brings that
	// down to roughly 1.65. Without the transform the extent would be ~165.
	minX, maxX := model.Vertices[0].X, model.Vertices[0].X
	for _, v := range model.Vertices {
		if v.X < minX {
			minX = v.X
		}
		if v.X > maxX {
			maxX = v.X
		}
	}
	if span := maxX - minX; span > 3 || span < 0.5 {
		t.Errorf("X extent %.3f: the root matrix was not applied", span)
	}
}

// A primitive without NORMAL leaves VertexNormals nil, so renderers keep
// deriving normals from faces as they do for OBJ and STL.
func TestGLBPrimitiveWithoutNormalsHasNoVertexNormals(t *testing.T) {
	model, err := ParseGLB(oneTriGLB(t, gltfNode{}))
	if err != nil {
		t.Fatal(err)
	}
	if model.VertexNormals != nil {
		t.Errorf("got %d vertex normals for a primitive with none", len(model.VertexNormals))
	}
}
