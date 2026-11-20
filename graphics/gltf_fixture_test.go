package graphics

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"math"
	"testing"
)

// gltfFixture is a scene under construction for the loader tests. Its
// geometry is built by appending accessors, buffer views and nodes by hand,
// so each test can describe exactly the file it needs — a mesh mounted
// twice, a mirrored instance, a cycle — without a modelling tool.
type gltfFixture struct {
	doc  gltfJSON
	blob []byte
}

func newGLTFFixture() *gltfFixture {
	return &gltfFixture{}
}

// addAccessor appends tightly packed float32 SCALAR/VEC2/VEC3/VEC4 data and
// returns the accessor index. Values are laid out in the order given.
func (f *gltfFixture) addFloatAccessor(kind string, values []float64) int {
	comps := map[string]int{"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[kind]
	start := len(f.blob)
	for _, v := range values {
		var buf [4]byte
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(float32(v)))
		f.blob = append(f.blob, buf[:]...)
	}
	f.doc.BufferViews = append(f.doc.BufferViews, gltfBufferView{
		Buffer:     0,
		ByteOffset: start,
		ByteLength: len(values) * 4,
	})
	idx := len(f.doc.Accessors)
	f.doc.Accessors = append(f.doc.Accessors, gltfAccessor{
		BufferView:    intPtr(len(f.doc.BufferViews) - 1),
		ComponentType: compTypeFloat,
		Count:         len(values) / comps,
		Type:          kind,
	})
	return idx
}

// addIndexAccessor appends uint16 triangle indices.
func (f *gltfFixture) addIndexAccessor(indices []int) int {
	start := len(f.blob)
	for _, i := range indices {
		var buf [2]byte
		binary.LittleEndian.PutUint16(buf[:], uint16(i))
		f.blob = append(f.blob, buf[:]...)
	}
	f.doc.BufferViews = append(f.doc.BufferViews, gltfBufferView{
		Buffer:     0,
		ByteOffset: start,
		ByteLength: len(indices) * 2,
	})
	idx := len(f.doc.Accessors)
	f.doc.Accessors = append(f.doc.Accessors, gltfAccessor{
		BufferView:    intPtr(len(f.doc.BufferViews) - 1),
		ComponentType: compTypeUnsignedShort,
		Count:         len(indices),
		Type:          "SCALAR",
	})
	return idx
}

// addMesh appends a mesh with one primitive over the given accessors.
func (f *gltfFixture) addMesh(position, normal, uv, indices int, material *int) int {
	attrs := map[string]int{"POSITION": position}
	if normal >= 0 {
		attrs["NORMAL"] = normal
	}
	if uv >= 0 {
		attrs["TEXCOORD_0"] = uv
	}
	prim := gltfPrimitive{Attributes: attrs, Material: material}
	if indices >= 0 {
		prim.Indices = intPtr(indices)
	}
	f.doc.Meshes = append(f.doc.Meshes, gltfMesh{Primitives: []gltfPrimitive{prim}})
	return len(f.doc.Meshes) - 1
}

// addNode appends a node and returns its index.
func (f *gltfFixture) addNode(node gltfNode) int {
	f.doc.Nodes = append(f.doc.Nodes, node)
	return len(f.doc.Nodes) - 1
}

// addTexturedMaterial appends a 2×1 red/blue texture and a material using
// it, and returns the material index.
func (f *gltfFixture) addTexturedMaterial(t *testing.T) int {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 2, 1))
	img.Set(0, 0, color.RGBA{R: 255, A: 255})
	img.Set(1, 0, color.RGBA{B: 255, A: 255})
	var pngBuf bytes.Buffer
	if err := png.Encode(&pngBuf, img); err != nil {
		t.Fatal(err)
	}
	start := len(f.blob)
	f.blob = append(f.blob, pngBuf.Bytes()...)
	f.doc.BufferViews = append(f.doc.BufferViews, gltfBufferView{
		Buffer:     0,
		ByteOffset: start,
		ByteLength: pngBuf.Len(),
	})
	f.doc.Images = append(f.doc.Images, gltfImage{BufferView: intPtr(len(f.doc.BufferViews) - 1), MimeType: "image/png"})
	source := len(f.doc.Images) - 1
	f.doc.Textures = append(f.doc.Textures, gltfTexture{Source: &source})
	texture := len(f.doc.Textures) - 1
	f.doc.Materials = append(f.doc.Materials, gltfMaterial{
		PbrMetallicRoughness: &gltfPbrMetallicRoughness{BaseColorTexture: &gltfTextureInfo{Index: texture}},
	})
	return len(f.doc.Materials) - 1
}

// scene makes the nodes so far one scene rooted at roots.
func (f *gltfFixture) scene(roots ...int) {
	f.doc.Scenes = []gltfScene{{Nodes: roots}}
	zero := 0
	f.doc.Scene = &zero
}

// glb returns the fixture as a binary glTF payload.
func (f *gltfFixture) glb(t *testing.T) []byte {
	t.Helper()
	blob := append([]byte(nil), f.blob...)
	for len(blob)%4 != 0 {
		blob = append(blob, 0)
	}
	f.doc.Buffers = []gltfBuffer{{ByteLength: len(blob)}}

	jsonBytes, err := json.Marshal(f.doc)
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}
	for len(jsonBytes)%4 != 0 {
		jsonBytes = append(jsonBytes, ' ')
	}

	var out bytes.Buffer
	_ = binary.Write(&out, binary.LittleEndian, uint32(glbMagic))
	_ = binary.Write(&out, binary.LittleEndian, uint32(2))
	_ = binary.Write(&out, binary.LittleEndian, uint32(12+8+len(jsonBytes)+8+len(blob)))
	_ = binary.Write(&out, binary.LittleEndian, uint32(len(jsonBytes)))
	_ = binary.Write(&out, binary.LittleEndian, uint32(chunkJSON))
	out.Write(jsonBytes)
	_ = binary.Write(&out, binary.LittleEndian, uint32(len(blob)))
	_ = binary.Write(&out, binary.LittleEndian, uint32(chunkBIN))
	out.Write(blob)
	return out.Bytes()
}

// triangle is a unit right triangle in the XY plane facing +Z.
func (f *gltfFixture) addUnitTriangle(normals bool) (posAcc, normalAcc, uvAcc, idxAcc int) {
	posAcc = f.addFloatAccessor("VEC3", []float64{
		0, 0, 0,
		1, 0, 0,
		0, 1, 0,
	})
	normalAcc, uvAcc = -1, -1
	if normals {
		normalAcc = f.addFloatAccessor("VEC3", []float64{
			0, 0, 1,
			0, 0, 1,
			0, 0, 1,
		})
	}
	idxAcc = f.addIndexAccessor([]int{0, 1, 2})
	return
}

func intPtr(v int) *int { return &v }
