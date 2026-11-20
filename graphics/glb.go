package graphics

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/thebanri/limoni/core/cell"
)

const (
	glbMagic  = 0x46546C67 // "glTF"
	chunkJSON = 0x4E4F534A // "JSON"
	chunkBIN  = 0x004E4942 // "BIN\0"

	compTypeUnsignedByte  = 5121
	compTypeUnsignedShort = 5123
	compTypeUnsignedInt   = 5125
	compTypeFloat         = 5126
)

type gltfJSON struct {
	Scene       *int             `json:"scene,omitempty"`
	Scenes      []gltfScene      `json:"scenes,omitempty"`
	Nodes       []gltfNode       `json:"nodes,omitempty"`
	Meshes      []gltfMesh       `json:"meshes,omitempty"`
	Accessors   []gltfAccessor   `json:"accessors,omitempty"`
	BufferViews []gltfBufferView `json:"bufferViews,omitempty"`
	Buffers     []gltfBuffer     `json:"buffers,omitempty"`
	Materials   []gltfMaterial   `json:"materials,omitempty"`
	Textures    []gltfTexture    `json:"textures,omitempty"`
	Images      []gltfImage      `json:"images,omitempty"`
}

type gltfScene struct {
	Name  string `json:"name,omitempty"`
	Nodes []int  `json:"nodes,omitempty"`
}

type gltfNode struct {
	Name        string    `json:"name,omitempty"`
	Mesh        *int      `json:"mesh,omitempty"`
	Children    []int     `json:"children,omitempty"`
	Matrix      []float64 `json:"matrix,omitempty"`      // column-major 4×4, the whole local transform
	Translation []float64 `json:"translation,omitempty"` // T of a TRS transform
	Rotation    []float64 `json:"rotation,omitempty"`    // quaternion x, y, z, w
	Scale       []float64 `json:"scale,omitempty"`       // S of a TRS transform
}

type gltfMesh struct {
	Name       string          `json:"name,omitempty"`
	Primitives []gltfPrimitive `json:"primitives,omitempty"`
}

type gltfPrimitive struct {
	Attributes map[string]int `json:"attributes"`
	Indices    *int           `json:"indices,omitempty"`
	Material   *int           `json:"material,omitempty"`
	Mode       *int           `json:"mode,omitempty"` // default 4 = TRIANGLES
}

type gltfMaterial struct {
	Name                 string                    `json:"name,omitempty"`
	PbrMetallicRoughness *gltfPbrMetallicRoughness `json:"pbrMetallicRoughness,omitempty"`
}

type gltfPbrMetallicRoughness struct {
	BaseColorFactor  []float64        `json:"baseColorFactor,omitempty"`
	BaseColorTexture *gltfTextureInfo `json:"baseColorTexture,omitempty"`
	MetallicFactor   *float64         `json:"metallicFactor,omitempty"`
	RoughnessFactor  *float64         `json:"roughnessFactor,omitempty"`
}

type gltfTextureInfo struct {
	Index int `json:"index"`
}

type gltfTexture struct {
	Source *int `json:"source,omitempty"`
}

type gltfImage struct {
	BufferView *int   `json:"bufferView,omitempty"`
	MimeType   string `json:"mimeType,omitempty"`
	URI        string `json:"uri,omitempty"`
}

type gltfAccessor struct {
	BufferView    *int      `json:"bufferView,omitempty"`
	ByteOffset    int       `json:"byteOffset,omitempty"`
	ComponentType int       `json:"componentType"`
	Count         int       `json:"count"`
	Type          string    `json:"type"` // "SCALAR", "VEC2", "VEC3", etc.
	Max           []float64 `json:"max,omitempty"`
	Min           []float64 `json:"min,omitempty"`
}

type gltfBufferView struct {
	Buffer     int `json:"buffer"`
	ByteOffset int `json:"byteOffset,omitempty"`
	ByteLength int `json:"byteLength"`
	ByteStride int `json:"byteStride,omitempty"`
}

type gltfBuffer struct {
	ByteLength int    `json:"byteLength"`
	URI        string `json:"uri,omitempty"`
}

// LoadGLB loads a binary glTF 2.0 (.glb) file and resolves textures and geometry.
func LoadGLB(path string) (Model3D, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Model3D{}, err
	}
	model, err := ParseGLB(data)
	if err != nil {
		return Model3D{}, fmt.Errorf("parse GLB %q: %w", path, err)
	}
	model.Name = path
	return model, nil
}

// ParseGLB parses binary glTF 2.0 data from a byte slice.
func ParseGLB(data []byte) (Model3D, error) {
	if len(data) < 12 {
		return Model3D{}, fmt.Errorf("invalid GLB: header too short (%d bytes)", len(data))
	}

	magic := binary.LittleEndian.Uint32(data[0:4])
	if magic != glbMagic {
		return Model3D{}, fmt.Errorf("invalid GLB magic: 0x%08X (expected 0x%08X)", magic, glbMagic)
	}

	version := binary.LittleEndian.Uint32(data[4:8])
	if version != 2 {
		return Model3D{}, fmt.Errorf("unsupported GLB version: %d (expected 2)", version)
	}

	totalLength := binary.LittleEndian.Uint32(data[8:12])
	if int(totalLength) > len(data) {
		return Model3D{}, fmt.Errorf("GLB length header (%d) exceeds file size (%d)", totalLength, len(data))
	}

	offset := 12
	var jsonChunk []byte
	var binChunk []byte

	for offset+8 <= len(data) {
		chunkLen := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
		chunkType := binary.LittleEndian.Uint32(data[offset+4 : offset+8])
		offset += 8

		if offset+chunkLen > len(data) {
			return Model3D{}, fmt.Errorf("chunk extends beyond GLB payload")
		}

		chunkData := data[offset : offset+chunkLen]
		offset += chunkLen

		switch chunkType {
		case chunkJSON:
			jsonChunk = chunkData
		case chunkBIN:
			binChunk = chunkData
		}
	}

	if len(jsonChunk) == 0 {
		return Model3D{}, fmt.Errorf("GLB missing JSON chunk")
	}

	var doc gltfJSON
	if err := json.Unmarshal(jsonChunk, &doc); err != nil {
		return Model3D{}, fmt.Errorf("decode GLB JSON chunk: %w", err)
	}

	buffers := [][]byte{binChunk}
	return buildModelFromGLTF(&doc, buffers, "")
}

// LoadGLTF loads a text/JSON glTF (.gltf) file and resolves referenced binary buffers and images.
func LoadGLTF(path string) (Model3D, error) {
	file, err := os.Open(path)
	if err != nil {
		return Model3D{}, err
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		return Model3D{}, err
	}

	var doc gltfJSON
	if err := json.Unmarshal(data, &doc); err != nil {
		return Model3D{}, fmt.Errorf("decode glTF JSON %q: %w", path, err)
	}

	baseDir := filepath.Dir(path)
	buffers := make([][]byte, len(doc.Buffers))
	for i, bufDef := range doc.Buffers {
		if strings.HasPrefix(bufDef.URI, "data:") {
			idx := strings.Index(bufDef.URI, ",")
			if idx == -1 {
				return Model3D{}, fmt.Errorf("invalid data URI in buffer %d", i)
			}
			raw, err := base64.StdEncoding.DecodeString(bufDef.URI[idx+1:])
			if err != nil {
				return Model3D{}, fmt.Errorf("decode base64 buffer %d: %w", i, err)
			}
			buffers[i] = raw
		} else if bufDef.URI != "" {
			bufPath := filepath.Join(baseDir, bufDef.URI)
			raw, err := os.ReadFile(bufPath)
			if err != nil {
				return Model3D{}, fmt.Errorf("read glTF buffer %q: %w", bufPath, err)
			}
			buffers[i] = raw
		}
	}

	model, err := buildModelFromGLTF(&doc, buffers, path)
	if err != nil {
		return Model3D{}, fmt.Errorf("build glTF model %q: %w", path, err)
	}
	model.Name = path
	return model, nil
}

type uvCoord struct {
	u, v float64
}

func buildModelFromGLTF(doc *gltfJSON, buffers [][]byte, sourcePath string) (Model3D, error) {
	model := Model3D{
		Name:      sourcePath,
		Materials: make(map[string]Material3D),
	}

	baseDir := ""
	if sourcePath != "" {
		baseDir = filepath.Dir(sourcePath)
	}

	getAccessorBytes := func(accIdx int) ([]byte, gltfAccessor, int, error) {
		if accIdx < 0 || accIdx >= len(doc.Accessors) {
			return nil, gltfAccessor{}, 0, fmt.Errorf("accessor index %d out of range", accIdx)
		}
		acc := doc.Accessors[accIdx]
		if acc.BufferView == nil {
			return nil, acc, 0, fmt.Errorf("accessor %d has no bufferView", accIdx)
		}
		bvIdx := *acc.BufferView
		if bvIdx < 0 || bvIdx >= len(doc.BufferViews) {
			return nil, acc, 0, fmt.Errorf("bufferView index %d out of range", bvIdx)
		}
		bv := doc.BufferViews[bvIdx]
		if bv.Buffer < 0 || bv.Buffer >= len(buffers) {
			return nil, acc, 0, fmt.Errorf("buffer index %d out of range", bv.Buffer)
		}
		bufData := buffers[bv.Buffer]
		start := bv.ByteOffset + acc.ByteOffset
		end := start + bv.ByteLength
		if start < 0 || start > len(bufData) {
			return nil, acc, 0, fmt.Errorf("bufferView offset out of range")
		}
		if end > len(bufData) {
			end = len(bufData)
		}
		stride := bv.ByteStride
		return bufData[start:end], acc, stride, nil
	}

	// Decode all embedded or referenced images
	decodedImages := make([]image.Image, len(doc.Images))
	for i, imgDef := range doc.Images {
		if imgDef.BufferView != nil && *imgDef.BufferView >= 0 && *imgDef.BufferView < len(doc.BufferViews) {
			bv := doc.BufferViews[*imgDef.BufferView]
			if bv.Buffer >= 0 && bv.Buffer < len(buffers) {
				bufData := buffers[bv.Buffer]
				start := bv.ByteOffset
				end := start + bv.ByteLength
				if start >= 0 && end <= len(bufData) {
					if img, _, err := image.Decode(bytes.NewReader(bufData[start:end])); err == nil {
						decodedImages[i] = img
					}
				}
			}
		} else if imgDef.URI != "" {
			if strings.HasPrefix(imgDef.URI, "data:") {
				idx := strings.Index(imgDef.URI, ",")
				if idx != -1 {
					if raw, err := base64.StdEncoding.DecodeString(imgDef.URI[idx+1:]); err == nil {
						if img, _, err := image.Decode(bytes.NewReader(raw)); err == nil {
							decodedImages[i] = img
						}
					}
				}
			} else if baseDir != "" {
				imgPath := filepath.Join(baseDir, imgDef.URI)
				if raw, err := os.ReadFile(imgPath); err == nil {
					if img, _, err := image.Decode(bytes.NewReader(raw)); err == nil {
						decodedImages[i] = img
					}
				}
			}
		}
	}

	// Each primitive is decoded once; a mesh mounted at several nodes is
	// decoded once and instanced at each node's composed transform.
	type gltfTriangle struct {
		i0, i1, i2 int
		color      cell.Color
		hasUV      bool
	}
	type primitiveGeometry struct {
		positions []Vertex3D
		normals   []Vector3D
		uvs       []UV
		triangles []gltfTriangle
	}
	type primitiveKey struct{ mesh, prim int }
	decoded := make(map[primitiveKey]*primitiveGeometry)

	decodePrimitive := func(meshIdx, primIdx int) (*primitiveGeometry, error) {
		key := primitiveKey{meshIdx, primIdx}
		if g, ok := decoded[key]; ok {
			return g, nil
		}
		mesh := doc.Meshes[meshIdx]
		prim := mesh.Primitives[primIdx]

		posAccIdx, hasPos := prim.Attributes["POSITION"]
		if !hasPos {
			decoded[key] = nil
			return nil, nil
		}

		posBytes, posAcc, posStride, err := getAccessorBytes(posAccIdx)
		if err != nil {
			return nil, err
		}

		elemSize := 12 // 3 * float32
		if posStride < elemSize {
			posStride = elemSize
		}
		positions := make([]Vertex3D, 0, posAcc.Count)
		for i := 0; i < posAcc.Count; i++ {
			offset := i * posStride
			if offset+12 > len(posBytes) {
				break
			}
			x := math.Float32frombits(binary.LittleEndian.Uint32(posBytes[offset:]))
			y := math.Float32frombits(binary.LittleEndian.Uint32(posBytes[offset+4:]))
			z := math.Float32frombits(binary.LittleEndian.Uint32(posBytes[offset+8:]))
			v := Vertex3D{X: float64(x), Y: float64(y), Z: float64(z)}
			if !finiteVertex(v) {
				return nil, fmt.Errorf("mesh %d primitive %d: position %d is not finite", meshIdx, primIdx, i)
			}
			positions = append(positions, v)
		}
		if len(positions) == 0 {
			decoded[key] = nil
			return nil, nil
		}

		// NORMAL is optional. A primitive that carries one usable unit
		// normal per vertex feeds those normals through; anything else
		// leaves them nil so the renderer derives them from the faces, as
		// it does for the other formats.
		var normals []Vector3D
		if normalAccIdx, hasNormal := prim.Attributes["NORMAL"]; hasNormal {
			if normalBytes, normalAcc, normalStride, err := getAccessorBytes(normalAccIdx); err == nil {
				if normalStride < elemSize {
					normalStride = elemSize
				}
				if normalAcc.Count >= len(positions) {
					normals = make([]Vector3D, 0, len(positions))
					for i := 0; i < len(positions); i++ {
						offset := i * normalStride
						if offset+12 > len(normalBytes) {
							normals = nil
							break
						}
						n := Vector3D{
							X: float64(math.Float32frombits(binary.LittleEndian.Uint32(normalBytes[offset:]))),
							Y: float64(math.Float32frombits(binary.LittleEndian.Uint32(normalBytes[offset+4:]))),
							Z: float64(math.Float32frombits(binary.LittleEndian.Uint32(normalBytes[offset+8:]))),
						}
						length := math.Sqrt(n.X*n.X + n.Y*n.Y + n.Z*n.Z)
						if !finiteVector(n) || length < 1e-9 {
							normals = nil
							break
						}
						normals = append(normals, Vector3D{X: n.X / length, Y: n.Y / length, Z: n.Z / length})
					}
				}
			}
		}

		// UV Coordinates (TEXCOORD_0)
		var uvs []UV
		if uvAccIdx, hasUV := prim.Attributes["TEXCOORD_0"]; hasUV {
			if uvBytes, uvAcc, uvStride, err := getAccessorBytes(uvAccIdx); err == nil {
				uvElemSize := 8 // 2 * float32
				if uvStride < uvElemSize {
					uvStride = uvElemSize
				}
				for i := 0; i < uvAcc.Count; i++ {
					offset := i * uvStride
					if offset+8 > len(uvBytes) {
						break
					}
					u := math.Float32frombits(binary.LittleEndian.Uint32(uvBytes[offset:]))
					v := math.Float32frombits(binary.LittleEndian.Uint32(uvBytes[offset+4:]))
					uvs = append(uvs, UV{U: float64(u), V: float64(v)})
				}
			}
		}

		// Material and Texture resolution
		var primImage image.Image
		var baseColor cell.Color

		if prim.Material != nil && *prim.Material >= 0 && *prim.Material < len(doc.Materials) {
			matDef := doc.Materials[*prim.Material]
			if matDef.PbrMetallicRoughness != nil {
				pbr := matDef.PbrMetallicRoughness
				if pbr.BaseColorTexture != nil {
					texIdx := pbr.BaseColorTexture.Index
					if texIdx >= 0 && texIdx < len(doc.Textures) {
						texDef := doc.Textures[texIdx]
						if texDef.Source != nil && *texDef.Source >= 0 && *texDef.Source < len(decodedImages) {
							primImage = decodedImages[*texDef.Source]
						}
					}
				}
				if len(pbr.BaseColorFactor) >= 3 {
					r := uint8(pbr.BaseColorFactor[0] * 255.0)
					g := uint8(pbr.BaseColorFactor[1] * 255.0)
					b := uint8(pbr.BaseColorFactor[2] * 255.0)
					baseColor = cell.NewColorRGB(r, g, b)
				}
			}
		}

		sampleTextureColor := func(u, v float64) cell.Color {
			if primImage == nil {
				if baseColor != 0 {
					return baseColor
				}
				return 0
			}
			bounds := primImage.Bounds()
			bw := bounds.Dx()
			bh := bounds.Dy()
			if bw <= 0 || bh <= 0 {
				return baseColor
			}

			// Wrap / clamp UV coordinates
			u = u - math.Floor(u)
			v = v - math.Floor(v)

			px := bounds.Min.X + int(u*float64(bw-1))
			py := bounds.Min.Y + int(v*float64(bh-1))

			r, g, b, _ := primImage.At(px, py).RGBA()
			return cell.NewColorRGB(uint8(r>>8), uint8(g>>8), uint8(b>>8))
		}

		g := &primitiveGeometry{positions: positions, normals: normals, uvs: uvs}
		if primImage != nil && model.Texture == nil {
			model.Texture = primImage
		}

		addTriangle := func(i0, i1, i2 int) {
			color := baseColor
			hasUV := false
			if len(uvs) > i0 && len(uvs) > i1 && len(uvs) > i2 {
				avgU := (uvs[i0].U + uvs[i1].U + uvs[i2].U) / 3.0
				avgV := (uvs[i0].V + uvs[i1].V + uvs[i2].V) / 3.0
				color = sampleTextureColor(avgU, avgV)
				hasUV = true
			}
			g.triangles = append(g.triangles, gltfTriangle{i0: i0, i1: i1, i2: i2, color: color, hasUV: hasUV})
		}

		// Indices / Faces
		if prim.Indices != nil {
			idxBytes, idxAcc, _, err := getAccessorBytes(*prim.Indices)
			if err != nil {
				return nil, err
			}

			indices := make([]int, 0, idxAcc.Count)
			switch idxAcc.ComponentType {
			case compTypeUnsignedByte:
				for i := 0; i < idxAcc.Count && i < len(idxBytes); i++ {
					indices = append(indices, int(idxBytes[i]))
				}
			case compTypeUnsignedShort:
				for i := 0; i < idxAcc.Count && (i+1)*2 <= len(idxBytes); i++ {
					indices = append(indices, int(binary.LittleEndian.Uint16(idxBytes[i*2:])))
				}
			case compTypeUnsignedInt:
				for i := 0; i < idxAcc.Count && (i+1)*4 <= len(idxBytes); i++ {
					indices = append(indices, int(binary.LittleEndian.Uint32(idxBytes[i*4:])))
				}
			}

			for i := 0; i+2 < len(indices); i += 3 {
				i0, i1, i2 := indices[i], indices[i+1], indices[i+2]
				if i0 < 0 || i1 < 0 || i2 < 0 ||
					i0 >= len(positions) || i1 >= len(positions) || i2 >= len(positions) {
					continue
				}
				addTriangle(i0, i1, i2)
			}
		} else {
			// Non-indexed triangles
			for i := 0; i+2 < len(positions); i += 3 {
				addTriangle(i, i+1, i+2)
			}
		}

		decoded[key] = g
		return g, nil
	}

	appendInstance := func(g *primitiveGeometry, world gltfMat4) error {
		normalMatrix, err := world.normalMatrix()
		if err != nil {
			return err
		}
		vertexOffset := len(model.Vertices)
		uvOffset := len(model.UVs)
		mirrored := world.determinant() < 0

		for _, p := range g.positions {
			v := world.transformPoint(p)
			if !finiteVertex(v) {
				return fmt.Errorf("composed transform produced a non-finite vertex")
			}
			model.Vertices = append(model.Vertices, v)
		}

		if g.normals != nil {
			for _, n := range g.normals {
				t := normalMatrix.transform(n)
				if !finiteVector(t) {
					return fmt.Errorf("composed transform produced a non-finite normal")
				}
				model.VertexNormals = append(model.VertexNormals, t)
			}
		} else {
			// Keep the slice aligned with Vertices; it is dropped below
			// unless every instance carried normals.
			model.VertexNormals = append(model.VertexNormals, make([]Vector3D, len(g.positions))...)
		}

		for _, uv := range g.uvs {
			model.UVs = append(model.UVs, uv)
		}

		for _, tri := range g.triangles {
			i0, i1, i2 := vertexOffset+tri.i0, vertexOffset+tri.i1, vertexOffset+tri.i2
			u0, u1, u2 := uvOffset+tri.i0, uvOffset+tri.i1, uvOffset+tri.i2
			if mirrored {
				// A mirroring transform reverses the triangle's winding;
				// swap two corners so front faces stay front faces. The face
				// UVs are in corner order, so they swap with them.
				i1, i2 = i2, i1
				u1, u2 = u2, u1
			}
			model.Faces = append(model.Faces, []int{i0, i1, i2})
			model.FaceColors = append(model.FaceColors, tri.color)
			if tri.hasUV {
				model.FaceUVs = append(model.FaceUVs, []int{u0, u1, u2})
			} else {
				model.FaceUVs = append(model.FaceUVs, []int{-1, -1, -1})
			}
		}
		return nil
	}

	appendMesh := func(meshIdx int, world gltfMat4) error {
		if meshIdx < 0 || meshIdx >= len(doc.Meshes) {
			return fmt.Errorf("mesh index %d out of range (%d meshes)", meshIdx, len(doc.Meshes))
		}
		for primIdx := range doc.Meshes[meshIdx].Primitives {
			g, err := decodePrimitive(meshIdx, primIdx)
			if err != nil {
				return err
			}
			if g == nil {
				continue
			}
			if err := appendInstance(g, world); err != nil {
				return fmt.Errorf("mesh %d primitive %d: %w", meshIdx, primIdx, err)
			}
		}
		return nil
	}

	if len(doc.Nodes) == 0 {
		// No scene graph at all: every mesh sits at the origin, exactly as
		// before node transforms were supported.
		for meshIdx := range doc.Meshes {
			if err := appendMesh(meshIdx, gltfIdentity); err != nil {
				return Model3D{}, err
			}
		}
	} else {
		if err := validateGLTFNodeGraph(doc); err != nil {
			return Model3D{}, err
		}
		roots, err := gltfSceneRoots(doc)
		if err != nil {
			return Model3D{}, err
		}
		var walk func(nodeIdx, depth int, parent gltfMat4) error
		walk = func(nodeIdx, depth int, parent gltfMat4) error {
			if depth > maxGLTFNodeDepth {
				return fmt.Errorf("node hierarchy is deeper than %d nodes", maxGLTFNodeDepth)
			}
			node := doc.Nodes[nodeIdx]
			local, err := gltfNodeTransform(node)
			if err != nil {
				return fmt.Errorf("node %d %q: %w", nodeIdx, node.Name, err)
			}
			world := parent.mul(local)
			if !world.isFinite() {
				return fmt.Errorf("node %d %q: composed transform is not finite", nodeIdx, node.Name)
			}
			if node.Mesh != nil {
				if err := appendMesh(*node.Mesh, world); err != nil {
					return fmt.Errorf("node %d %q: %w", nodeIdx, node.Name, err)
				}
			}
			for _, child := range node.Children {
				if err := walk(child, depth+1, world); err != nil {
					return err
				}
			}
			return nil
		}
		for _, root := range roots {
			if err := walk(root, 0, gltfIdentity); err != nil {
				return Model3D{}, err
			}
		}
	}

	// Vertex normals are all-or-nothing: a renderer that finds fewer than
	// one per vertex falls back to face normals for the whole model.
	if len(model.VertexNormals) != len(model.Vertices) {
		model.VertexNormals = nil
	} else {
		for i, n := range model.VertexNormals {
			length := math.Sqrt(n.X*n.X + n.Y*n.Y + n.Z*n.Z)
			if length < 1e-9 {
				model.VertexNormals = nil
				break
			}
			model.VertexNormals[i] = Vector3D{X: n.X / length, Y: n.Y / length, Z: n.Z / length}
		}
	}

	if len(model.Vertices) == 0 {
		return Model3D{}, fmt.Errorf("glTF contains no vertices")
	}

	return model, nil
}

// validateGLTFNodeGraph checks every node's indices and local transform, and
// rejects a child cycle or a hierarchy deeper than maxGLTFNodeDepth, so a
// bad file is reported once with a clear message instead of failing while
// it is being drawn.
func validateGLTFNodeGraph(doc *gltfJSON) error {
	for i, node := range doc.Nodes {
		if _, err := gltfNodeTransform(node); err != nil {
			return fmt.Errorf("node %d %q: %w", i, node.Name, err)
		}
		if node.Mesh != nil && (*node.Mesh < 0 || *node.Mesh >= len(doc.Meshes)) {
			return fmt.Errorf("node %d %q: mesh index %d out of range (%d meshes)", i, node.Name, *node.Mesh, len(doc.Meshes))
		}
		for _, child := range node.Children {
			if child < 0 || child >= len(doc.Nodes) {
				return fmt.Errorf("node %d %q: child index %d out of range (%d nodes)", i, node.Name, child, len(doc.Nodes))
			}
		}
	}

	const (
		onStack = iota + 1
		done
	)
	state := make([]int, len(doc.Nodes))
	var visit func(idx, depth int) error
	visit = func(idx, depth int) error {
		if depth > maxGLTFNodeDepth {
			return fmt.Errorf("node hierarchy is deeper than %d nodes", maxGLTFNodeDepth)
		}
		switch state[idx] {
		case onStack:
			return fmt.Errorf("node %d %q is its own ancestor (cycle in the node graph)", idx, doc.Nodes[idx].Name)
		case done:
			return nil
		}
		state[idx] = onStack
		for _, child := range doc.Nodes[idx].Children {
			if err := visit(child, depth+1); err != nil {
				return err
			}
		}
		state[idx] = done
		return nil
	}
	for i := range doc.Nodes {
		if err := visit(i, 0); err != nil {
			return err
		}
	}
	return nil
}
