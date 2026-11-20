package widgets

import (
	"testing"

	"github.com/thebanri/limoni/core/buffer"
	"github.com/thebanri/limoni/core/cell"
	"github.com/thebanri/limoni/graphics"
)

// glowingQuad is a square in the XY plane carrying its own vertex normals,
// so it can be lit without depending on winding.
func glowingQuad(centerX float64) graphics.Model3D {
	return graphics.Model3D{
		Vertices: []graphics.Vertex3D{
			{X: centerX - 0.25, Y: -0.25, Z: 0},
			{X: centerX + 0.25, Y: -0.25, Z: 0},
			{X: centerX + 0.25, Y: 0.25, Z: 0},
			{X: centerX - 0.25, Y: 0.25, Z: 0},
		},
		VertexNormals: []graphics.Vector3D{{Z: 1}, {Z: 1}, {Z: 1}, {Z: 1}},
		Faces:         [][]int{{0, 1, 2, 3}},
	}
}

// aProjectedQuad is where a model's first face lands on screen, drawn with
// Viewer3D at its defaults.
func aProjectedQuad(t *testing.T, model graphics.Model3D) (minX, maxX, minY, maxY float64) {
	t.Helper()
	v := &Viewer3D{Model: model, Shading: ShadingFlat}
	c := drawViewer(t, v)
	minX, maxX, minY, maxY = 1e9, -1e9, 1e9, -1e9
	for y := 0; y < int(c.height); y++ {
		for x := 0; x < int(c.width); x++ {
			if c.grid[y*int(c.width)+x] != 0 {
				if float64(x) < minX {
					minX = float64(x)
				}
				if float64(x) > maxX {
					maxX = float64(x)
				}
				if float64(y) < minY {
					minY = float64(y)
				}
				if float64(y) > maxY {
					maxY = float64(y)
				}
			}
		}
	}
	if minX > maxX {
		t.Fatal("nothing drawn")
	}
	return
}

// A model whose own vertices already sit apart draws apart. This is the
// contract the GLB loader fulfils: it bakes node transforms into Vertices,
// so a file that places the same mesh twice projects to two places.
func TestViewer3DProjectsPreplacedInstancesApart(t *testing.T) {
	left := glowingQuad(-1)
	right := glowingQuad(1)
	both := graphics.Model3D{
		Vertices:      append(append([]graphics.Vertex3D{}, left.Vertices...), right.Vertices...),
		VertexNormals: append(append([]graphics.Vector3D{}, left.VertexNormals...), right.VertexNormals...),
		Faces:         [][]int{{0, 1, 2, 3}, {4, 5, 6, 7}},
	}

	// Each quad alone projects to a narrow band; the pair spans the gap
	// between them, which is only possible if both were drawn at their own
	// positions. If both landed on the origin, the pair would span no more
	// than one quad does.
	_, leftMax, _, _ := aProjectedQuad(t, glowingQuad(-1))
	rightMin, _, _, _ := aProjectedQuad(t, glowingQuad(1))
	bothMin, bothMax, _, _ := aProjectedQuad(t, both)
	if leftMax >= rightMin {
		t.Fatalf("the fixtures overlap on screen (left ends %.0f, right starts %.0f)", leftMax, rightMin)
	}
	if bothMin > leftMax || bothMax < rightMin {
		t.Errorf("the pair spans columns %.0f..%.0f, but the instances sit at %.0f and %.0f: one did not move",
			bothMin, bothMax, leftMax, rightMin)
	}
	if bothMax-bothMin < rightMin-leftMax {
		t.Errorf("the pair spans %.0f columns, narrower than the gap between the instances (%.0f)", bothMax-bothMin, rightMin-leftMax)
	}
}

// Imported vertex normals drive Gouraud shading on their own: a quad whose
// normals face the light is bright, and the same quad with normals facing
// away is dim. Without the normals the two would shade identically, because
// the face geometry is the same.
func TestViewer3DVertexNormalsFeedGouraud(t *testing.T) {
	render := func(normal graphics.Vector3D) int {
		model := glowingQuad(0)
		for i := range model.VertexNormals {
			model.VertexNormals[i] = normal
		}
		v := &Viewer3D{Model: model, Shading: ShadingGouraud, FaceColors: []cell.Color{cell.NewColorRGB(200, 200, 200)}}
		c := drawViewer(t, v)
		total := 0
		for i, dots := range c.grid {
			if dots == 0 {
				continue
			}
			r, _, _ := c.styles[i].Fg.RGB()
			total += int(r)
		}
		return total
	}
	light := graphics.DefaultLight().Direction
	if toward, away := render(light), render(graphics.Vector3D{X: -light.X, Y: -light.Y, Z: -light.Z}); toward <= away {
		t.Errorf("normals towards the light shade %d, away %d: the model's normals were ignored", toward, away)
	}
}

// A Viewer3D renders a fresh context each time, so a model with normals is
// still allocation-free per frame.
func TestViewer3DVertexNormalsDrawDoesNotAllocate(t *testing.T) {
	area := cell.NewRect(0, 0, 40, 20)
	buf := buffer.NewBuffer(area)
	ctx := cell.NewContext(area, cell.Style{})
	model := graphics.NewSphere(1, 12, 12)
	model.VertexNormals = make([]graphics.Vector3D, len(model.Vertices))
	for i, v := range model.Vertices {
		model.VertexNormals[i] = graphics.Vector3D{X: v.X, Y: v.Y, Z: v.Z}.Normalize()
	}
	v := &Viewer3D{Model: model, Shading: ShadingGouraud, RotX: 20, RotY: 30}
	v.Draw(ctx, buf)
	if got := testing.AllocsPerRun(20, func() { v.RotY += 7; v.Draw(ctx, buf) }); got != 0 {
		t.Errorf("%.0f allocs per Draw with vertex normals, want 0", got)
	}
}
