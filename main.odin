// Living Card - a CPU ray marcher that draws a yellow smiley face inside a box
// of six differently colored walls.
//
// Everything that produces pixels is plain Odin working on a byte buffer.
// Raylib is only used for the window, the mouse, and pushing the finished
// buffer to the screen, so the render side can later move to a microcontroller
// with the blit swapped for a display-driver frame push.
//
// Build:  odin build . -o:speed
// Run:    ./ray-march-test
// Drag with the left mouse button to orbit the camera around the face; the
// colored walls show how the view behind it shifts.
// Frame-time statistics are printed to stdout once per second.
package main

import "core:fmt"
import "core:math"
import "core:math/linalg"
import "core:time"
import rl "vendor:raylib"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

RES    :: 120 // internal render resolution, RES x RES pixels
WINDOW :: 480 // window size; the buffer is scaled up to fill it

MAX_STEPS :: 64        // step budget per ray; one that spends it all is treated as touching a surface (see march)
EPSILON   :: f32(0.001) // "close enough to count as touching the surface"
MAX_DIST  :: f32(20.0)  // a ray that travels this far without a hit is a miss

FOV_DEG     :: f32(90.0) // vertical field of view; wide, so the room frames the face rather than the face filling the view
CAM_DIST    :: f32(3.0)  // radius of the camera's orbit around the origin
MOUSE_SENS  :: f32(0.01) // radians of camera rotation per pixel of mouse travel
PITCH_LIMIT :: f32(89.0 * math.PI / 180.0) // keep the camera just short of the poles

// All vector math is f32 because the eventual target FPU is single precision.
// Odin's fixed arrays already support component-wise +, -, * and swizzles like
// v.x or v.xy, so plain arrays are enough for vector types.
Vec2 :: [2]f32
Vec3 :: [3]f32

// ---------------------------------------------------------------------------
// Signed distance primitives
// ---------------------------------------------------------------------------
//
// A signed distance function (SDF) answers one question: "how far is this
// point from the nearest surface?" Positive outside, zero on the surface,
// negative inside. Ray marching needs nothing else to render a scene, and
// shapes combine with plain min/max, which is how the face is built below.
//
// Every primitive is defined at the origin in its own natural orientation. To
// place one elsewhere, move the *point* the other way before evaluating it:
// sd_sphere(p - center, r) is a sphere at center.

// Sphere of radius r.
sd_sphere :: proc(p: Vec3, r: f32) -> f32 {
	return linalg.length(p) - r
}

// Torus (a ring) lying flat in the XY plane: a circle of radius ring_r around
// the Z axis, thickened into a tube of radius tube_r. The trick: the distance
// from p to the ring's circle is a 2D problem. "How far is p from the Z axis,
// minus ring_r" and "how far above the plane" form a 2D point q, and the
// tube is just a circle of radius tube_r around the origin in q-space.
sd_torus :: proc(p: Vec3, ring_r, tube_r: f32) -> f32 {
	q := Vec2{linalg.length(p.xy) - ring_r, p.z}
	return linalg.length(q) - tube_r
}

// ---------------------------------------------------------------------------
// Materials and combining operations
// ---------------------------------------------------------------------------

// Which color a surface takes. The scene reports the material of whichever
// shape ends up nearest, so carving a dark sphere out of the head leaves a
// dark socket.
Material :: enum u8 {
	Skin,
	Feature, // eyes and mouth
	// The six walls of the room, named from the starting view: the camera
	// begins on +Z looking at the face, so +X is on screen right and the
	// back wall is the one behind the face.
	WallLeft,    // -X
	WallRight,   // +X
	WallFloor,   // -Y
	WallCeiling, // +Y
	WallBack,    // -Z
	WallFront,   // +Z, behind the camera at the start
}

// A variable rather than a constant: Odin only lets a constant array be
// indexed by a constant, and the material is only known at run time.
albedo := [Material]Vec3{
	.Skin        = {1.0, 0.85, 0.15},
	.Feature     = {0.06, 0.05, 0.04},
	.WallLeft    = {0.75, 0.25, 0.22}, // red
	.WallRight   = {0.20, 0.60, 0.35}, // green
	.WallFloor   = {0.40, 0.30, 0.20}, // brown
	.WallCeiling = {0.65, 0.65, 0.70}, // light grey
	.WallBack    = {0.20, 0.40, 0.70}, // blue
	.WallFront   = {0.60, 0.25, 0.60}, // purple
}

// A distance query result: how far to the nearest surface, and what it is.
Surface :: struct {
	d:   f32,
	mat: Material,
}

// Carve b out of a. Where p is inside b (b.d < 0), the nearest surface of the
// carved shape is b's surface seen from the inside, which is why b's distance
// flips sign. max() keeps whichever constraint is more limiting. The result
// is never larger than the true distance, so marching with it is still safe.
op_subtract :: proc(a, b: Surface) -> Surface {
	if -b.d > a.d {
		return Surface{-b.d, b.mat}
	}
	return a
}

// Nearest of two shapes: a point's distance to "a or b" is the smaller of the
// two, and the nearer shape is what a ray reaching that distance hits.
op_union :: proc(a, b: Surface) -> Surface {
	return a if a.d < b.d else b
}

// ---------------------------------------------------------------------------
// The scene: a smiley face in a box of colored walls
// ---------------------------------------------------------------------------
//
// The head is a unit sphere. The eyes are two spheres carved out of it, and
// the mouth is a groove carved along an arc of the surface. The camera starts
// on the +Z side, so the face is built to look toward +Z.
//
// Around it is a hollow box the camera orbits inside, each wall a different
// flat color. The face never moves, so the only thing an orbit changes is
// which walls sit behind it and where their edges fall, which makes the
// camera's motion easy to see.

// Eye center on the surface of the unit sphere (x, y chosen, z solved so the
// point sits exactly on the sphere: 0.32² + 0.28² + 0.905² ≈ 1). Only one is
// defined; the other comes from mirroring x.
EYE_CENTER :: Vec3{0.32, 0.28, 0.905}
EYE_RADIUS :: f32(0.12)

// The smile is the lower half of a ring that lies on the head's surface. A
// ring of radius ring_r in a plane at distance lift from the origin sits on
// the unit sphere when ring_r² + lift² = 1. The ring is tilted downward by
// MOUTH_TILT so it is centered below the eyes rather than between them.
MOUTH_TILT   :: f32(22.0 * math.PI / 180.0)
MOUTH_RING_R :: f32(0.45) // radius of the smile's arc
MOUTH_TUBE_R :: f32(0.07) // half-width of the groove

// Anything the scene needs that is constant is computed once, here, not per
// call: scene() runs hundreds of thousands of times per frame, and sin, cos
// and sqrt are the expensive operations in this whole program.
mouth_cos  := math.cos(MOUTH_TILT)
mouth_sin  := math.sin(MOUTH_TILT)
mouth_lift := math.sqrt(1 - MOUTH_RING_R * MOUTH_RING_R)

// Express p in the mouth's own frame. That frame is the world frame rotated
// about X so its +Z axis points from the head's center through the middle of
// the mouth, then slid out along that axis by mouth_lift, which puts the
// ring's plane where the surface is. Rotating about X only mixes y and z;
// it's the ordinary 2D rotation formula.
to_mouth_frame :: proc(p: Vec3) -> Vec3 {
	c, s := mouth_cos, mouth_sin
	return Vec3{p.x, p.y * c + p.z * s, -p.y * s + p.z * c} - Vec3{0, 0, mouth_lift}
}

// Half the width of the room: the walls sit at ±ROOM_HALF on every axis. The
// orbit must fit inside, or the camera would start out inside a wall.
ROOM_HALF :: f32(4.0)
#assert(ROOM_HALF > CAM_DIST)

// Seen from inside, the distance to the nearer wall of a pair on one axis is
// simply how far |p| is short of ROOM_HALF. The nearest over the three axes
// is the distance to the room, and the sign of p on that axis names the wall.
// This is only right inside the box, which holds for every point a ray from
// the camera can visit: the camera starts inside and a march never crosses a
// wall.
room :: proc(p: Vec3) -> Surface {
	s := Surface{ROOM_HALF - abs(p.x), .WallLeft if p.x < 0 else .WallRight}
	if dy := ROOM_HALF - abs(p.y); dy < s.d {
		s = Surface{dy, .WallFloor if p.y < 0 else .WallCeiling}
	}
	if dz := ROOM_HALF - abs(p.z); dz < s.d {
		s = Surface{dz, .WallBack if p.z < 0 else .WallFront}
	}
	return s
}

scene :: proc(p: Vec3) -> Surface {
	head := Surface{sd_sphere(p, 1.0), .Skin}

	// Eyes. abs(p.x) folds the left half of space onto the right, so one
	// sphere carves both sockets.
	pe := Vec3{abs(p.x), p.y, p.z} - EYE_CENTER
	eye := Surface{sd_sphere(pe, EYE_RADIUS), .Feature}
	head = op_subtract(head, eye)

	// Mouth. Build the full ring in its own frame, then keep only the half
	// below the mouth's center line: max() with the half-space "y <= 0" is an
	// intersection, and the half-space's own distance is simply y.
	pm := to_mouth_frame(p)
	ring := sd_torus(pm, MOUTH_RING_R, MOUTH_TUBE_R)
	smile := Surface{max(ring, pm.y), .Feature}
	head = op_subtract(head, smile)

	return op_union(head, room(p))
}

// ---------------------------------------------------------------------------
// Ray marching
// ---------------------------------------------------------------------------

Hit :: struct {
	hit:   bool,
	t:     f32, // distance along the ray at which the surface was found
	steps: int, // how many steps it took; later this drives a cheap AO term
}

// Sphere tracing, the core of the renderer.
//
// Start at the ray origin ro and walk along the direction rd. At each point,
// ask the scene how far away the nearest surface is. That distance is always a
// safe amount to step: a sphere of that radius around the current point
// contains no geometry, so advancing the full amount can't tunnel through
// anything. Near a surface the steps shrink; once they are smaller than
// EPSILON we call it a hit. If the ray wanders off past MAX_DIST, it's a miss.
//
// A ray that uses up every step without either is one that spent them
// creeping along a silhouette, where the safe step shrinks toward zero. It is
// still a hair from the surface, so it counts as a hit: shading it as the rim
// it is grazing beats leaving a dark speck in the wall behind.
march :: proc(ro, rd: Vec3) -> Hit {
	t: f32 = 0
	for step in 0 ..< MAX_STEPS {
		p := ro + rd * t
		d := scene(p).d
		if d < EPSILON {
			return Hit{hit = true, t = t, steps = step}
		}
		t += d
		if t > MAX_DIST {
			return Hit{hit = false, t = t, steps = step}
		}
	}
	return Hit{hit = true, t = t, steps = MAX_STEPS}
}

// Surface normal by central differences.
//
// The gradient of a distance field points straight away from the surface. We
// estimate it by nudging p a tiny amount along each axis and seeing how the
// distance changes, then normalizing the result. Six scene evaluations, but
// only paid for pixels that actually hit something.
calc_normal :: proc(p: Vec3) -> Vec3 {
	h :: f32(0.001)
	n := Vec3{
		scene(p + Vec3{h, 0, 0}).d - scene(p - Vec3{h, 0, 0}).d,
		scene(p + Vec3{0, h, 0}).d - scene(p - Vec3{0, h, 0}).d,
		scene(p + Vec3{0, 0, h}).d - scene(p - Vec3{0, 0, h}).d,
	}
	return linalg.normalize(n)
}

// ---------------------------------------------------------------------------
// Shading
// ---------------------------------------------------------------------------

// Single directional light that rides with the camera, like a lamp mounted
// above and to the right of the viewer. It is given in camera space (right,
// up, toward the viewer) and rotated into world space once per frame, so the
// face stays lit from the viewer's side however far the camera orbits. A
// world-fixed light would leave the far side of the orbit nearly black; the
// parallax experiment keeps a world-fixed light because there the scene, not
// the camera, is what moves.
LIGHT_CAM :: Vec3{0.5, 0.8, 0.6}

light_direction :: proc(cam: Camera) -> Vec3 {
	// forward points from the camera into the scene, so toward the viewer is
	// -forward.
	return linalg.normalize(cam.right * LIGHT_CAM.x + cam.up * LIGHT_CAM.y - cam.forward * LIGHT_CAM.z)
}

// Materials the light acts on. The walls are left out: they are a flat
// backdrop whose job is to show the camera moving, and half of them face away
// from the light, which would leave them nearly black under Lambert.
LIT :: bit_set[Material]{.Skin, .Feature}

// Lambert shading: brightness is the cosine of the angle between the surface
// normal and the direction to the light. Facing the light gives 1, edge-on
// gives 0, facing away is clamped to 0. dot(n, l) is that cosine when both
// vectors are unit length. A small ambient term keeps the dark side from going
// fully black. An unlit material is just its albedo, and skipping the normal
// for it spares six scene evaluations on every wall pixel.
shade :: proc(p: Vec3, mat: Material, light_dir: Vec3) -> Vec3 {
	if mat not_in LIT {
		return albedo[mat]
	}
	n := calc_normal(p)
	ambient := f32(0.12)
	diffuse := max(linalg.dot(n, light_dir), 0)
	return albedo[mat] * (ambient + diffuse)
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

// An orbit camera. It sits on a sphere of radius CAM_DIST around the origin
// and always looks at the origin: yaw spins it around the vertical axis,
// pitch raises and lowers it.
//
// right, up and forward are the camera's own axes expressed in world space.
// Together they are the 3x3 view basis: a direction (cx, cy, cz) in camera
// space becomes right*cx + up*cy + forward*cz in world space. That single
// rotation is all the ray generator needs.
Camera :: struct {
	pos:     Vec3,
	right:   Vec3,
	up:      Vec3,
	forward: Vec3,
}

WORLD_UP :: Vec3{0, 1, 0}

make_camera :: proc(yaw, pitch: f32) -> Camera {
	cp, sp := math.cos(pitch), math.sin(pitch)
	cy, sy := math.cos(yaw), math.sin(yaw)

	// Spherical coordinates -> position on the orbit. At yaw = pitch = 0 the
	// camera sits on +Z looking toward -Z. Increasing yaw swings it toward +X,
	// increasing pitch lifts it toward +Y.
	pos := Vec3{cp * sy, sp, cp * cy} * CAM_DIST

	// Look-at basis. forward points from the camera to the origin. right is
	// perpendicular to both forward and world-up (that's what a cross product
	// gives), and up is then perpendicular to forward and right, so all three
	// are mutually orthogonal unit vectors. The pitch clamp guarantees forward
	// is never parallel to WORLD_UP, which would make the first cross zero.
	forward := linalg.normalize(-pos)
	right   := linalg.normalize(linalg.cross(forward, WORLD_UP))
	up      := linalg.cross(right, forward)
	return Camera{pos = pos, right = right, up = up, forward = forward}
}

// ---------------------------------------------------------------------------
// Rendering into the pixel buffer
// ---------------------------------------------------------------------------

// One pixel: R, G, B, A. Matches raylib's UNCOMPRESSED_R8G8B8A8 byte order.
Pixel :: [4]u8

// The frame buffer lives in static storage, allocated once for the life of the
// program. Nothing inside the frame loop allocates.
pixels: [RES * RES]Pixel

// Color of a ray that escapes past MAX_DIST. The room encloses the camera, so
// none does; it stays as the miss color for a scene without a room.
BACKGROUND :: Vec3{0.08, 0.09, 0.12}

// Convert a 0..1 color to 8-bit RGBA. The +0.5 rounds instead of truncating.
to_pixel :: proc(c: Vec3) -> Pixel {
	return Pixel{
		u8(clamp(c.r, 0, 1) * 255 + 0.5),
		u8(clamp(c.g, 0, 1) * 255 + 0.5),
		u8(clamp(c.b, 0, 1) * 255 + 0.5),
		255,
	}
}

// Trace one ray per pixel and write the shaded color into buf.
render :: proc(buf: ^[RES * RES]Pixel, cam: Camera) {
	// Think of an image plane hanging 1 unit in front of the camera. Its
	// half-height is tan(fov/2); the image is square, so the half-width is
	// the same. Every pixel maps to a point (u, v) on that plane.
	half_extent := math.tan(FOV_DEG * 0.5 * math.PI / 180.0)
	light_dir := light_direction(cam)

	for y in 0 ..< RES {
		// Row -> vertical plane coordinate: +half_extent on the top row,
		// -half_extent on the bottom. The +0.5 aims through the pixel center.
		v := (1.0 - (f32(y) + 0.5) / RES * 2.0) * half_extent
		for x in 0 ..< RES {
			// Column -> horizontal plane coordinate, -half_extent on the left.
			u := ((f32(x) + 0.5) / RES * 2.0 - 1.0) * half_extent

			// The ray leaves the camera position and passes through (u, v, 1)
			// on the image plane. In camera space that direction is just
			// (u, v, 1); the view basis rotates it into world space.
			rd := linalg.normalize(cam.right * u + cam.up * v + cam.forward)

			color := BACKGROUND
			h := march(cam.pos, rd)
			if h.hit {
				p := cam.pos + rd * h.t // reconstruct the hit point from t
				color = shade(p, scene(p).mat, light_dir) // one more query tells us what we hit
			}
			buf[y * RES + x] = to_pixel(color)
		}
	}
}

// ---------------------------------------------------------------------------
// Window, input, blit
// ---------------------------------------------------------------------------

main :: proc() {
	rl.InitWindow(WINDOW, WINDOW, "Living Card - ray march prototype")
	defer rl.CloseWindow()

	// Wrap the pixel buffer in a raylib Image so it can become a GPU texture.
	// The texture is created once; every frame we just push fresh bytes into
	// it with UpdateTexture.
	image := rl.Image{
		data    = &pixels[0],
		width   = RES,
		height  = RES,
		mipmaps = 1,
		format  = .UNCOMPRESSED_R8G8B8A8,
	}
	texture := rl.LoadTextureFromImage(image)
	defer rl.UnloadTexture(texture)

	// Nearest-neighbour upscaling so each rendered pixel shows as a crisp
	// block, the way it will look on the real LCD.
	rl.SetTextureFilter(texture, .POINT)

	src := rl.Rectangle{0, 0, RES, RES}       // the whole texture ...
	dst := rl.Rectangle{0, 0, WINDOW, WINDOW} // ... stretched over the whole window

	yaw, pitch: f32 = 0, 0

	// Frame-time statistics, averaged over one-second windows so stdout stays
	// readable. render_ms is the ray-march cost alone; frame_ms includes the
	// texture upload, draw, and buffer swap.
	stat_render_ms: f64
	stat_frame_ms:  f64
	stat_frames:    int
	stat_window_start := time.tick_now()

	for !rl.WindowShouldClose() {
		// --- input: left-drag orbits the camera ---
		if rl.IsMouseButtonDown(.LEFT) {
			d := rl.GetMouseDelta()
			yaw   += d.x * MOUSE_SENS
			pitch -= d.y * MOUSE_SENS // screen y grows downward; dragging up should lift the camera
			pitch  = clamp(pitch, -PITCH_LIMIT, PITCH_LIMIT)
		}

		// --- render into the CPU buffer, timing just this part ---
		cam := make_camera(yaw, pitch)
		render_start := time.tick_now()
		render(&pixels, cam)
		render_ms := time.duration_milliseconds(time.tick_since(render_start))

		// --- push the buffer to the GPU and draw it scaled to the window ---
		rl.UpdateTexture(texture, &pixels[0])
		rl.BeginDrawing()
		rl.ClearBackground(rl.BLACK)
		rl.DrawTexturePro(texture, src, dst, {0, 0}, 0, rl.WHITE)
		rl.EndDrawing()

		// --- statistics ---
		stat_render_ms += render_ms
		stat_frame_ms  += f64(rl.GetFrameTime()) * 1000
		stat_frames    += 1
		elapsed := time.duration_seconds(time.tick_since(stat_window_start))
		if elapsed >= 1.0 {
			n := f64(stat_frames)
			fmt.printf(
				"render %.3f ms | frame %.3f ms | %.0f fps\n",
				stat_render_ms / n,
				stat_frame_ms / n,
				n / elapsed,
			)
			stat_render_ms = 0
			stat_frame_ms = 0
			stat_frames = 0
			stat_window_start = time.tick_now()
		}
	}
}
