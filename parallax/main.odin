// Living Card - parallax experiment.
//
// The smiley prototype uses an orbit camera: it circles the object and always
// looks at the center with a fixed field of view. That is the wrong model for
// a card you tilt in your hand. What sells the illusion is treating the screen
// as a WINDOW: a fixed rectangle in the card's own space, with a shadow box
// built behind it and the viewer's eye somewhere in front. Tilting the card
// moves the eye relative to the window, and every pixel's ray runs from the
// eye through that pixel's spot on the glass. There is no field of view to
// choose and the scene never moves. That one change gives correct perspective
// on the box walls, correct clipping at the screen edges, and the
// near-shifts-less, far-shifts-more motion your brain reads as depth.
//
// The objects inside the box are ray-marched SDFs exactly as before. The box
// itself is five axis-aligned planes, and a ray's exit point from a box has a
// closed-form answer, so the walls are intersected analytically. Sphere
// tracing crawls toward a plane it approaches at a grazing angle, which is
// every wall here since the eye looks almost straight in, so this one
// shortcut is worth roughly a 3x speedup and the walls come out exact.
//
// Things IN FRONT of the glass are the weak spot of any single-eye parallax
// display: with no stereo, the only thing that says "in front" is that the
// object moves against your head while the box moves with it, and that
// motion is proportional to its distance from the glass. Two things make it
// sell. First, the object sits well out in front, which means it has to stay
// clear of the screen edges over the whole tilt range: an object cut off by
// an edge that is supposedly behind it snaps straight back to the glass.
// Second, a frame is drawn on the glass itself, inset from the screen edge,
// and the box opens behind that frame. The floating object can then pass in
// front of the frame, and occluding a frame is the strongest depth cue 2D
// media has. It's also the one cue that survives in a still photo.
//
// Units: 1 = half the screen width. The screen is the square x, y in [-1, 1]
// at z = 0 in card space, and +z points out of the screen toward the viewer.
//
// Build and run, from the project directory:  odin run parallax -o:speed
// Or build alone with  odin build parallax -o:speed  which writes
// ./parallax.bin (the plain name is taken by this directory).
// Move the mouse to move your head relative to the card. Space toggles an
// automatic sway. Frame-time statistics print to stdout once per second.
//
// Head tracking: run the webcam tracker alongside this program,
//     .venv/bin/python tracker.py        (from this directory; see tracker.py)
// and the card is treated as fixed while your measured eye position drives
// the view directly, which is the classic head-coupled window. Press C to
// re-center on your current position. When the packets stop, the mouse takes
// over again. See the Head tracking section for the protocol.
package main

import "core:fmt"
import "core:math"
import "core:math/linalg"
import "core:net"
import "core:strconv"
import "core:strings"
import "core:time"
import rl "vendor:raylib"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

RES    :: 120 // internal render resolution, RES x RES pixels
WINDOW :: 480 // window size; the buffer is scaled up to fill it

MAX_STEPS :: 64
// The hit threshold scales with distance along the ray: a pixel's footprint
// grows with distance, so a surface 0.005 units away at 10 units out is
// well under a pixel. Fixed thresholds waste steps far away.
EPSILON_REL :: f32(0.0005)

SHADOW_STEPS    :: 20
SHADOW_MAX      :: f32(4.0)  // the box is small, shadow rays never need to go far
SHADOW_SOFTNESS :: f32(10.0) // higher is sharper

// How far the viewer's eye is from the screen, in half-screen-widths. 10 is
// about right for a 480 px window at arm's length. For a 30 mm card screen
// held 35 cm away it would be about 23; the tilt effect barely changes with
// this number, only the at-rest perspective does.
EYE_DIST :: f32(10.0)
MAX_TILT :: f32(25.0 * math.PI / 180.0) // card tilt at the edge of the mouse range

// Input low-pass time constants in seconds. Smooth motion matters as much as
// correct geometry; jitter breaks the illusion instantly. The camera tracker
// updates at ~30 Hz and its distance estimate is the noisiest thing in the
// pipeline, so those get longer constants than the mouse.
SMOOTH_TAU       :: f32(0.06)
HEAD_SMOOTH_TAU  :: f32(0.10)
HEAD_DEPTH_TAU   :: f32(0.40)

HEAD_PORT    :: 5005
HEAD_TIMEOUT :: 0.5 // seconds without a packet before the mouse takes over

BOX_DEPTH :: f32(2.0) // how far the shadow box extends behind the glass

// The frame drawn on the glass. The box opening is the square inside it, so
// everything behind the glass is seen through the frame and anything in
// front of the glass can cover it.
FRAME_WIDTH :: f32(0.10)
FRAME_LIP   :: f32(0.02) // lighter inner edge that makes it read as a solid rim
INNER       :: f32(1.0) - FRAME_WIDTH
FLOOR_Y     :: -INNER

AMBIENT    :: f32(0.28)
SPEC_POWER :: f32(40.0)

Vec2 :: [2]f32
Vec3 :: [3]f32

// ---------------------------------------------------------------------------
// Signed distance primitives
// ---------------------------------------------------------------------------
//
// Same idea as the smiley prototype: each primitive lives at the origin, and
// you place one by moving the query point the other way before evaluating it.

sd_sphere :: proc(p: Vec3, r: f32) -> f32 {
	return linalg.length(p) - r
}

// Torus flat in the XY plane: ring of radius ring_r around Z, tube radius tube_r.
sd_torus :: proc(p: Vec3, ring_r, tube_r: f32) -> f32 {
	q := Vec2{linalg.length(p.xy) - ring_r, p.z}
	return linalg.length(q) - tube_r
}

// Axis-aligned box with half-extents b. Fold p into the positive octant with
// abs(), then q is how far outside the box each axis is. Outside, the distance
// is the length of the positive parts of q; inside, it's the (negative)
// largest component, i.e. how far to the nearest face.
sd_box :: proc(p, b: Vec3) -> f32 {
	q := Vec3{abs(p.x) - b.x, abs(p.y) - b.y, abs(p.z) - b.z}
	outside := Vec3{max(q.x, 0), max(q.y, 0), max(q.z, 0)}
	inside := min(max(q.x, max(q.y, q.z)), 0)
	return linalg.length(outside) + inside
}

// Box with rounded edges: shrink the box by r, then pad the distance by r.
sd_round_box :: proc(p, b: Vec3, r: f32) -> f32 {
	return sd_box(p, b - r) - r
}

// ---------------------------------------------------------------------------
// Materials and combining operations
// ---------------------------------------------------------------------------

Material :: enum u8 {
	Wall,
	Frame,
	Skin,
	Feature,
	Blue,
	Green,
	Red,
	Purple,
}

albedo := [Material]Vec3{
	.Wall    = {0.62, 0.60, 0.66},
	.Frame   = {0.13, 0.13, 0.15},
	.Skin    = {1.0, 0.85, 0.15},
	.Feature = {0.06, 0.05, 0.04},
	.Blue    = {0.25, 0.45, 0.95},
	.Green   = {0.30, 0.80, 0.40},
	.Red     = {0.95, 0.25, 0.25},
	.Purple  = {0.65, 0.35, 0.85},
}

// Strength of the specular highlight. Walls are matte plaster with just a
// hint of sheen; the objects are glossy.
gloss := [Material]f32{
	.Wall    = 0.06,
	.Frame   = 0.04,
	.Skin    = 0.35,
	.Feature = 0.1,
	.Blue    = 0.5,
	.Green   = 0.5,
	.Red     = 0.5,
	.Purple  = 0.3,
}

WALL_LINE :: Vec3{0.30, 0.30, 0.36} // grid line color on the walls
FRAME_LIP_COLOR :: Vec3{0.42, 0.42, 0.46}

Surface :: struct {
	d:   f32,
	mat: Material,
}

// Nearest of two shapes.
op_union :: proc(a, b: Surface) -> Surface {
	return a if a.d < b.d else b
}

// Carve b out of a (see the smiley prototype for the reasoning).
op_subtract :: proc(a, b: Surface) -> Surface {
	if -b.d > a.d {
		return Surface{-b.d, b.mat}
	}
	return a
}

// ---------------------------------------------------------------------------
// The smiley head, reused from the prototype (unit sphere, face toward +Z)
// ---------------------------------------------------------------------------

EYE_CENTER   :: Vec3{0.32, 0.28, 0.905}
EYE_RADIUS   :: f32(0.12)
MOUTH_TILT   :: f32(22.0 * math.PI / 180.0)
MOUTH_RING_R :: f32(0.45)
MOUTH_TUBE_R :: f32(0.07)

mouth_cos  := math.cos(MOUTH_TILT)
mouth_sin  := math.sin(MOUTH_TILT)
mouth_lift := math.sqrt(1 - MOUTH_RING_R * MOUTH_RING_R)

to_mouth_frame :: proc(p: Vec3) -> Vec3 {
	c, s := mouth_cos, mouth_sin
	return Vec3{p.x, p.y * c + p.z * s, -p.y * s + p.z * c} - Vec3{0, 0, mouth_lift}
}

smiley :: proc(p: Vec3) -> Surface {
	head := Surface{sd_sphere(p, 1.0), .Skin}
	pe := Vec3{abs(p.x), p.y, p.z} - EYE_CENTER
	head = op_subtract(head, Surface{sd_sphere(pe, EYE_RADIUS), .Feature})
	pm := to_mouth_frame(p)
	ring := sd_torus(pm, MOUTH_RING_R, MOUTH_TUBE_R)
	head = op_subtract(head, Surface{max(ring, pm.y), .Feature})
	return head
}

// ---------------------------------------------------------------------------
// The scene: objects inside a shadow box behind the glass
// ---------------------------------------------------------------------------
//
// Everything is placed in card space, so it moves with the card exactly like
// something physically embedded in it would. Objects sit at different depths
// on purpose: the parallax shift of a point is proportional to its distance
// from the glass, so depth variety is what makes the motion legible.

HEAD_POS :: Vec3{0, -0.12, -1.3}
HEAD_R   :: f32(0.5)

BALL_R   :: f32(0.22)
BALL_POS :: Vec3{-0.62, FLOOR_Y + BALL_R, -0.5} // resting on the floor, near the front left

RING_R    :: f32(0.22)
RING_TUBE :: f32(0.08)
RING_POS  :: Vec3{0.6, FLOOR_Y + RING_TUBE, -1.6} // lying flat on the floor at the back right

PILLAR_HALF :: Vec3{0.11, 0.45, 0.11}
PILLAR_POS  :: Vec3{-0.62, FLOOR_Y + PILLAR_HALF.y, -1.7} // standing at the back left

// Floating in FRONT of the glass. How far out it can sit is bounded by the
// screen edge: at the full tilt the eye moves EYE_DIST*sin(MAX_TILT) sideways
// and the ball's image shifts against it by about that times z/(EYE_DIST-z).
// With these numbers, and the tilt clamped to a disc (see main), the image
// stays inside the screen by about 0.03 at the worst tilt, crossing over the
// drawn frame on the way, which is the point.
POP_POS :: Vec3{0.34, 0.36, 0.8}
POP_R   :: f32(0.15)

// Distance to the nearest object. The walls are not in here; see box_exit.
objects :: proc(p: Vec3) -> Surface {
	// Smiley head, scaled: evaluate the unit head at p / r and scale the
	// distance back up. A bounding sphere skips the eyes and mouth whenever
	// p is clearly outside the head; the sphere's distance is never larger
	// than the carved head's, so returning it is still safe for marching.
	ph := (p - HEAD_POS) / HEAD_R
	bound := sd_sphere(ph, 1.0)
	s: Surface
	if bound > 0.05 {
		s = Surface{bound * HEAD_R, .Skin}
	} else {
		s = smiley(ph)
		s.d *= HEAD_R
	}

	s = op_union(s, Surface{sd_sphere(p - BALL_POS, BALL_R), .Blue})
	// The torus is defined around Z; swapping y and z with .xzy lays it flat
	// on the floor, around Y.
	s = op_union(s, Surface{sd_torus((p - RING_POS).xzy, RING_R, RING_TUBE), .Green})
	s = op_union(s, Surface{sd_round_box(p - PILLAR_POS, PILLAR_HALF, 0.03), .Purple})
	s = op_union(s, Surface{sd_sphere(p - POP_POS, POP_R), .Red})
	return s
}

// Where a ray leaves the inside of the box, and the inward normal of the
// wall it leaves through. The cavity is x, y in [-INNER, INNER] and
// z >= -BOX_DEPTH, open toward +z. Per axis, the ray crosses the far side of
// that slab at t = (bound - origin) / direction; the wall actually hit is the
// nearest of the three crossings. Rays reaching here entered through the
// opening, inside the x and y slabs, so only the far crossings matter.
box_exit :: proc(ro, rd: Vec3) -> (t: f32, n: Vec3) {
	t = math.F32_MAX
	if rd.x != 0 {
		tx := ((INNER if rd.x > 0 else -INNER) - ro.x) / rd.x
		if tx < t { t, n = tx, Vec3{-1 if rd.x > 0 else 1, 0, 0} }
	}
	if rd.y != 0 {
		ty := ((INNER if rd.y > 0 else -INNER) - ro.y) / rd.y
		if ty < t { t, n = ty, Vec3{0, -1 if rd.y > 0 else 1, 0} }
	}
	if rd.z < 0 {
		tz := (-BOX_DEPTH - ro.z) / rd.z
		if tz < t { t, n = tz, Vec3{0, 0, 1} }
	}
	return
}

// ---------------------------------------------------------------------------
// Ray marching
// ---------------------------------------------------------------------------

Hit :: struct {
	hit:   bool,
	t:     f32,
	steps: int,
}

// Sphere tracing against the objects, as in the prototype, but stopping at
// t_max: past that point the ray is inside a wall anyway.
march :: proc(ro, rd: Vec3, t_max: f32) -> Hit {
	t: f32 = 0
	for step in 0 ..< MAX_STEPS {
		d := objects(ro + rd * t).d
		if d < EPSILON_REL * t {
			return Hit{hit = true, t = t, steps = step}
		}
		t += d
		if t > t_max {
			return Hit{hit = false, t = t, steps = step}
		}
	}
	return Hit{hit = true, t = t, steps = MAX_STEPS}
}

calc_normal :: proc(p: Vec3) -> Vec3 {
	h :: f32(0.001)
	n := Vec3{
		objects(p + Vec3{h, 0, 0}).d - objects(p - Vec3{h, 0, 0}).d,
		objects(p + Vec3{0, h, 0}).d - objects(p - Vec3{0, h, 0}).d,
		objects(p + Vec3{0, 0, h}).d - objects(p - Vec3{0, 0, h}).d,
	}
	return linalg.normalize(n)
}

// Soft shadow. March from the surface toward the light and remember the
// closest the ray ever came to an object, scaled by how far along the ray
// that happened: passing close to something a short way out means a deep
// shadow, passing close far away means a faint one. The naive version just
// samples the distance at each step, which bands visibly (the sample points
// straddle the true closest approach); estimating the closest approach
// between consecutive samples, as Inigo Quilez describes, removes the bands.
//
// Only objects cast shadows. The box is lit as if it had its own internal
// lighting; with the walls as occluders, a light from the front puts most of
// the interior in the ceiling's shadow and the whole thing goes black.
soft_shadow :: proc(ro, rd: Vec3) -> f32 {
	res: f32 = 1
	t: f32 = 0.02
	prev_h: f32 = 1e10
	for _ in 0 ..< SHADOW_STEPS {
		h := objects(ro + rd * t).d
		if h < 0.001 {
			return 0
		}
		y := h * h / (2 * prev_h)
		d := math.sqrt(h * h - y * y)
		res = min(res, SHADOW_SOFTNESS * d / max(t - y, 0.0001))
		prev_h = h
		t += h
		if t > SHADOW_MAX {
			break
		}
	}
	return clamp(res, 0, 1)
}

// ---------------------------------------------------------------------------
// Shading
// ---------------------------------------------------------------------------

GRID_STEP :: f32(0.5)
GRID_LINE :: f32(0.03) // half-width of a grid line

// Wall color with grid lines. The lines are computed from the hit position in
// card space, so they are glued to the walls and give the eye straight lines
// to track. The axis the wall faces is skipped: every point on that wall
// shares that coordinate, so it would paint the whole wall as a line.
wall_color :: proc(p, n: Vec3) -> Vec3 {
	for i in 0 ..< 3 {
		if abs(n[i]) > 0.5 {
			continue
		}
		nearest := math.round(p[i] / GRID_STEP) * GRID_STEP
		if abs(p[i] - nearest) < GRID_LINE {
			return WALL_LINE
		}
	}
	return albedo[.Wall]
}

// The frame is flat matte plastic with a lighter inner lip so the edge reads.
frame_color :: proc(p: Vec3) -> Vec3 {
	if max(abs(p.x), abs(p.y)) < INNER + FRAME_LIP {
		return FRAME_LIP_COLOR
	}
	return albedo[.Frame]
}

// Lambert + soft shadow + a Blinn-Phong highlight. The highlight depends on
// the eye position, so it slides across surfaces as you tilt, exactly like a
// reflection on a real glossy object would. That's a strong cue for "solid
// thing" rather than "picture".
shade :: proc(p, n, view_dir, light: Vec3, mat: Material) -> Vec3 {
	base := albedo[mat]
	if mat == .Wall {
		base = wall_color(p, n)
	} else if mat == .Frame {
		base = frame_color(p)
	}

	diffuse := max(linalg.dot(n, light), 0)
	shadow: f32 = 1
	if diffuse > 0 {
		// Start a little off the surface so the ray doesn't immediately "hit"
		// the surface it started on.
		shadow = soft_shadow(p + n * 0.01, light)
	}

	half := linalg.normalize(light + view_dir)
	spec := math.pow(max(linalg.dot(n, half), 0), SPEC_POWER) * gloss[mat]

	return base * (AMBIENT + diffuse * shadow) + spec * shadow
}

// ---------------------------------------------------------------------------
// The card, the eye, and the light
// ---------------------------------------------------------------------------

// How the card is rotated in the world: about its own X axis (nod) and Y axis
// (turn), in radians. This is exactly what an IMU will supply later.
Tilt :: struct {
	x, y: f32,
}

// Everything the renderer needs, expressed in card space.
View :: struct {
	eye:   Vec3,
	light: Vec3,
}

// The room light, fixed in the world: in front of the card, above and to the
// right of the viewer. Rotating it into card space each frame means the
// shading changes as you tilt, the way a real lit object's does.
light_world := linalg.normalize(Vec3{0.35, 0.5, 0.8})

// Card tilts under a head that stays put: rotate the world-fixed eye and
// light into card space.
view_from_tilt :: proc(tilt: Tilt) -> View {
	// The card's orientation as a 3x3 rotation: turn about Y, then nod about X.
	rot := linalg.matrix3_rotate_f32(tilt.y, {0, 1, 0}) * linalg.matrix3_rotate_f32(tilt.x, {1, 0, 0})

	// The eye and the light are fixed in the world; the scene is fixed in the
	// card. To render in card space we need those two world vectors in card
	// coordinates, which is the inverse rotation, and for a pure rotation the
	// inverse is the transpose. The eye sits straight in front of the card's
	// rest pose, EYE_DIST away.
	to_card := linalg.transpose(rot)
	return View{eye = to_card * Vec3{0, 0, EYE_DIST}, light = to_card * light_world}
}

// Card fixed, head moves: the eye is wherever the tracker measured it, and
// the light does not rotate because the card (here, the monitor) never moved.
view_from_head :: proc(eye: Vec3) -> View {
	return View{eye = eye, light = light_world}
}

// ---------------------------------------------------------------------------
// Head tracking
// ---------------------------------------------------------------------------
//
// A separate tracker process watches the webcam and sends the viewer's eye
// position to localhost:HEAD_PORT as one text datagram per camera frame:
// "x y z", in screen half-widths, relative to the screen center, +x right,
// +y up, +z toward the viewer. Keeping the tracker out of process means the
// renderer has no camera or vision dependency, and the IMU later plugs into
// the same seam. Anything that can send a UDP packet can drive the view.

Head_Tracker :: struct {
	socket:      net.UDP_Socket,
	ok:          bool,
	eye:         Vec3, // newest position received
	last_packet: f64,  // rl.GetTime() when it arrived; 0 = never
}

tracker_open :: proc() -> Head_Tracker {
	socket, err := net.make_bound_udp_socket(net.IP4_Loopback, HEAD_PORT)
	if err != nil {
		fmt.eprintfln("head tracker: cannot listen on UDP port %d (%v); mouse only", HEAD_PORT, err)
		return {}
	}
	// Non-blocking so the frame loop can drain whatever has arrived and move on.
	net.set_blocking(socket, false)
	return Head_Tracker{socket = socket, ok = true}
}

// Read every waiting datagram and keep the newest valid one.
tracker_poll :: proc(t: ^Head_Tracker) {
	if !t.ok {
		return
	}
	buf: [128]byte
	for {
		n, _, err := net.recv_udp(t.socket, buf[:])
		if err != nil || n <= 0 { // Would_Block: nothing left this frame
			break
		}
		if eye, ok := parse_eye(string(buf[:n])); ok {
			t.eye = eye
			t.last_packet = rl.GetTime()
		}
	}
}

// "x y z" as three floats. fields_iterator does not allocate.
parse_eye :: proc(s: string) -> (eye: Vec3, ok: bool) {
	rest := s
	for i in 0 ..< 3 {
		field, more := strings.fields_iterator(&rest)
		if !more {
			return
		}
		v, good := strconv.parse_f32(field)
		if !good {
			return
		}
		eye[i] = v
	}
	return eye, true
}

tracker_active :: proc(t: Head_Tracker) -> bool {
	return t.ok && t.last_packet > 0 && rl.GetTime() - t.last_packet < HEAD_TIMEOUT
}

// ---------------------------------------------------------------------------
// Rendering into the pixel buffer
// ---------------------------------------------------------------------------

Pixel :: [4]u8
pixels: [RES * RES]Pixel

to_pixel :: proc(c: Vec3) -> Pixel {
	return Pixel{
		u8(clamp(c.r, 0, 1) * 255 + 0.5),
		u8(clamp(c.g, 0, 1) * 255 + 0.5),
		u8(clamp(c.b, 0, 1) * 255 + 0.5),
		255,
	}
}

render :: proc(buf: ^[RES * RES]Pixel, view: View) {
	for y in 0 ..< RES {
		for x in 0 ..< RES {
			// Where this pixel physically sits on the glass, in card space:
			// x across [-1, 1], y up [-1, 1], z = 0.
			on_glass := Vec3{
				(f32(x) + 0.5) / RES * 2.0 - 1.0,
				1.0 - (f32(y) + 0.5) / RES * 2.0,
				0,
			}

			// The ray leaves the eye and passes through that point. This is
			// the entire camera model: no basis, no field of view. Off-center
			// eyes automatically get the asymmetric perspective a real window
			// shows.
			rd := linalg.normalize(on_glass - view.eye)

			// What this ray ends on if no object is in the way: the frame,
			// right at the glass, if the pixel is under the frame; otherwise
			// the wall the ray reaches inside the box. Then march the objects
			// only that far. Whichever comes first is what we see, so an
			// object in front of the glass is drawn over the frame.
			t_back: f32
			n_back: Vec3
			mat_back: Material
			if max(abs(on_glass.x), abs(on_glass.y)) > INNER {
				t_back = linalg.length(on_glass - view.eye)
				n_back = Vec3{0, 0, 1}
				mat_back = .Frame
			} else {
				t_back, n_back = box_exit(view.eye, rd)
				mat_back = .Wall
			}
			h := march(view.eye, rd, t_back)

			p, n: Vec3
			mat: Material
			if h.hit {
				p = view.eye + rd * h.t
				n = calc_normal(p)
				mat = objects(p).mat
			} else {
				p = view.eye + rd * t_back
				n = n_back
				mat = mat_back
			}
			buf[y * RES + x] = to_pixel(shade(p, n, -rd, view.light, mat))
		}
	}
}

// ---------------------------------------------------------------------------
// Window, input, blit
// ---------------------------------------------------------------------------

main :: proc() {
	rl.InitWindow(WINDOW, WINDOW, "Living Card - parallax experiment")
	defer rl.CloseWindow()

	image := rl.Image{
		data    = &pixels[0],
		width   = RES,
		height  = RES,
		mipmaps = 1,
		format  = .UNCOMPRESSED_R8G8B8A8,
	}
	texture := rl.LoadTextureFromImage(image)
	defer rl.UnloadTexture(texture)
	rl.SetTextureFilter(texture, .POINT)

	src := rl.Rectangle{0, 0, RES, RES}
	dst := rl.Rectangle{0, 0, WINDOW, WINDOW}

	tilt: Tilt
	sway := false
	last_mouse: rl.Vector2
	mouse_seen := false

	tracker := tracker_open()
	defer if tracker.ok {
		net.close(tracker.socket)
	}
	tracker_was_active := false
	head_rest: Vec3       // tracker reading that counts as "straight ahead"
	head_calibrated := false
	eye := Vec3{0, 0, EYE_DIST} // smoothed eye position while head tracking

	stat_render_ms: f64
	stat_frame_ms:  f64
	stat_frames:    int
	stat_window_start := time.tick_now()

	for !rl.WindowShouldClose() {
		tracker_poll(&tracker)
		using_tracker := tracker_active(tracker)
		if using_tracker != tracker_was_active {
			fmt.println("head tracker connected" if using_tracker else "head tracker lost, back to mouse")
			tracker_was_active = using_tracker
		}

		view: View
		if using_tracker {
			// The first reading, or the reading when C is pressed, becomes the
			// rest position: whatever the camera geometry, "where you are now"
			// is straight ahead. Distance is used as measured.
			if !head_calibrated || rl.IsKeyPressed(.C) {
				head_rest = Vec3{tracker.eye.x, tracker.eye.y, 0}
				head_calibrated = true
			}
			target := tracker.eye - head_rest

			// Same limit as the tilt clamp, expressed as an angle from straight
			// ahead, so the pop-out ball's edge margin still holds.
			lateral := linalg.length(target.xy)
			limit := target.z * math.tan(MAX_TILT)
			if lateral > limit {
				target.x *= limit / lateral
				target.y *= limit / lateral
			}

			dt := rl.GetFrameTime()
			k_xy := 1 - math.exp(-dt / HEAD_SMOOTH_TAU)
			k_z := 1 - math.exp(-dt / HEAD_DEPTH_TAU)
			eye.x += (target.x - eye.x) * k_xy
			eye.y += (target.y - eye.y) * k_xy
			eye.z += (target.z - eye.z) * k_z
			view = view_from_head(eye)
		} else {
		// --- input: where is your head relative to the card? ---
		// hx, hy in [-1, 1], right and up positive. The mouse stands in for
		// head tracking. Until the mouse has moved over the window the card
		// rests at (0, 0); raylib reports the cursor at the origin before any
		// event, which would otherwise start the card tilted to a corner.
		// (raylib's IsCursorOnScreen depends on enter/leave events that turn
		// out to be unreliable on macOS, hence the position check instead.)
		if rl.IsKeyPressed(.SPACE) {
			sway = !sway
		}
		m := rl.GetMousePosition()
		if m != last_mouse {
			last_mouse = m
			mouse_seen = true
		}
		inside := m.x >= 0 && m.x < WINDOW && m.y >= 0 && m.y < WINDOW

		hx, hy: f32
		if sway {
			t := f32(rl.GetTime())
			hx = 0.85 * math.sin(t * 0.9)
			hy = 0.55 * math.sin(t * 1.7 + 0.6)
		} else if mouse_seen && inside {
			hx = clamp(m.x / WINDOW * 2 - 1, -1, 1)
			hy = clamp(1 - m.y / WINDOW * 2, -1, 1)
		}

		// Tilt is limited by magnitude, not per axis: a hand tilts a card
		// within a cone, not a square, and the pop-out ball's edge margin is
		// worked out for the cone. A corner of the mouse range would tilt
		// both axes fully at once and push the ball off the screen.
		if l := math.sqrt(hx * hx + hy * hy); l > 1 {
			hx /= l
			hy /= l
		}

		// Moving your head to the right is the same as turning the card to
		// the left, hence the sign on the turn.
		target := Tilt{x = hy * MAX_TILT, y = -hx * MAX_TILT}

		// Exponential smoothing toward the target, framerate independent.
		k := 1 - math.exp(-rl.GetFrameTime() / SMOOTH_TAU)
		tilt.x += (target.x - tilt.x) * k
		tilt.y += (target.y - tilt.y) * k
		view = view_from_tilt(tilt)
		}

		// --- render ---
		render_start := time.tick_now()
		render(&pixels, view)
		render_ms := time.duration_milliseconds(time.tick_since(render_start))

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
