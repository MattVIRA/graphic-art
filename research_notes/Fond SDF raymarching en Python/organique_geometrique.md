# Organic and repeated-geometric SDF backgrounds: building blocks, operators, scene recipes and parameter spaces (GLSL 330, real-time raymarching)

Scope: the ORGANIC (metaballs, blobs, gyroids/TPMS, noise-displaced, cellular) and REPEATED GEOMETRIC (domain repetition, tunnels, Truchet, columns, rings, lattices) background modes of a full-screen raymarched fragment shader behind a 3D model, in moderngl / OpenGL 3.3 core on macOS. It covers formulas, parameters and ranges, cost, pitfalls and audio targets. Fractals and landscapes are covered elsewhere.

Labelling: anything under "Cited Findings" comes from a source. Parameter ranges, recipes and audio mappings are mostly my own engineering inferences and are labelled that way. The TPMS gradient bounds were computed numerically for this note (numpy, 160³ grid over one period).

---

## 1. Primitive library: which primitives are cheapest and most expressive

### Takeaway
iq's exact primitives are all cheap, closed-form and branch-free (or nearly). A small core is enough:
- sphere, box / round box, box frame, torus, capsule, link and infinite cylinder;
- the bound octahedron (one dot product);
- elongation, rounding and onion to derive more shapes for free.

These operations stay exact or bound, so they keep the field safe for sphere tracing. For a background, the most expressive per instruction are round box, torus, box frame and capsule. Polyhedra are better taken from hg_sdf's `fGDF` family.

### Cited Findings
- iq's first block of primitives are "all exact and true SDFs". He warns that many Shadertoy implementations are incorrect. — [iq, distance functions](https://iquilezles.org/articles/distfunctions/)
- Core formulas, verbatim from iq — [iq, distance functions](https://iquilezles.org/articles/distfunctions/):
  - Sphere: `length(p) - r`
  - Box: `vec3 q = abs(p) - b; return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);`
  - Round box: same with `q = abs(p) - b + r` and `... - r`
  - Torus: `vec2 q = vec2(length(p.xz)-t.x,p.y); return length(q)-t.y;`
  - Link (chain link): `vec3 q = vec3(p.x, max(abs(p.y)-le,0.0), p.z); return length(vec2(length(q.xy)-r1,q.z)) - r2;`
  - Infinite cylinder: `length(p.xz-c.xy)-c.z`
  - Vertical capsule: `p.y -= clamp(p.y, 0.0, h); return length(p) - r;`
  - Capsule between a and b: a clamp-projected segment distance minus r.
  - Box frame: three box evaluations with `p = abs(p)-b; q = abs(p+e)-e;`. It is a natural "architectural wireframe" primitive.
  - Capped torus: takes `sc` = sin/cos of the aperture angle, so it gives an animated open ring.
- Octahedron: the exact version has a 3-way branch. The bound version is one line, `p = abs(p); return (p.x+p.y+p.z-s)*0.57735027;`, labelled "bound (not exact)". — [iq, distance functions](https://iquilezles.org/articles/distfunctions/)
- Derived shapes:
  - Elongation (`q = p - clamp(p,-h,h)`) is "a perfect distance preserving operation". A capsule is an elongated sphere. The 1-line version leaves a zero-distance core inside for 2D/3D elongation. A second version with `max(q,0)+min(max(q.x,max(q.y,q.z)),0)` fixes interiors.
  - Rounding (`d - rad`) is exact.
  - Onion (`abs(d) - thickness`) is exact and can be applied repeatedly to make concentric shells.
  - Source for all three: [iq, distance functions](https://iquilezles.org/articles/distfunctions/)
- Change of metric (replacing `length` with an Lⁿ norm, e.g. `length8`) gives rounded-square shapes. iq does not recommend it except for toy demos: the result is only a lower bound and needs more steps. — [iq, distance functions](https://iquilezles.org/articles/distfunctions/)
- Uniform scaling is exact via `primitive(p/s)*s`. Non-uniform scaling is not possible while keeping a correct SDF. Rotation and translation are exact (apply the inverse transform to p). — [iq, distance functions](https://iquilezles.org/articles/distfunctions/)
  - For non-uniform scale, Jamie Wong recommends multiplying by the smallest scale factor, since underestimating is safe. — [Jamie Wong](https://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/)
- hg_sdf (Mercury, version 2021-07-28, MIT or CC-BY-NC-4.0) offers more primitives — [hg_sdf.glsl](https://mercury.sexy/hg_sdf/hg_sdf.glsl):
  - `fBoxCheap` and `fBox`;
  - `fBlob`, an organic icosahedral blob;
  - `fCylinder`, `fCapsule`, `fTorus`, `fHexagonCircumcircle/Incircle` and `fCone`;
  - `fGDF`, a "generalized distance function" built from a table of normals. It gives octahedron, dodecahedron, icosahedron, truncated octahedron and truncated icosahedron, each with an exponent `e` variant that rounds them.
- hg_sdf's design rule: "Stay Lipschitz continuous… don't have any distance gradient larger than 1". Also: "Don't ever multiply distances by some value to 'fix' a Lipschitz continuity violation". — [hg_sdf.glsl](https://mercury.sexy/hg_sdf/hg_sdf.glsl)
  - This conflicts with common Shadertoy practice (e.g. `*0.5` on gyroids, see §3). hg_sdf's position is a modelling-hygiene rule. The 0.5 factor is a pragmatic fix for fields that are not SDFs to begin with.

### Inferences
- Rough cost ranking, cheapest first:
  - plane, sphere, infinite cylinder and bound octahedron (about 1 length or dot);
  - torus, capsule and link (2 lengths);
  - box and round box (abs, max, length);
  - capped torus;
  - box frame (3 boxes);
  - exact octahedron (branches);
  - `fGDF` icosahedral variants (7–19 dot products; dodeca/icosa use 6–10 normals);
  - cone and other exotic primitives.

  For a background evaluated about 60–100 times per pixel, stick to the first four groups inside repetition loops.
- A continuously morphable "primitive family" with float parameters only, no branches:
  - round box: corner radius 0 to min(b) moves from box to sphere-like;
  - torus: ratio t.y/t.x from 0.05 (thin ring) to 1 (horn torus / blob);
  - elongated sphere: h = 0 gives a sphere, h > 0 a capsule or pill;
  - onion count and thickness;
  - `mix(sdA, sdB, t)` between any two primitives (see §5 for why the mix is safe).
- Polyhedra via `fGDF` give a "crystal / architectural" vocabulary. The `e` exponent variant uses `pow`, so it is more expensive. Prefer the non-`e` version (max of dot products) inside repetition.

### Gaps
- No per-primitive instruction counts or GPU timings were found (Apple Silicon or otherwise). The ranking above is by operation count, not measured.

---

## 2. Operators: booleans, smooth min variants, shell/round/elongate, displacement, twist/bend, warping, folds, repetition, per-cell randomization, hg_sdf

### Takeaway
The safe toolbox:
- `min`, `max`, `max(-a,b)`;
- iq's normalized quadratic smin, the default because it is fast, rigid and conservative;
- exponential smin when blending many objects in any order (it is the only associative DD-family smin);
- rounding, onion and elongation (exact);
- abs/mirror folds;
- `round()`-based repetition with neighbour checks, or `clamp`ed IDs for limited repetition;
- hg_sdf's chamfer, stairs, columns, pipe, engrave and groove operators for "architectural" joins.

Displacement, twist, bend and domain warping break the distance property. They need a reduced step (a Lipschitz factor), or should be kept small. iq's fBM-as-SDF (smin/smax of random sphere grids) is the safe way to add detail.

### Cited Findings — booleans and smooth minimum
- Booleans — [iq, distance functions](https://iquilezles.org/articles/distfunctions/):
  - union `min(a,b)`;
  - subtraction `max(-a,b)`, not commutative;
  - intersection `max(a,b)`;
  - xor `max(min(a,b),-max(a,b))`.

  Union and Xor give a true SDF, but only in the exterior. Subtraction and intersection give bounds.
- iq's smooth minimum article lists exponential, root, sigmoid, quadratic, cubic, quartic, circular and circular-geometrical variants. Each pre-multiplies k by a normalization factor (the kernel value g(0)) so that k is the actual thickness of the blend in distance units. The same k therefore produces about the same neck size for every variant, and k is also the bounding-box inflation needed. — [iq, smooth minimum](https://iquilezles.org/articles/smin/)
- Verbatim kernels — [iq, smooth minimum](https://iquilezles.org/articles/smin/):
  - Quadratic (default): `k *= 4.0; float h = max(k-abs(a-b),0.0)/k; return min(a,b) - h*h*k*(1.0/4.0);`
  - Cubic: `k *= 6.0; … - h*h*h*k*(1.0/6.0);`
  - Quartic: `k *= 16.0/3.0; … - h*h*h*(4.0-h)*k*(1.0/16.0);`
  - Circular: `k *= 1.0/(1.0-sqrt(0.5)); … - k*0.5*(1.0+h-sqrt(1.0-h*(h-2.0)));`
  - Exponential: `float r = exp2(-a/k) + exp2(-b/k); return -k*log2(r);`
  - Root: `k *= 2.0; x=b-a; return 0.5*(a+b-sqrt(x*x+k*k));`
  - Circular geometrical (the Mercury `fOpUnionRound`): `k *= 1/(1-sqrt(.5)); return max(k,min(a,b)) - length(max(k-vec2(a,b),0.0));`
  - Legacy quadratic, still widely circulated: `h = clamp(0.5+0.5*(b-a)/k,0,1); return mix(b,a,h) - k*h*(1.0-h);`
- Property table (Rigid / Local / Conservative / Associative) — [iq, smooth minimum](https://iquilezles.org/articles/smin/):

  | Variant | Rigid | Local | Conservative | Associative |
  |---|---|---|---|---|
  | Quadratic, cubic, quartic, circular | Yes | No | Yes | No |
  | Exponential | No | No | Yes | **Yes** |
  | Sigmoid, root | No | No | Yes | No |
  | Circular geometrical | Yes | Yes | **No** | Yes |

  iq: "the Exponential and the Circular Geometrical can be blended in any order". The CD family is not associative: the blend order matters.
- The circular geometrical variant overestimates distance in convex regions, which leaves holes in raymarched renders. It can be rescued by letting the marcher back-track: break on `abs(d)` instead of signed d. Because it is local, it needs fewer iterations (in iq's comparison image, fewer pixels need 30+ steps). — [iq, smooth minimum](https://iquilezles.org/articles/smin/)
- All DD/CD-family smins have gradient length ≤ 1 in the blend region, so they never overestimate and are safe for raymarching. Their under-estimation band is non-local: it extends to infinity along the Voronoi edges between shapes, so the march slows there. Among CD kernels, the quadratic has the largest slow region and the circular and quartic smaller ones (shown visually, not quantified). — [iq, smooth minimum](https://iquilezles.org/articles/smin/)
- Material blend factor: the quadratic smin returning `vec2(dist, mix)` — [iq, smooth minimum](https://iquilezles.org/articles/smin/):

  ```glsl
  float h = 1.0 - min(abs(a-b)/(4.0*k), 1.0); float w = h*h; float m = w*0.5; float s = w*k;
  return (a<b) ? vec2(a-s,m) : vec2(b-s,1.0-m);
  ```

  `.y` is used to mix colors or materials. A cubic version uses `h*h*h` and `6k`.
- Smooth subtraction is `-smin(a,-b,k)` and smooth intersection is `-smin(-a,-b,k)` (bounds). — [iq, distance functions](https://iquilezles.org/articles/distfunctions/)
- iq's own default is the quadratic: "it's fast, close enough to circular, never overestimates, and while not local it still not too bad". — [iq, smooth minimum](https://iquilezles.org/articles/smin/)

### Cited Findings — hg_sdf combination operators (Mercury)
- Mercury dislikes smin "because it does not preserve Lipschitz continuity in many cases". Their operators treat (a, b) as a local 2D coordinate system and evaluate a 2D shape there. They are correct when the two surfaces meet at right angles. Beyond about 30° off, the Lipschitz condition fails, and parallel close surfaces are the worst case. The radius r "should be much smaller than the object size". — [hg_sdf.glsl](https://mercury.sexy/hg_sdf/hg_sdf.glsl)
- Verbatim ops — [hg_sdf.glsl](https://mercury.sexy/hg_sdf/hg_sdf.glsl):
  - `fOpUnionChamfer(a,b,r) = min(min(a,b),(a-r+b)*sqrt(0.5))` gives a 45° bevel;
  - `fOpUnionRound`: `u = max(vec2(r-a,r-b),0); return max(r,min(a,b)) - length(u);`
  - `fOpUnionStairs(a,b,r,n)`: `s=r/n; u=b-r; return min(min(a,b), 0.5*(u+a+abs(mod(u-a+s,2*s)-s)));` gives terraced steps, nice for architecture;
  - `fOpUnionColumns(a,b,r,n)` gives n-1 circular columns along the 45° join and uses an `if((a<r)&&(b<r))` early-out;
  - `fOpUnionSoft`, the Media Molecule quadratic smin;
  - `fOpPipe(a,b,r) = length(vec2(a,b)) - r` keeps only a tube along the intersection curve. It turns the intersection of two surfaces into glowing pipes;
  - `fOpEngrave(a,b,r) = max(a,(a+r-abs(b))*sqrt(0.5))` cuts a V-groove;
  - `fOpGroove` and `fOpTongue` give carpentry-style grooves and tongues.

### Cited Findings — domain operators (hg_sdf)
- Repetition and mirroring — [hg_sdf.glsl](https://mercury.sexy/hg_sdf/hg_sdf.glsl):
  - `pMod1/2/3(inout p, size)` repeats and returns the cell index: `c = floor((p+half)/size); p = mod(p+half,size)-half`;
  - `pModMirror1/2` mirrors every second cell "so they match at the boundaries";
  - `pModGrid2` also mirrors at the diagonal;
  - `pModSingle1` repeats only the positive half-space;
  - `pModInterval1(p,size,start,stop)` gives a finite count;
  - `pModPolar(inout vec2 p, float repetitions)` does angular repetition and fixes the cell index for odd counts;
  - `pMirror(p,dist)` is `abs(p)-dist`;
  - `pMirrorOctant` folds into 1/8 of the plane;
  - `pReflect(p, n, offset)` reflects across an arbitrary plane;
  - `pR(p,a)` and `pR45` rotate.

### Cited Findings — domain repetition correctness (iq, 2023 article)
- Naive repetition — [iq, domain repetition](https://iquilezles.org/articles/sdfrepetition/):

  ```glsl
  vec2 id = round(p/s); return sdf(p - s*id, id);
  ```

  It costs one evaluation for infinitely many copies. It is only correct when the shape is symmetric about the tile boundary and identical in every cell. Varied size, rotation or jitter per cell, or asymmetric shapes, break the field and cause holes, cracks and broken shadows.
- Correct version: evaluate the own cell plus the nearest neighbours, picked with `o = sign(p - s*id)`, and take the min — [iq, domain repetition](https://iquilezles.org/articles/sdfrepetition/):

  ```glsl
  vec2 id = round(p/s); vec2 o = sign(p-s*id); float d=1e20;
  for(int j=0;j<2;j++) for(int i=0;i<2;i++){ vec2 rid=id+vec2(i,j)*o; d=min(d,sdf(p-s*rid,rid)); }
  ```

  Cost: 2 evaluations (1D), 4 (2D), 8 (3D). Instances that span more than one tile need a wider neighbourhood.
- Mirrored repetition (flip every other cell) keeps one evaluation and stays correct for shapes that are allowed to be mirrored. — [iq, domain repetition](https://iquilezles.org/articles/sdfrepetition/)
- Limited repetition — [iq, domain repetition](https://iquilezles.org/articles/sdfrepetition/); [iq, distance functions](https://iquilezles.org/articles/distfunctions/):
  - intersecting an infinite grid with a box via `max` gives continuous but incorrect distances;
  - clamp the id instead: `rid = clamp(rid, -(count-1)*0.5, (count-1)*0.5)`, with the neighbour loop. Even counts need a half-cell offset;
  - iq's short form: `q = p - s*clamp(round(p/s),-l,l)`.
- Angular repetition: make the polar angle periodic with n sectors (`2π/n`) and evaluate the two nearest sectors (2 evaluations) for correctness. — [iq, domain repetition](https://iquilezles.org/articles/sdfrepetition/)
- Per-cell variation: pass `id` into the SDF and hash it to vary size, color, orientation or phase. "Breaking regularity is important for natural-looking results." Trig on the GPU is cheap: "avoid over-optimizing it". — [iq, domain repetition](https://iquilezles.org/articles/sdfrepetition/)
- Symmetry: `p.x = abs(p.x)` or `p.xz = abs(p.xz)` before evaluation. The result is a bound, not exact, if the mirrored object crosses the mirror plane. — [iq, distance functions](https://iquilezles.org/articles/distfunctions/)

### Cited Findings — deformations
- iq on deformations — [iq, distance functions](https://iquilezles.org/articles/distfunctions/):
  - they "distort the distance field and make it non-Euclidean"; you "will probably need to decrease your step size";
  - the reduction factor is "inversely proportional to the compression of the space, which is given by the Jacobian";
  - better to get close to the shape with exact operations first, then apply "as small of a distortion or displacement as possible".
- Code from iq — [iq, distance functions](https://iquilezles.org/articles/distfunctions/):
  - Displacement: `d1 + displacement(p)`, with the example `sin(20*p.x)*sin(20*p.y)*sin(20*p.z)`.
  - Twist: `c=cos(k*p.y); s=sin(k*p.y); q=vec3(mat2(c,-s,s,c)*p.xz,p.y)`, with an example k = 10.
  - Cheap bend: the same, rotating p.xy by `k*p.x`.
- Adding fBM or a sine to an SDF stops it being an SDF. It works for small amplitude and frequency, and "breaks rapidly when we push it". — [iq, fBM SDF](https://iquilezles.org/articles/fbmsdf/)
- iq's alternative for SDF detail — [iq, fBM SDF](https://iquilezles.org/articles/fbmsdf/):
  - `sdBase(p)` is an infinite grid of random-radius spheres (radius = 0.5·hash per lattice vertex, ≤ half the cell). It is the min over the 8 corner spheres of the cell.
  - Octaves combine as follows, with a rotation-and-scale-by-2 matrix applied to p each octave:

    ```glsl
    n = s*sdBase(p); n = smax(n, d-0.1*s, 0.3*s); d = smin(n, d, 0.3*s); s *= 0.5;
    ```

    The constants (0.1·s inflation, 0.3·s smoothness) are proportional to s.
  - The result stays a valid SDF "within the capabilities of smin() and smax()". iq uses 11 octaves.
- Domain warping is `f(p + h(p))`. iq's canonical recipe is `q = (fbm(p), fbm(p+vec2(5.2,1.3)))`, then `r = (fbm(p+4q+(1.7,9.2)), fbm(p+4q+(8.3,2.8)))`, then `fbm(p+4r)`. The intermediate q and r are reused for coloring. He also mentions recursive warping with sine waves instead of fBM. — [iq, warping](https://iquilezles.org/articles/warp/)
- Lipschitz factors for modifiers. One developer's SDF tool proposes:
  - primitives get a factor of 1;
  - modifiers multiply their child's factor by their own (phrased as a proposal);
  - the factor is applied only at march time, so shell thicknesses are not distorted.

  With the factor, "a fairly extreme Twist rendered without glitches", but the march then hit the step limit quickly. Practical test: set extreme parameters, lower the step multiplier, and see whether the shape changes. — [incoherency.co.uk notes, 2025-04-10](https://incoherency.co.uk/notes/20250410.html)
- Galin et al.'s Segment Tracing computes local Lipschitz bounds along the ray for different operators. The adaptive step is inversely proportional to the bound. — [Galin et al., Segment Tracing Using Local Lipschitz Bounds, EG 2020](https://diglib7.eg.org/bitstream/handle/10.1111/cgf13951/v39i2pp545-554.pdf)
- Over-relaxation (Keinert et al., "Enhanced Sphere Tracing", 2014) speeds up sphere tracing by stepping ω·d with fallback when spheres stop overlapping. — [Keinert et al.](https://diglib.eg.org/handle/10.2312/stag.20141233.001-008)
  - A 2023 EG short paper compares enhanced sphere tracing at ω = 0.88 against relaxed sphere tracing at ω = 1.5. — [Bán & Valasek 2023](https://diglib.eg.org/bitstream/handle/10.2312/egs20231014/057-060.pdf)

### Inferences
- **Twist Lipschitz factor (derived, not sourced).** Twisting about y at rate k moves a point at radius r tangentially at k·r per unit of y. The Jacobian's largest stretch is therefore about sqrt(1 + (k·r)²). Multiply the distance by `1/sqrt(1+(k*rmax)^2)` for the scene's maximum radius rmax, or use the local r. Bend behaves similarly with the distance from the bend axis. Verify with the step-slider test above.
- **Displacement Lipschitz factor (derived).** For `d + A·sin(ω·p.x)·…` the gradient grows by up to about A·ω·√3. Scale the step by about `1/(1 + A·ω·√3)`. To keep the background fast, cap A·ω (e.g. A·ω ≤ 0.5) rather than shrinking the step a lot.
- **Domain warping `p += A·h(p)`** with |∇h| ≤ L stretches space by up to (1 + A·L): divide by that.
- **For audio**, drive amplitudes (A, k-twist, warp) but clamp their product with frequency. Spikes from kicks should go into things that are distance-safe:
  - smin k (it only shrinks gradients);
  - radius or rounding (exact);
  - onion thickness (exact);
  - rotation angles (exact).
- **Recommended defaults:**
  - quadratic smin with k in about 0.05–0.6 world units for blobs (iq normalization means k = neck thickness);
  - exponential smin for N-ball metaballs (order-independent, so ball order in the uniform array doesn't matter);
  - hg_sdf stairs/columns/chamfer with r ≤ 0.2 × object size for architecture.
- **GLSL 330 compatibility:**
  - `round`, `mod`, `floor`, `sign`, `atan(y,x)`, integer bit operations and uniform-driven loop counts are all available in GLSL 1.30+ / 3.30;
  - hg_sdf's `mod(c,2)*2 - 1` uses int literals (`2`) in float expressions, which strict GLSL 330 core compilers reject. Write `2.0`;
  - hg_sdf also defines `sgn` itself (unlike `sign` it never returns 0).

### Gaps
- No primary source found for an exact twist or bend Lipschitz constant formula. The sqrt(1+(kr)²) factor is my derivation.
- The ω sweep results from Keinert et al. were not retrieved (full text not accessed).
- iq's 2D "smooth voronoi" and "voronoi edges" articles were not fetched. Cellular SDF coverage below relies on the fBM-SDF sphere grid.

---

## 3. Organic scenes: metaballs, TPMS (gyroid / Schwarz P / diamond), noise blobs, liquid morphing, cellular

### Takeaway
Three cheap and very different organic families:
- **N-ball metaballs** with exponential or quadratic smin, positions animated on the CPU and sent as a `uniform vec4 balls[16]`;
- **TPMS shells** of the form `abs(f(scale·p) - bias)/(scale·L) - thickness`, with L the gradient bound (√3 for gyroid, P and diamond);
- **iq's fBM-SDF** (random sphere grid) for coral or porous detail.

TPMS are the cheapest per evaluation (a few sin/cos), fill infinite space with no repetition artifacts, and have the most continuous knobs: scale, bias, thickness, warp and type blend.

### Cited Findings
- **Gyroid in shader practice.** A common form (Art-of-Code lineage) is `abs(0.7*dot(sin(p), cos(p.yzx))/10.) - .02` after `p *= 10.`, with the gyroid domain rotated over time (`p.yz *= Rot(T)`). It is combined with a sphere shell (`abs(length(p)-1.)-.03`) via smin. The template is credited to Martijn Steinrucken (The Art of Code), MIT; the original is Shadertoy M3yGzh. — [Godot Shaders, "alien orb ball"](https://godotshaders.com/shader/alien-orb-ball)
- **Shadertoy-style `sdGyroid(p, scale, thickness, bias)`:**
  - scales p, takes `abs(dot(sin(p), cos(p.zxy)) - bias)`, divides by scale, subtracts thickness, and multiplies by 0.5;
  - the code comment says the gyroid "isn't Lipschitz continuous", so halving reduces overshoot;
  - example call: scale 2, thickness 0.0001, bias 0.8, intersected with a capped cylinder.

  (Seen via search-result summary; the page itself returned HTTP 429.) — [Observable, makio135 "gyroid column"](https://observablehq.com/@makio135/gyroid-column)
- **A graded gyroid lattice** is raymarched in WebGL2 with cell size graded radially (dense core, open skin) and three quality tiers of 56 / 80 / 88 steps. — [three.js forum, Graded Gyroid](https://discourse.threejs.org/t/graded-gyroid-a-lattice-ray-marched-from-a-distance-field-no-mesh/93435)
- **Sums of gyroid layers** with rescaled coordinates give a cheap fBM-like "cyclic noise". — [0b5vr, Cyclic Noise](https://scrapbox.io/0b5vr/Cyclic_Noise)
- **TPMS level-set approximations:**
  - Schwarz Primitive: `cos x + cos y + cos z + c`;
  - Neovius: `3(cos 2x + cos 2y + cos 2z) + 4 cos 2x cos 2y cos 2z + c`.

  — [arXiv 2511.19913](https://arxiv.org/pdf/2511.19913)

  The diamond form is a sum of products of sines and cosines, with per-axis cell constants α, β, γ and offset c. — [encyclopedia.pub TPMS entry](https://encyclopedia.pub/entry/39188)
  - The gyroid equation printed in the encyclopedia.pub entry has a non-standard term arrangement and should not be copied from there. Use `sin x cos y + sin y cos z + sin z cos x`.
- **TPMS families** can be continuously deformed into each other. The gyroid and the Lidinoid each lie in a separate 1-parameter family. Meeks' 5-parameter family covers the genus-3 TPMS known at the time, except the gyroid. — [Wikipedia, Triply periodic minimal surface](https://en.wikipedia.org/wiki/Triply_periodic_minimal_surface)
- **Metaballs as uniforms.** A Godot SDF metaball shader hard-codes `const int n_balls = 8; uniform vec3 ball_pos[n_balls]; uniform float radi[n_balls];`. — [Godot Shaders, metaballs](https://godotshaders.com/shader/metaballs/)
  - A web component loops 25 spheres merged into a central blob with polynomial smin. Cost scales with the sphere count, because every march step evaluates all of them. — [21st.dev iridescent raymarched glass metaballs](https://21st.dev/community/components/cldgayo/iridescent-raymarched-glass-metaballs/default)
- **Exponential smin** is the associative option (blend order irrelevant). The CD family (quadratic etc.) is order-dependent. — [iq, smooth minimum](https://iquilezles.org/articles/smin/)
- **iq's fBM SDF:** a random-radius sphere grid with smin/smax octaves gives "detail" that remains an SDF. It works on a plane host (for organic ground or coral walls) or any host shape. — [iq, fBM SDF](https://iquilezles.org/articles/fbmsdf/)
- **hg_sdf's `fBlob`** is an icosahedral-symmetry organic blob primitive (cos-modulated sphere using φ-based normals). — [hg_sdf.glsl](https://mercury.sexy/hg_sdf/hg_sdf.glsl)

### Computed (this research, numpy over one period) — TPMS gradient magnitudes

| Field f (unscaled) | Value range | max ∥∇f∥ everywhere | ∥∇f∥ on surface (\|f\|<0.05) min / mean / max |
|---|---|---|---|
| Gyroid `sin x cos y + sin y cos z + sin z cos x` | ±1.5 | 1.732 (√3) | 1.414 / 1.531 / 1.732 |
| Schwarz P `cos x + cos y + cos z` | ±3 | 1.732 | 0.999 / 1.314 / 1.732 |
| Diamond `sxsysz + sxcycz + cxsycz + cxcysz` | ±1.414 | 1.732 | 1.413 / 1.495 / 1.732 |
| Neovius `3(cx+cy+cz) + 4cxcycz` | ±13 | 7.0 | 0.999 / 2.575 / 5.196 |

### Inferences
**TPMS SDF, safe version.**

```glsl
float tpms(vec3 p, int type){                 // or a float blend, see §5
  vec3 s = sin(p), c = cos(p);
  if(type==0) return dot(s, c.yzx);                         // gyroid
  if(type==1) return c.x+c.y+c.z;                           // Schwarz P
  return s.x*s.y*s.z + s.x*c.y*c.z + c.x*s.y*c.z + c.x*c.y*s.z; // diamond
}
float sdTPMS(vec3 p, float scale, float bias, float thick){
  float f = tpms(p*scale, 0);
  return (abs(f - bias)/(1.7320508*scale)) - thick;  // 1-Lipschitz since |∇f| ≤ √3·scale
}
```

- Dividing by √3·scale makes the field a valid lower bound (no overstepping). The common `*0.5/scale` is also safe (0.5 < 0.577) but slightly slower.
- Because ∥∇f∥ on the surface varies (1.41–1.73 for the gyroid), the actual wall thickness varies by about ±10% across the surface. Usually that is invisible.
- Schwarz P varies more (1.0–1.73), and Neovius needs a divisor of about 7, which makes it slow. Prefer gyroid, diamond and P.

**TPMS parameters and useful ranges** (inference, in scene units where the visible volume is about 10–40 units):

| Parameter | Range | Effect |
|---|---|---|
| `scale` | 0.3–3 (period 2π/scale ≈ 2–20 units) | Cell size. Large cells give cathedral caverns, small ones sponge or foam |
| `bias` | −1.2…+1.2 for the gyroid (its range is ±1.5) | Moves the level set. Near 0 gives two equal labyrinths. Large \|bias\| makes the surface break into isolated pockets or tubes, a dramatic topology change from one float |
| `thick` | 0.005–0.3 | Wall thickness. Use `abs()` for a sheet; drop `abs` for solid/void "skeletal" network variants (`f - bias`) |
| Anisotropy (`p*vec3(sx,sy,sz)`) | 0.5–2 per axis | Stretched membranes. Divide by the largest axis scale × √3 |
| Domain rotation per axis | radians | Slowly rotating cells. Exact, so free to animate |
| Warp | `p += A*sin(p.yzx*w + t)` with A·w ≤ 0.5 | Melting or liquid look. Divide the distance by (1 + A·w) |
| Type blend | `mix(gyroid, diamond, t)` | Continuous morph between families. Both are √3-bounded, so the mix is also √3-bounded |
| Grading | `scale = mix(s0, s1, smoothstep(r0, r1, length(p)))` | Dense near the centre or skin. The gradient picks up an extra ∂scale/∂p term, so add margin to the divisor |

**Metaball scene recipe.**
- Use N = 6–16 balls, as `uniform vec4 uBall[16]` (xyz = position, w = radius) plus `uniform int uCount`.
- Compute orbits in Python, e.g. Lissajous (`sin(a_i t + φ_i)` per axis), per-ball radius from an EQ band, and a kick impulse that throws balls outward then lets them re-merge.
- Field: `d = -k*log2(Σ exp2(-(length(p-c_i)-r_i)/k))`. That is 1 log2 plus N exp2 per step, and order-independent.
- Bound it with one sphere test (`if(length(p-center) - Rbound > 0.5) return length(p-center)-Rbound;`). That avoids N evaluations for rays far away, which matters for a background where most rays miss.
- k range 0.1–1.0. A large k gives a single "lava lamp" mass, a small k separate droplets.
- Audio:
  - k ← section energy (calm = separate, drop = merged);
  - radius_i ← EQ band i;
  - orbit speed ← BPM;
  - position kick ← kick impulse with a spring (the project already has `Spring` in `app/routing.py`).
- Cost per step is about N × (length + exp2). With N = 12 at 64 steps that is about 770 sphere evaluations per pixel. That is fine at half resolution, but is the main cost driver of this mode.

**Liquid morphing.**
- `mix(sdA, sdB, t)` gives a continuous morph between two shapes (e.g. a metaball cluster and a gyroid ball intersected with a sphere).
- Plus a low-frequency sine warp of the domain, e.g. `p += 0.3*sin(p.yzx*1.7 + time)`, divided by (1 + 0.3·1.7) ≈ 1.5.
- Plus smin with a ground or ceiling plane to make drips.
- Audio:
  - t ← slow section energy;
  - warp phase ← beat phase;
  - warp amplitude ← RMS (clamped).

**Noise-displaced blob.**
- `length(p) - R + A*noise(p*w + t)` with A·w ≤ about 0.5, step × 1/(1 + A·w·L_noise).
- Or, safer and more "organic coral": iq's sdFbm with 3–5 octaves instead of 11 (cost × octaves × 8 corner spheres).
- Audio:
  - A ← high EQ bands (shimmer);
  - w ← constant;
  - octave count ← fixed.

**Cellular / Voronoi SDF.**
- iq's `sdBase` (8 random spheres per lattice cell, min over corners) is already a cellular SDF: bubbles or foam.
- Variants:
  - subtract it from a box or plane, `max(host, -sdBase)`, to get Swiss-cheese or porous walls;
  - onion it, `abs(sdBase)-t`, to get bubble shells;
  - smin it with itself at scale ×2 to get froth.
- True Voronoi-edge distance in 3D (3×3×3 = 27-cell search) is far more expensive. Avoid it in the background or use the 8-corner variant.

**Shell-of-gyroid on a sphere.** `max(sphereShell, gyroidSheet)` or `smin` (Art of Code style). This makes "organic orbs" that can surround the central model at a large radius (e.g. R = 8–15) as an enclosing cage.

### Gaps
- No primary article by The Art of Code on the gyroid was retrieved, only derivative code. No source found for Lidinoid or I-WP nodal formulas in shader form.
- No benchmark found for metaball counts on Apple GPUs.

---

## 4. Repeated geometric scenes: infinite grids, tunnels, Truchet 3D, columns/architecture, rotating rings, lattices — and keeping parallax readable behind a central model

### Takeaway
Repetition gives infinite architecture for one evaluation of the cell SDF. The look is set by:
- the repetition topology: cartesian 1D/2D/3D, polar (`pModPolar`) and mirrored;
- the cell content: box frame, round box, capsule beams, torus rings, hg_sdf stairs or columns joins;
- per-cell hashed variation.

The two classic traps:
1. Per-cell variation or asymmetric shapes with naive `mod`/`round`. Fix with mirrored repetition, by keeping the shape inside a margin of the cell, or with a 2- or 4-neighbour check.
2. Moiré and aliasing in the far field. Fix with fog, distance-based detail fade and capped max distance.

Behind a central model, readability comes from low contrast, depth fog and keeping repetition geometry away from the model's screen footprint (e.g. a tunnel or cage around the camera path).

### Cited Findings
- The infinite repetition cost is "constant-time regardless of the number of copies". Correct handling of variation needs 2 (1D), 4 (2D) or 8 (3D) evaluations. — [iq, domain repetition](https://iquilezles.org/articles/sdfrepetition/)
- Mirrored repetition (`pModMirror1/2`, `pModGrid2`) makes neighbouring cells match at the boundaries. — [hg_sdf.glsl](https://mercury.sexy/hg_sdf/hg_sdf.glsl)
  - It is iq's single-evaluation fix when mirroring is acceptable artistically. — [iq, domain repetition](https://iquilezles.org/articles/sdfrepetition/)
- `pModPolar(p.xy, n)` repeats around an axis and returns the sector index, with an odd-count fix. Combined with `pMod1(p.z, period)` along the axis, this is the standard tunnel construction (rings or ribs around z, repeated along z). — [hg_sdf.glsl](https://mercury.sexy/hg_sdf/hg_sdf.glsl)
  - Polar repetition with variation needs 2 evaluations (two nearest sectors). — [iq, domain repetition](https://iquilezles.org/articles/sdfrepetition/)
- **Curving tunnels:** derivative Shadertoy code offsets the tunnel's x/y by a path function of z, `p.xy -= path(p.z)`, which is the usual way to make a tunnel wind; the camera follows the same path. — [scrapbox background.gl code (CC-BY-NC-SA)](https://scrapbox.io/api/code/yutaro/background%20gl/blocks.gl)
- **Truchet:** a raymarched Truchet-tile WebGL shader project exists as boilerplate. — [pjkarlik/truchet-webgl](https://github.com/pjkarlik/truchet-webgl)
- **Limited repetition** (`clamp` id) is for "human made things where we usually don't have infinite of anything", such as colonnades or rows of pillars. — [iq, distance functions](https://iquilezles.org/articles/distfunctions/)
- **hg_sdf architectural joins:** `fOpUnionStairs` (terraces), `fOpUnionColumns` (fluting or columns along joins), `fOpUnionChamfer` (bevel), `fOpEngrave` / `fOpGroove` (grooves cut into walls) and `fOpPipe` (pipes along intersections). — [hg_sdf.glsl](https://mercury.sexy/hg_sdf/hg_sdf.glsl)
- **Soft shadows** from near misses are "almost free" with SDFs, by attenuating light with the minimum of k·d/t along the shadow ray. — [Cambridge, Ray Marching and SDFs lecture](https://www.cl.cam.ac.uk/teaching/1718/FGraphics/1.%20Ray%20Marching%20and%20Signed%20Distance%20Fields.pdf)
  - For a background that is probably too expensive, but the same idea underlies glow.

### Inferences — scene recipes (cell content, parameters and ranges, audio targets)

**A. Infinite lattice / space frame.**
- Content: `pMod3(p, vec3(P))`, then the min of three infinite capsules or boxes along x, y and z through the cell centre: `min(length(p.xy), length(p.yz), length(p.zx)) - r`.
- Alternatives: `sdBoxFrame(p, vec3(P*0.5), e)`, or a sphere at nodes plus beams.
- Parameters:
  - period P = 2–8;
  - beam radius r = 0.02–0.3·P;
  - node sphere radius 0–0.4·P;
  - a global rotation of the lattice (exact, cheap, readable);
  - a per-axis period ratio for non-cubic lattices.
- It is symmetric per cell, so the naive repetition is correct. Keep r < P/2.
- Audio:
  - beam radius ← low EQ (the lattice "breathes");
  - node radius ← kick;
  - lattice rotation ← slow integrate of mid band.

**B. Grid of objects with per-cell variation (pillars, cubes, "3D equalizer").**
- 2D repetition on xz with `id = round(p.xz/P)`.
- Height h(id) = `hash(id)` × band energy[`int(hash2(id)*7)`]. Each pillar listens to one EQ band, which gives a city or equalizer field.
- Use `sdRoundBox(q, vec3(w, h, w), rr)` with w ≤ 0.35·P.
- Because heights differ between neighbours, use iq's 2×2 neighbour loop, or keep w small and accept minor artifacts (low-risk because pillars are vertical and only height varies).
- Adding per-cell rotation `pR(q.xz, hash*τ)` requires w·√2 < P/2 or the neighbour check.

**C. Tunnel (camera flies along z).**
- Construction:

  ```glsl
  p.xy -= path(p.z); float a = pModPolar(p.xy, N); p.x -= R; float c = pMod1(p.z, L);
  ```

  then the cell content (box, capsule, torus segment). The tunnel wall itself is `R - length(p.xy)`, possibly onioned.
- Parameters:
  - N = 3–24 sectors (integer, but can be a float and floored, with a crossfade);
  - radius R = 3–10;
  - period L = 0.5–4;
  - twist along z: `pR(p.xy, z*twist)`, twist 0–0.3 rad/unit. Low-risk because the tunnel radius is bounded: factor 1/sqrt(1+(twist·R)²).
- Camera: the project's orbit camera looks at the model. The tunnel can be "scrolled" past it by adding `scroll = ∫ speed dt` to p.z. That needs no camera change, so parallax stays correct with the orbit.
- Audio:
  - scroll speed ← BPM × section energy;
  - a step on each beat: `scroll += L` eased over beat phase, so rings pass exactly on beats;
  - twist ← snare impulse with decay;
  - ring emissive ← kick;
  - sector count change on section changes only (it is a jump).

**D. Rotating rings / gyroscope.**
- Limited polar or z-repetition of `sdTorus` or `sdCappedTorus`. Each ring i is rotated by angle_i(t) about a different axis: `p = rot_i * p` before evaluation. Only a few rings (4–10), done as an explicit loop (no repetition) since each has its own rotation.
- Parameters:
  - ring count;
  - radius ratio between rings (e.g. 1.2–1.6 geometric);
  - tube thickness 0.02–0.2;
  - capped-torus aperture angle (0–π), animatable.
- Audio:
  - ring i speed ← EQ band i;
  - aperture ← RMS;
  - kick → all rings snap 90° on a spring.

**E. Columns / architecture / "infinite cathedral".**
- `pModInterval1` or clamped-id repetition along x for a colonnade, `pMod1` along z, and `pMirror` on x for a symmetric hall.
- Columns: `sdCylinder` (infinite on y) unioned with floor and ceiling via `fOpUnionColumns` or `fOpUnionStairs`. Arches: the box minus a cylinder, `max(box, -cyl)`. Box frames for windows.
- Parameters:
  - column spacing 2–6;
  - column radius 0.1–0.5;
  - number of stairs n (2–8) and stair radius r ≤ 0.3;
  - hall width;
  - fold (mirror) angle `pR(p.xz, θ)` before `abs` for kaleidoscopic halls.
- Audio:
  - fold angle ← slow LFO plus snare nudges;
  - emissive grooves via `fOpEngrave` glowing on the kick.

**F. 3D Truchet.**
- Per cubic cell, pick one of 2–8 orientations by `hash(id)`, using mirrors `p.x = -p.x` and so on, or axis permutations.
- Inside, place three quarter-torus segments connecting the centres of opposite faces. The classic version is three tori centred on three cell corners, which makes a continuous network of tubes.
- Exact per cell only if tubes stay inside the cell and meet faces at their centres. They do by construction, so the naive repetition works and no neighbour check is needed.
- Parameters:
  - cell size;
  - tube radius 0.05–0.2·cell;
  - torus major radius = 0.5·cell (fixed by connectivity);
  - optional square tubes via `length8` or a box cross-section.
- Audio:
  - re-seed the hash on section changes (pattern changes);
  - color flows along tubes using the torus angle as a 1D coordinate (e.g. pulses travel along tubes at BPM);
  - tube radius ← bass.

**G. Kaleidoscopic fold space (bridges toward fractals).**
- Repeat `p = abs(p) - offset; pR(p.xy, a1); pR(p.yz, a2);` 2–5 times, then a primitive (box frame or capsule).
- Abs-folds and rotations are exact operations on space, but the image is a bound if the primitive crosses fold planes.
- Very wide variety from just (offset vec3, a1, a2, iterations). Cost is about iterations × (abs + 2 rotations) + 1 primitive.
- Audio: a1/a2 ← slow integrations of mid and high bands; offset ← bass. Keep the rates slow, because small angle changes cascade into big image changes.

### Inferences — artifacts and readability
- **Domain repetition artifacts and fixes:**
  - **Cracks or holes** with per-cell variation: neighbour check, mirrored repetition, or keep the shape within (P/2 − variation margin).
  - **Banding or stepping at cell borders**: the marcher overshoots into the neighbour cell. Clamp the step to about 0.5·P: `t += min(d, 0.5*P)`. It is cheap and fixes most errors when the cell content is close to the boundary. This is a common Shadertoy trick; I did not find a primary citation.
  - **Moiré and shimmer** at distance (sub-pixel repeats): exponential fog `mix(col, fog, 1-exp(-t*density))`, max distance 30–80 units, and fading per-cell detail with distance (e.g. raise rounding or lower displacement as t grows).
  - **Seams from `mod` with negative coordinates**: use hg_sdf's `mod(p+half, size)-half` or iq's `round`-based form. Both are correct for negatives.
- **Readability behind the central model:**
  - the model is lit and bloomed in the project's pipeline, so the background should sit about 1–2 stops darker and be desaturated or tinted;
  - fog is the strongest depth cue;
  - avoid high-frequency repetition directly behind the model's silhouette: a tunnel or cage *around* the view axis leaves the centre empty;
  - the orbit camera makes world-space backgrounds parallax naturally. Large structures (period ≥ 3× model size) read as "space", small ones read as texture or noise;
  - half-resolution rendering plus the existing bloom softens it into a "depth-of-field" layer, which helps separation.
- **Scrolling vs. moving the world:** don't modulate the repetition period P directly with audio. The whole world slides or zooms around the origin and looks like jitter. Modulate cell *content* (radius, rotation, emissive) and advance a scroll offset monotonically.

### Gaps
- Could not retrieve a primary tutorial for 3D Truchet raymarching (Shane's Shadertoy work) or a tunnel breakdown. The Truchet and tunnel recipes are from general knowledge and should be validated against Shadertoy examples.
- No source found quantifying the step-clamp fix for repetition.

---

## 5. A generic "scene template" with continuous parameters: what varies most, integer vs continuous params, what needs recompilation

### Takeaway
One uber-shader works if it is a pipeline of stages with continuous knobs, with integer choices limited to "mode" and "primitive type". The stages:
1. domain transform (rotation, scroll, warp);
2. fold / kaleidoscope (iterations, angles);
3. repetition (type, period, polar order);
4. cell content (primitive blend);
5. combination (smin k with a host);
6. shell / round / onion.

Integer mode switches through a `uniform int` branch are coherent: every pixel takes the same path, so the branch is cheap. But all branches are compiled into one program, which raises register pressure and lowers occupancy. Compiling one program per mode (GLSL `#define MODE`) at startup in moderngl and switching programs is the cleaner option. Morphs between modes can use `mix()` of two SDFs, which stays a valid bound.

### Cited Findings
- iq: if f() is long, the compiler may inline the 4 calls to `map()` in the normal function and crash or slow compilation. Use a loop whose bound depends on a non-constant zero (`#define ZERO (min(iFrame,0))`) to stop inlining. Tetrahedral normals need 4 evaluations instead of 6 for central differences. — [iq, normals for SDFs](https://iquilezles.org/articles/normalsSDF/)
  - Jamie Wong's version uses central differences: 6 SDF evaluations per shaded pixel. — [Jamie Wong](https://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/)
- k in iq's normalized smin maps directly to blend thickness in distance units. That makes it a perceptually linear, audio-friendly knob. — [iq, smooth minimum](https://iquilezles.org/articles/smin/)
- Elongation, rounding and onion are exact and parameterized by continuous values (h vec3, radius, thickness). — [iq, distance functions](https://iquilezles.org/articles/distfunctions/)
- A graded gyroid demo uses three quality tiers (56 / 80 / 88 steps). That is an indication of realistic step budgets for TPMS scenes in the browser. — [three.js forum](https://discourse.threejs.org/t/graded-gyroid-a-lattice-ray-marched-from-a-distance-field-no-mesh/93435)

### Inferences
**Mix of SDFs is safe.** For 1-Lipschitz a and b and t ∈ [0,1], `t·a + (1−t)·b` has gradient ≤ t∥∇a∥ + (1−t)∥∇b∥ ≤ 1. So it is a valid bound (it never overestimates). Its zero set is a genuine in-between shape, not a crossfade. This makes "primitive type blend" and even "scene A to scene B" morphs continuous and safe. The cost is evaluating both SDFs during the morph only: branch on `t>0 && t<1`, which is uniform, so coherent.

**Template sketch (GLSL 330).** U* are uniforms written each frame from Python; `ubo` or plain uniforms both work in moderngl.

```glsl
uniform int   uMode;        // 0 organic-TPMS, 1 metaballs, 2 lattice, 3 tunnel, 4 truchet, 5 fold-space
uniform vec3  uRotAxisSpeed; uniform float uScroll;    // domain transform (exact)
uniform float uWarpAmp, uWarpFreq;                     // non-exact, divide by (1+amp*freq)
uniform int   uFoldIter; uniform vec3 uFoldOffset; uniform vec2 uFoldAngles;
uniform vec3  uPeriod; uniform float uPolarN;          // repetition
uniform vec4  uPrimBlend;   // weights box / sphere / torus / capsule (sum = 1) -> convex mix, safe
uniform vec3  uPrimSize; uniform float uRound, uOnion, uSmoothK, uTwist;
uniform float uTpmsScale, uTpmsBias, uTpmsThick, uTpmsType;  // type as float: mix gyroid->diamond->P
uniform vec4  uBall[16]; uniform int uBallCount;

float map(vec3 p){
  float L = 1.0;                                // accumulated Lipschitz factor
  p = rotate(p, uTime*uRotAxisSpeed); p.z += uScroll;
  if(uWarpAmp>0.0){ p += uWarpAmp*sin(p.yzx*uWarpFreq + uTime); L *= 1.0+uWarpAmp*uWarpFreq; }
  for(int i=0;i<uFoldIter;i++){ p = abs(p)-uFoldOffset; pR(p.xy,uFoldAngles.x); pR(p.yz,uFoldAngles.y); }
  // ... mode-specific repetition + content ...
  return d / L;
}
```

**Parameter classes:**

| Class | Examples | Behaviour | Audio suitability |
|---|---|---|---|
| Exact continuous (free to animate) | rotations, scroll, radius/rounding, onion thickness, smin k, elongation h, primitive blend weights, TPMS bias/thickness, fold angles | No step penalty | Best targets for fast impulses (kick, snare) |
| Bounded continuous (raise L) | warp amp/freq, twist, bend, displacement amp, TPMS scale grading, anisotropy | Need a step factor; artifacts if under-corrected | Slow modulation only (section energy, smoothed RMS), clamped |
| World-scale continuous | repetition period, tunnel radius, TPMS scale | Whole world rescales or slides | Change slowly or on section boundaries; never per beat |
| Integer (uniform) | fold iterations, polar sector count N, ball count, octaves, hash seed | Discrete jumps. N can be faked continuous by `mix` of floor(N) and ceil(N) results (2× cost) | Change on section changes, or crossfade |
| Compile-time (`#define`) | scene mode, max steps, neighbour-check on/off, material model | Separate programs, compiled at startup | Mode switch through a short screen crossfade (render both for ~0.5 s or fade through fog/black) |

**Recompilation.**
- In GLSL 330, loop bounds from uniforms and `if(uMode==…)` both work without recompiling. The costs are code size and register allocation for the worst branch.
- On Apple GPUs (via Apple's OpenGL-over-Metal layer), long uber-shaders compile slowly and can hit occupancy limits. Per-mode programs keep each one lean.
- moderngl `ctx.program()` compiles once per mode at startup (a few hundred ms in total). Switching programs per frame is free.
- I could not verify Apple-specific occupancy behaviour (see Gaps).

**Step budget.**
- Plan for 48–96 steps, a hit epsilon proportional to t (e.g. `eps = 0.001*t` for a background), max distance 40–80, and fog to hide misses.
- For TPMS (divided by √3), steps are about 1.5–1.7× more numerous than for a true SDF. Over-relaxation (ω about 1.2–1.5 with fallback) can claw that back (Keinert et al.; ω values per Bán & Valasek, §2).
- Background at half resolution: on a 2880×1800 Retina framebuffer, half-res is 1440×900 ≈ 1.3 Mpx. At 64 steps × a 30-instruction map, that is about 2.5 G instructions per frame. That is plausible within 1–3 ms on an M-series GPU, but unmeasured.

**What makes looks "truly different" from one shader**, in order of impact (inference):
1. Repetition topology: none / cartesian / polar / folds.
2. TPMS type and bias crossing a topological threshold.
3. Primitive family.
4. smin k (separate vs merged).
5. Fold iterations.
6. Palette and material model.
7. Twist and warp amounts.

A preset system can randomize within the safe ranges of the table above. Integer topology choices give the big jumps, continuous ones the evolution.

### Gaps
- No source found on Apple-Silicon GLSL uber-shader compile times, register pressure or occupancy in the OpenGL 4.1/Metal translation layer. No measured timings of raymarched backgrounds on M-series GPUs were found.

---

## 6. Materials and coloring for organic/geometric SDF scenes: cosine palettes, per-cell colors, fresnel, iridescence, emissive edges, near-miss glow

### Takeaway
Cheap, expressive coloring comes from:
- iq's cosine palette `a + b·cos(2π(c·t + d))`, indexed by a scalar the SDF already produces: cell ID hash, TPMS value or bias sign, smin blend factor, march distance, step count, sector index or Truchet tube angle;
- a fresnel/rim term;
- emissive lines (bands where a secondary distance is small);
- glow accumulated from near misses during the march.

All of these are nearly free compared with extra map evaluations, and they are the best audio targets because they don't affect step safety.

### Cited Findings
- **Cosine palette:**
  - `vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d){ return a + b*cos(6.283185*(c*t+d)); }`;
  - for exact cycling over 0..1, make c an integer number of halves; for C1 continuity, make c an integer;
  - example (a, b, c, d) sets: (0.5, 0.5, 0.5 / 0.5, 0.5, 0.5 / 1, 1, 1 / 0.00, 0.33, 0.67) gives the rainbow; (… / 1, 1, 1 / 0.00, 0.10, 0.20); (… / 1, 1, 0.5 / 0.80, 0.90, 0.30); (0.8, 0.5, 0.4 / 0.2, 0.4, 0.2 / 2, 1, 1 / 0.00, 0.25, 0.25);
  - GPU cosine is "way, way faster than any attempt to approximate it".

  — [iq, palettes](https://iquilezles.org/articles/palettes/)
- **Smin mix factor** (`.y` of the vec2 smin) blends the materials of the two blended shapes (snow to stone in iq's example). — [iq, smooth minimum](https://iquilezles.org/articles/smin/)
- **Domain-warp intermediates** q and r are reused for coloring: mix a ramp by f, then by length(q), then by r.y. — [iq, warping](https://iquilezles.org/articles/warp/)
- **Per-cell color** from the repetition id hash. — [iq, domain repetition](https://iquilezles.org/articles/sdfrepetition/)
- **Iteration count** as occlusion: `1.0 - loop/maxLoop` serves as a lightweight occlusion factor in a Unity raymarching library. — [uRaymarching (libraries.io)](https://libraries.io/npm/com.hecomi.uraymarching)
- **Soft shadows** are almost free by tracking the minimum ratio d/t along a ray. — [Cambridge lecture](https://www.cl.cam.ac.uk/teaching/1718/FGraphics/1.%20Ray%20Marching%20and%20Signed%20Distance%20Fields.pdf)
- **Iridescent raymarched glass metaballs** exist as a design pattern (a web component). — [21st.dev](https://21st.dev/community/components/cldgayo/iridescent-raymarched-glass-metaballs/default)

### Inferences
- **Near-miss glow** (standard Shadertoy practice, no primary source retrieved): inside the march loop, `glow += exp(-d*sharp) * stepWeight` or `glow += 0.01/(0.01 + d*d)`, then `col += glowColor * glow * intensity`. It costs nothing beyond the march.
  - It gives neon or volumetric looks even when rays miss everything. That is ideal for a background, because missed pixels still carry signal.
  - Audio: intensity ← kick/RMS; glowColor ← palette(beat phase).
  - Normalize by step count, or the glow depends on the step budget and on the Lipschitz factor (more steps where the field is shrunk).
- **Emissive lines / edges:**
  - **TPMS:** `line = smoothstep(w, 0.0, abs(fract(f*bands) - 0.5))` on the raw TPMS value, or on a second TPMS at a different scale. That gives contour lines crawling over the surface; animate bias or phase with the beat.
  - **Repetition:** emissive where `min(abs(q.x), abs(q.y))` of the cell-local coordinate is small (grid seams), or on box frames.
  - **hg_sdf `fOpEngrave` / `fOpPipe`:** compute the groove or pipe distance separately and use it as an emissive mask.
  - **Tunnel:** emissive per sector `id` when `id == beatCounter % N` gives a rotating light chase locked to the beat.
- **Fresnel / rim:** `pow(1.0 - max(dot(n, -rd), 0.0), 3..5)` drives rim light and the iridescence hue.
- **Iridescence (cheap thin-film fake):** `palette(fresnel*c + thickness_term + time*s, a, b, c, d)`, with a rainbow-ish palette and c = 1–2. The "thickness term" can be the TPMS value or noise. It reads as soap-film or oil. Physically based thin-film interference is more expensive and probably unnecessary for a background.
- **Per-cell colors:** `palette(hash(id) + time*0.05 + band[id%7]*0.3, …)`. Brightness per cell from its assigned EQ band makes the repeated scene a spatial spectrum analyser without changing geometry. That is very readable and step-safe.
- **Depth and fog coloring:**
  - fog color from the same palette at t = 0.5;
  - the project's AgX tonemap and bloom (T21) amplify emissive values above 1 into halos. Keep background emissive in the 0.5–3 range so it blooms less than the model;
  - the background pass should write linear HDR to match the project's linear pipeline.
- **Lighting budget:**
  - one directional light plus a cheap ambient: `0.5 + 0.5*n.y`, times AO from the step count;
  - the 4-evaluation tetrahedral normal with the non-inlining loop trick (§5);
  - skip soft shadows in the background (each costs another march);
  - for metaballs, normals can be computed analytically from the exponential-smin weights (each ball's normal weighted by exp2(−d_i/k)), saving the 4 extra map evaluations. This derivation is mine, not verified in a source.

### Gaps
- No primary source retrieved for near-miss glow accumulation or for a cheap iridescence approximation in shaders. Both are described from general practice.
- No source quantifying how much the glow accumulation changes with Lipschitz-scaled steps.
