# Fractal DE / SDF backgrounds (FRACTALS mode) for a real-time raymarched music-visualizer background

Scope: distance-estimated 3D fractals that a GLSL 330 full-screen fragment shader (moderngl, macOS / Apple Silicon) can raymarch behind a 3D model, with many continuous parameters and audio-driven animation.

Source note: the Syntopia blog ("Distance Estimated 3D Fractals" series, blog.hvidtfeldts.net) could not be read directly. HTTPS failed with a TLS certificate mismatch, plain HTTP was blocked by the host's web firewall ("455 Security Incident Detected"), and web.archive.org is blocked for this agent. To make up for it, the notes rely on the **primary source code** of Syntopia's own tool **Fragmentarium** (GitHub, examples written by Syntopia, Knighty, Kali and Rrrola) and on **Mandelbulber2's formula sources** (GitHub). Both are verbatim, authoritative implementations. Shadertoy pages need JavaScript or an API key and could not be fetched. Shader code recalled from memory is kept in the Gaps sections and marked as unverified.

---

## 1. Fractal family catalogue: DE formula, parameters, ranges, visual effect, cost, pitfalls, audio targets

### Takeaway
Two DE patterns cover almost every family. In the **escape-time pattern** (Mandelbulb, quaternion Julia), the DE is `0.5·log(r)·r/dr`, with a running scalar derivative `dr`. In the **IFS / fold pattern** (Mandelbox, KIFS Menger / Sierpinski / octahedron / icosahedron, pseudo-Kleinian, Amazing Surf, Kalisets), a running scale factor is kept in `p.w` or `DEfactor`, and the DE is `length(z)·Scale^-n` or `(length(p)-|S-1|)/p.w`. All the reference implementations expose a few continuous uniforms: scale, min radius, fold offsets, a Julia constant, and a rotation axis and angle applied *inside* the loop. Those are exactly the knobs that vary the look widely.

### Cited Findings

**Mandelbulb (escape-time, power-N triplex)**
- Fragmentarium Mandelbulb DE. In spherical coordinates `theta=acos(z.z/r)`, `phi=atan(z.y,z.x)`. The derivative is `dr = pow(r,Power-1)*Power*dr + 1`. The new point is `z = r^Power * (sin(θP)cos(φP), sin(φP)sin(θP), cos(θP))`, then `z += (Julia ? JuliaC : pos)`. The loop runs `while (r<Bailout && i<Iterations)`, and the result is `return 0.5*log(r)*r/dr` — [Fragmentarium Mandelbulb.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Historical%203D%20Fractals/Mandelbulb.frag)
- Parameter ranges in that file: Power slider 0–16 (8 is "standard"), Bailout 0–30 (default 5, presets 6.279), Iterations 0–100 (default 9, presets 12–14). A `JuliaC` vec3 sits in [-2,2]. A rotation `z*=rot` (axis `RotVector`, angle `RotAngle` 0–180°) is applied *every iteration*. `init()` builds `rot` with `RotAngle+time*10.0`, so rotation inside the loop is already used as an animation. One preset uses Power 8.18304 with RotVector (1,0,0) and RotAngle 77.8° — [Mandelbulb.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Historical%203D%20Fractals/Mandelbulb.frag)
- There are two power functions. The standard one is spherical with acos. The "Alternate" one (by Enforcer) uses `asin(z.z/r)` with cos/sin swapped, and the file says it "looks more like a Mandelbrot for Power=2". The poles differ between the two variants, so a boolean switch gives a second family of shapes — [Mandelbulb.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Historical%203D%20Fractals/Mandelbulb.frag)
- iq uses power 8 and notes that higher powers give more symmetric shapes. The gradient of the Hubbard-Douady potential is computed analytically inside the loop and can double as the shading normal. On GPUs the trig (polar) version is fine, since GPUs have fast trig. The trig-free polynomial version is "~5x" faster mainly on CPUs. Real-time on GPU at about 720p with shadows. Reference Shadertoy: ltfSWn. iq also shows a morph through the bulb's associated Julia sets as an animation — [iq, Mandelbulb](https://iquilezles.org/articles/mandelbulb/)

**Quaternion Julia (4D, escape-time)**
- Fragmentarium code: `p=vec4(pos,0)`, `dp=vec4(1,0,0,0)`. Each iteration does `dp = 2·p·dp` (quaternion product) and `p = p² + C`, breaking when `dot(p,p) > Threshold`. The DE is `0.5*r*log(r)/length(dp)`. Defaults: Iterations 16, Threshold 10, C = (0.18, 0.88, 0.24, 0.16) with each component in [-1,1]. The file credits Subblue's implementation, which builds on a CG shader by Keenan Crane — [Fragmentarium QuaternionJulia.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Historical%203D%20Fractals/QuaternionJulia.frag)

**Mandelbox (Tom Lowe, box fold + sphere fold)**
- Definition: `v = s·ballFold(r, f·boxFold(v)) + c`. The box fold reflects each component above 1 to `2−v` and below −1 to `−2−v`. The ball fold works on the magnitude m: if `m<r` then `m = m/r²`, else if `m<1` then `m = 1/m`. The standard Mandelbox is s=2, r=0.5, f=1. For 1<|s|<2 the box-counting dimension is 3 (solid centre) — [Tom Lowe, What is a Mandelbox](https://sites.google.com/site/mandelbox/what-is-a-mandelbox)
- Rrrola's optimised GLSL DE (the DE idea is Buddhi's) appears in Fragmentarium as follows: `vec4 p=vec4(pos,1), p0=p; scale=vec4(S,S,S,|S|)/MinRad2;` then in the loop `p.xyz*=rot; p.xyz=clamp(p.xyz,-1,1)*2-p.xyz; r2=dot; p*=clamp(max(MinRad2/r2,MinRad2),0,1); p=p*scale+p0; if(r2>1000) break;` and finally `return (length(p.xyz)-|S-1|)/p.w - |S|^(1-Iterations)`. In this form the folding limit and the fixed radius are both 1, and MinRad2 is the squared min radius — [Fragmentarium Mandelbox.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Historical%203D%20Fractals/Mandelbox.frag)
- Ranges in that file: Scale −3…5 (comment: "a perfect Menger is 3.0"), MinRad2 0…2 (default 0.25), Iterations 0…300 (default 17). There is also a rotation inside the loop. Presets: (Iter 10, MinRad2 0.25, Scale 2.39128), (Iter 18, MinRad2 0.0172, Scale −1.49272, bounding sphere 7.59) and (Iter 12, MinRad2 0.25, Scale 2.04344). Positive and negative scales give very different shapes — [Mandelbox.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Historical%203D%20Fractals/Mandelbox.frag)
- Mandelbulber's Mandelbox family uses a default bailout of 100 and a linear analytic DE (`aux.DE = aux.DE*|m| + 1`) — [Mandelbulber2 fractal_amazing_surf.cpp](https://github.com/buddhi1980/mandelbulber2/blob/master/mandelbulber2/formula/definition/fractal_amazing_surf.cpp)

**Kalibox (Kali's Mandelbox variant)**
- Same structure as Rrrola's Mandelbox, but the box fold is replaced by `p.xyz = abs(p.xyz) + Trans` and the added constant is a fixed `Julia` instead of `p0`. Defaults: Scale 1.3 (range −3…3), Trans (0.5,0.5,0.5) in ±5, Julia (−1,−1,−1) in [−5,0], MinRad2 0.25, plus a rotation inside the loop. The DE is the same as the Mandelbox — [Fragmentarium Kalibox.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Kali%27s%20Creations/Kalibox.frag)

**Menger sponge (Knighty KIFS)**
- Loop: `z=rot*z; z=abs(z);` then sort the components in descending order (`if(z.x<z.y) swap xy; if(z.x<z.z) swap xz; if(z.y<z.z) swap yz`), then `z = Scale*z - Offset*(Scale-1); if (z.z < -0.5*Offset.z*(Scale-1)) z.z += Offset.z*(Scale-1);`. The DE is `length(z)*pow(Scale,-n)`. Scale slider 0–4 (3 gives the exact Menger), Offset in [0,1]³, Iterations default 8. A "broken symmetry" preset uses Scale 3.1, RotVector (1, 0.408, 0.184), RotAngle 2.1366°, Offset (1, 0.956, 1) and 13 iterations — [Fragmentarium Menger.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Kaleidoscopic%20IFS/Menger.frag)

**Sierpinski tetrahedron / octahedron / icosahedron (Knighty KIFS)**
- Tetrahedron: `z*=rot1; if(z.x+z.y<0) z.xy=-z.yx; if(z.x+z.z<0) z.xz=-z.zx; if(z.y+z.z<0) z.zy=-z.yz; z = z*Scale - Offset*(Scale-1); z*=rot2;` with DE `length(z)*pow(Scale,-n)`. Defaults Scale 2 (0–4), Offset (1,1,1), Iterations 13. There are **two** rotations, one before the fold and one after the scale, each with an axis in [−1,1]³ and an angle in ±180° — [Fragmentarium Tetrahedron.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Kaleidoscopic%20IFS/Tetrahedron.frag)
- Octahedron: same template with four folds (`x+y<0 → -yx`, `x+z<0 → -zx`, `x−y<0 → yx`, `x−z<0 → zx`) and default Offset (1,0,0) — [Fragmentarium Octahedron.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Kaleidoscopic%20IFS/Octahedron.frag)
- Icosahedron: fold-plane normals are built from Phi (slider −5…5, default 1.618): `n1=normalize(-Phi,Phi-1,1)`, `n2=normalize(1,-Phi,Phi+1)`, `n3=(0,0,-1)`. Prefolds run once (`abs`, then n1, n2, n3, n2 reflections `if(dot(z,n)>0) z-=2*dot(z,n)*n`). Each iteration does `abs`, the n1 reflection, then a **single 4×4 matrix** `M = rot2·translate(Offset)·scale(Scale)·translate(−Offset)·rot1`, computed once in `init()`. Default Offset is (0.8507, 0.5257, 0). Folding all the affine work into one precomputed matrix is the efficient way to pass "rotate–scale–translate" as one uniform — [Fragmentarium Icosahedron.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Kaleidoscopic%20IFS/Icosahedron.frag)

**Pseudo-Kleinian (Knighty / Theli-at)**
- The comment describes it as a "Scale 1 JuliaBox + Something". The loop runs `MI` iterations (default 5, range 0–20): `p = 2*clamp(p,-CSize,CSize) - p; r2=dot(p,p); k=max(Size/r2,1); p*=k; DEfactor*=k; p+=C;`. It exits early when `ap==p`. The final DE is a basic shape scaled by `DEfactor`: by default a plane, `abs(0.5*abs(p.z-Offset.z)/DEfactor - DEoffset)`. A "Thingy" (`(abs(length(p.xy)*p.z)-e)/sqrt(dot(p,p)+|e|)`) and a RoundBox are commented-out alternatives, and `#define USE_INF_NORM` is set. Parameters: Size 0–2 (default 1), CSize vec3 0–2 (default 1), Julia C in ±2, TThickness 0–2, DEoffset 0–0.01, Offset in ±1 — [Fragmentarium PseudoKleinian.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Knighty%20Collection/PseudoKleinian.frag)
- Mandelbulber2 ports the same formula, citing the Fragmentarium file as its reference, with a dedicated `pseudoKleinianDEFunction`. It adds optional per-iteration-range transforms such as sphere inversion, a prism fold and an extra box fold, each enabled between start and stop iterations — [Mandelbulber2 fractal_pseudo_kleinian.cpp](https://github.com/buddhi1980/mandelbulber2/blob/master/mandelbulber2/formula/definition/fractal_pseudo_kleinian.cpp)

**Amazing Surf (Kali, with features by DarkBeam, from Mandelbulb3D)**
- It is a Mandelbox with **no z fold**. Box fold on x and y only: `z.x = |z.x+a.x| - |z.x-a.x| - z.x` (likewise for y, with `a` = additionConstant111, i.e. 1). Then `rr=dot(z,z)`, with an optional "force cylinder fold" `rr -= z.z²`. Then `dividend = rr<sqrt(minR) ? sqrt(minR) : min(rr,1)`, `m = scale/dividend`, `z *= (m-1)*scale1 + 1`, `DE = DE*|m| + 1`. The constant c is added with c.x and c.y swapped, and a rotation is applied every iteration. Default bailout 100, linear analytic DE. The result is a sheet-like, surface-type fractal — [Mandelbulber2 fractal_amazing_surf.cpp](https://github.com/buddhi1980/mandelbulber2/blob/master/mandelbulber2/formula/definition/fractal_amazing_surf.cpp)

**Kaliset / Kalisets1 / ABoxModKali**
- Kali's original Kaliset loop (posted 2012-07-10) is `p=abs(p); p=p/dot(p,p); p=p*Scale+Julia;`. It accumulates `expsmooth += exp(-1/abs(lnprev-ln))`, where `ln=length(p)`. This value is **not a DE**. It was used for colouring (`orbitTrap = k*ColorScale+ColorOffset`) and for small bump distortion (`p += k*Strength`) on top of a simple DE such as a sphere. Strength is "usually very small" — [fractalforums, Kaliset 3D used as coloring and bump mapping](https://www.fractalforums.com/3d-fractal-generation/kaliset-3d-fractal-used-as-coloring-and-bump-mapping-for-de-systems)
- Mandelbulber's "Kalisets1 Mod", based on M3D Kalisets1 and KaliDucks, does give a DE: `z=abs(z); sqs=|z|²; m = sqs<minR ? 1/sqrt(minR) : scale/sqs; z*=m; DE=DE*|m|+1; z += c or juliaC; rotate`. Default bailout 10. The M3D note says to try Julia mode and low bailouts from 2 down to 1, and that a cutoff at z=0 may be needed to see anything — [Mandelbulber2 fractal_kalisets1.cpp](https://github.com/buddhi1980/mandelbulber2/blob/master/mandelbulber2/formula/definition/fractal_kalisets1.cpp)
- ABoxModKali (from M3D): an optional rotation, then `z = additionConstant0555 - abs(z)`, then a sphere fold with `minR06` and `m = scale015/dividend`, `DE=DE*|m|+1` — [Mandelbulber2 fractal_abox_mod_kali.cpp](https://github.com/buddhi1980/mandelbulber2/blob/master/mandelbulber2/formula/definition/fractal_abox_mod_kali.cpp)

**Living KIFS (Kali) — a "generic fold+scale+rotate" in practice**
- The loop has three folding modes: `p.x=abs(p.x+Fold.x)-Fold.x` (one axis), the same on `.xy` (two axes), or on `.xyz` (three axes). Each iteration then does `p=p*Sc+Julia; p*=rot;`. The DE is `length(p)*pow(Sc,-Iterations) - Fatness`. Ranges: Scale 1–3, Fold vec3 ±3, Julia vec3 ±3 (default −0.5), RotVector in ±1, RotAngle ±180°, Fatness 0–2 (it thickens the shape by subtracting a constant from the DE) — [Fragmentarium LivingKIFS.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Kali%27s%20Creations/LivingKIFS.frag)
- The default preset needs FudgeFactor 0.3494, MaxRaySteps 308 and Detail −2.66, a sign of how much these generic KIFS overstep — [LivingKIFS.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Kali%27s%20Creations/LivingKIFS.frag)

**Hybrids**
- Hvidtfeldt's epilogue: hybrids mix building blocks from different fractals and are very popular in Mandelbulb 3D. Nothing prevents applying a different transformation at each iteration. For general combinations there is no easy DE: the analytic DE often overshoots and needs a fudge factor — [Syntopia, DE 3D Fractals Part VIII: Epilogue (via search summary)](https://blog.hvidtfeldts.net/index.php/2012/05/distance-estimated-3d-fractals-part-viii-epilogue/)
- Mandelbulber formulas carry `startIterations` / `stopIterations` gates on their sub-transforms, so one transform can run only on some iterations. That is how hybrid "sequences" are built inside one formula — [Mandelbulber2 fractal_abox_mod_kali.cpp](https://github.com/buddhi1980/mandelbulber2/blob/master/mandelbulber2/formula/definition/fractal_abox_mod_kali.cpp); [fractal_pseudo_kleinian.cpp](https://github.com/buddhi1980/mandelbulber2/blob/master/mandelbulber2/formula/definition/fractal_pseudo_kleinian.cpp)

**Quality pitfalls**
- The classic Mandelbulb-style estimator is strictly valid only for conformal maps. The 3D Mandelbulb variants and the 2D abs variants are non-conformal, which causes overstepping, and lowering the fudge factor alleviates it — [fractalforums, about the 4 point DE (via search summary)](https://www.fractalforums.com/3d-fractal-generation/about-the-4-point-de)
- With some hybrids, lowering the DE step factor does not always reduce holes in the surface: the error curve reaches a minimum and then rises again — [fractalforums, distorted surface in 3D fractals (holes) (via search summary)](https://www.fractalforums.com/general-discussion/distorted-surface-in-3d-fractals-(holes))

### Inferences
- **Cost ranking (inference from the loop bodies above).** KIFS folds (abs, compares, one mat3 or mat4 multiply) are the cheapest per iteration. The Mandelbox and Amazing Surf add a dot product, a division and a clamp. The pseudo-Kleinian is cheap per iteration, and its default of only 5 iterations makes it the cheapest "rich" fractal. The Mandelbulb is the most expensive (acos, atan, pow, sin and cos per iteration) and needs 8–12 iterations. So for a 1–3 ms half-resolution budget, KIFS, pseudo-Kleinian and Mandelbox at about 8–12 iterations are the safest options, and the Mandelbulb at 6–8.
- **Audio targets, smooth ones** (inferred from what each parameter does in the cited code; to verify in practice):
  - Mandelbulb: Power within ±0.5 of the base value (it morphs continuously; integer powers have higher symmetry), in-loop RotAngle, and JuliaC (Julia mode only).
  - Mandelbox: Scale within one sign (it crosses into a different shape class at |S|<1 and changes completely at the sign flip), MinRad2 between 0.05 and 0.5, and the in-loop rotation angle.
  - KIFS: Offset components, the two rotation angles (small values; 0–10° already changes the look strongly, as the Menger preset with 2.1° shows), and Scale within ±0.1 of the base value.
  - Pseudo-Kleinian: CSize components, Size, and C.
  - Amazing Surf and Kalibox: Trans/Fold and Julia.
- **Jumpy or dangerous targets**: iteration count (an integer, so it pops; use a fractional blend between n and n+1 or keep it fixed), Mandelbox scale sign changes, Power going below about 2, and Bailout (it changes the DE scale abruptly). Kick and snare impulses are better sent to the in-loop rotation angle or the fold offset with a spring-decay envelope than to scale.

### Gaps
- iq's "Apollonian" Shadertoy (4ds3zn) source could not be fetched. Unverified recollection: `for 8 iters { p = -1 + 2*fract(0.5*p+0.5); r2=dot(p,p); k=s/r2; p*=k; scale*=k; } return 0.25*abs(p.y)/scale;` with `s` around 1.0–1.5 animated over time, and orbit trap `min(orb, vec4(abs(p),r2))`. Verify against the Shadertoy source before using.
- Kali's "Fractal Land" Shadertoy (XsBXWt, a fly-through tunnel built from KIFS folds) could not be found or fetched, so its parameters are unverified.
- The Mandelbox shape classes by scale value (−1.5 "classic negative", 2, 3) are only partially documented in what could be fetched. Tom Lowe has separate pages, such as "Negative 1.5 Mandelbox", that were not fetched.
- Numeric defaults for Amazing Surf scale and minR were not found in the definition file. The parameter names `minR05`, `scale015`, `minR06` and `additionConstant0555` suggest 0.5, 1.5, 0.6 and 0.5 but are not confirmed.
- Juliabulb specifics (Mandelbulb Julia constant ranges) beyond Fragmentarium's JuliaC slider of ±2 were not found.

---

## 2. A generic, parameterizable "fold–scale–rotate" KIFS template (one shader, many looks, no recompilation)

### Takeaway
The reference shaders share one skeleton: optional rotation, then one or more folds, then an optional sphere fold, then a scale about an offset or plus a Julia constant, then optional rotation, while tracking a running scale. One uniform-driven loop with fold "weights" and precomputed matrices can therefore cover the Menger, Sierpinski, octahedral and icosahedral KIFS, Mandelbox, Kalibox, Amazing Surf, pseudo-Kleinian and Living KIFS.

### Cited Findings
- The Knighty KIFS template is `rot1 → symmetry folds → z = Scale*z − Offset*(Scale−1) → rot2`, with `DE = length(z)·Scale^−n`. Only the fold set changes between the tetrahedron (3 plane folds), the octahedron (4) and the Menger (abs plus sort) — [Tetrahedron.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Kaleidoscopic%20IFS/Tetrahedron.frag); [Octahedron.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Kaleidoscopic%20IFS/Octahedron.frag); [Menger.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Kaleidoscopic%20IFS/Menger.frag)
- The rotate–translate–scale–translate–rotate chain can be collapsed on the CPU into one `mat4 M` uniform (`M = rot2·T(Offset)·S(Scale)·T(−Offset)·rot1`), so the per-iteration cost is a single matrix multiply — [Icosahedron.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Kaleidoscopic%20IFS/Icosahedron.frag)
- The Mandelbox family uses the same skeleton with the box fold `clamp(p,−L,L)*2 − p` and the sphere fold `p *= clamp(max(MinRad2/r2, MinRad2), 0, 1)`, tracking scale in `p.w` — [Mandelbox.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Historical%203D%20Fractals/Mandelbox.frag)
- Swapping the box fold for `abs(p)+Trans` gives the Kalibox [Kalibox.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Kali%27s%20Creations/Kalibox.frag). Box-folding only x and y gives Amazing Surf [fractal_amazing_surf.cpp](https://github.com/buddhi1980/mandelbulber2/blob/master/mandelbulber2/formula/definition/fractal_amazing_surf.cpp). Setting scale to 1 and using `max(Size/r2, 1)` as the inversion gives the pseudo-Kleinian [PseudoKleinian.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Knighty%20Collection/PseudoKleinian.frag)
- Living KIFS shows that a fold offset `abs(p+Fold)−Fold` on 1, 2 or 3 axes, plus `p*Sc+Julia` and an in-loop rotation, already spans a large shape space. A "Fatness" term subtracted from the DE thickens thin structures, and a position-dependent scale (`Sc = clamp(Scale + p·ScaleVary·…, MinMaxScale)`) breaks self-similarity — [LivingKIFS.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Kali%27s%20Creations/LivingKIFS.frag)

### Inferences
- A proposed single-shader template (a GLSL 330 sketch synthesised from the cited loops, not tested):
  ```glsl
  uniform int   uIter;        // 6..14
  uniform mat3  uRotPre, uRotPost;   // built on the CPU, angles audio-driven
  uniform vec3  uFold;        // abs(p+uFold)-uFold offset (Living KIFS)
  uniform float wAbs, wTetra, wSort, wBox, wSphere;  // 0..1 fold weights
  uniform float uBoxLimit, uMinR2, uFixR2;           // Mandelbox folds
  uniform float uScale;       // -3..3.5
  uniform vec3  uOffset;      // KIFS scaling centre
  uniform vec3  uJulia; uniform float wJulia;        // p += mix(p0, uJulia, wJulia)
  uniform float uFat;
  float DE(vec3 p0, out vec4 trap){
    vec4 p = vec4(p0,1.0); trap = vec4(1e9);
    for(int i=0;i<14;i++){ if(i>=uIter) break;
      p.xyz = uRotPre*p.xyz;
      p.xyz = mix(p.xyz, abs(p.xyz+uFold)-uFold, wAbs);
      if(wTetra>0.5){ if(p.x+p.y<0.) p.xy=-p.yx; if(p.x+p.z<0.) p.xz=-p.zx; if(p.y+p.z<0.) p.zy=-p.yz; }
      if(wSort>0.5){ if(p.x<p.y)p.xy=p.yx; if(p.x<p.z)p.xz=p.zx; if(p.y<p.z)p.yz=p.zy; }
      p.xyz = mix(p.xyz, clamp(p.xyz,-uBoxLimit,uBoxLimit)*2.0-p.xyz, wBox);
      float r2 = dot(p.xyz,p.xyz);
      float k  = mix(1.0, clamp(uFixR2/max(r2,uMinR2), 1.0, uFixR2/uMinR2), wSphere);
      p *= k;
      p.xyz = p.xyz*uScale - uOffset*(uScale-1.0);  p.w *= abs(uScale);
      p.xyz += mix(vec3(0), mix(p0, uJulia, wJulia), step(0.001, wJulia+wBox));
      p.xyz = uRotPost*p.xyz;
      trap = min(trap, vec4(abs(p.xyz), r2));
    }
    return length(p.xyz)/p.w - uFat;
  }
  ```
  Fold weights between 0 and 1 make **continuous morphs** possible between fold types (the abs and box branches). Discrete folds (tetra, sort) are better toggled during a beat-synchronised flash or a fade to black. Mixing fold types breaks the exactness of the DE, so a global fudge factor of about 0.5–0.8 is needed (see LivingKIFS's 0.35).
- Uniform layout: about 25–35 floats per frame, comparable to what `gfx/renderer.py` already uploads for the model, so the CPU and upload cost is negligible.
- Building `uRotPre` and `uRotPost` (Rodrigues) on the CPU each frame avoids trig in the shader and reuses the project's existing axis-angle code. The renderer already uses Rodrigues for the model axis, per `architecture.md`.

### Gaps
- Fractal Lab (Subblue) source and parameter list were not retrieved. Its generic "fold + scale + rotate + translate" formula would be a direct reference.
- No published "universal KIFS" with a verified DE bound was found. The fudge factor for the mixed template has to be tuned empirically.

---

## 3. Coloring: orbit traps, iteration count, cosine palettes, step-count glow, AO

### Takeaway
Every reference implementation keeps a `vec4 orbitTrap = min(orbitTrap, abs(vec4(p.xyz, r2)))` over the first `ColorIterations` iterations. Feeding those four values (or `log` of them) into iq's cosine palette gives rich colour at almost no cost. Ambient occlusion and glow come for free from the raymarch step count and the origin trap.

### Cited Findings
- iq's Mandelbulb colouring uses four traps: the origin and the planes x=0, y=0 and z=0. The origin trap is a multiplicative factor that **simulates ambient occlusion**. The three plane traps blend the base colour with three other colours, and the colour follows the fractal's structure — [iq, Mandelbulb](https://iquilezles.org/articles/mandelbulb/)
- iq's 3D orbit-trap article: track the orbit's closest approach to a shape (cylinder, plane, sine wave, sphere). Using a trap as geometry distorts the distance metric, so divide by the gradient or slow the raymarcher down — [iq, 3D orbit traps](https://iquilezles.org/articles/orbittraps3d/)
- Fragmentarium examples use `orbitTrap = min(orbitTrap, abs(vec4(z, dot(z,z))))`, limited to the first `ColorIterations` iterations (Mandelbox default 3, Mandelbulb 9, Icosahedron 3). The raymarcher clears the trap when steps reach MaxRaySteps — [Mandelbox.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Historical%203D%20Fractals/Mandelbox.frag); [DE-Raytracer.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Include/DE-Raytracer.frag)
- Living KIFS offers alternative colouring modes: a sum of `exp(-1/|l−prevl+ColorOffset|)` (Kaliset-style smooth colouring) or the average of `|l−prevl+ColorOffset|` over the colour iterations, scaled by ColorScale — [LivingKIFS.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Kali%27s%20Creations/LivingKIFS.frag)
- The Kaliset value can colour or bump-map any DE surface, and it can be computed only at the hit point rather than at every ray step (Fragmentarium's `providesColor`) — [fractalforums Kaliset coloring](https://www.fractalforums.com/3d-fractal-generation/kaliset-3d-fractal-used-as-coloring-and-bump-mapping-for-de-systems)
- Cosine palette: `col = a + b*cos(6.283185*(c*t + d))`. Example (a, b, c, d) sets: (0.5,0.5,0.5 | 0.5,0.5,0.5 | 1,1,1 | 0,0.33,0.67), (… | 1,1,1 | 0,0.10,0.20), (… | 1,1,1 | 0.30,0.20,0.20), (… | 1,1,0.5 | 0.8,0.9,0.3), (… | 1,0.7,0.4 | 0,0.15,0.20), (… | 2,1,0 | 0.5,0.2,0.25), (0.8,0.5,0.4 | 0.2,0.4,0.2 | 2,1,1 | 0,0.25,0.25). For an exact wrap, c should be a multiple of 0.5. Cosines are cheap on the GPU. The palette is meant for colourising scalar fields such as fractal iteration data — [iq, palettes](https://iquilezles.org/articles/palettes/)
- Fragmentarium's raytracer has AO "based on the number of raymarching steps" and a Glow colour with a GlowMax step threshold — [DE-Raytracer.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Include/DE-Raytracer.frag)

### Inferences
- Palette phase `d` and offset `t` are ideal **smooth audio targets**. Sending them to the beat phase or to the section energy gives colour motion with no geometry change and no DE risk. `c` (palette frequency) can follow the spectral content.
- Step-count glow (`glow += f(steps)/MaxSteps`, or accumulating `exp(-k*d)` along the ray) gives a cheap "bloom-like" halo that already reacts to geometry. Tied to RMS, it suits a background, because the renderer's bloom pass (T21) amplifies it.
- For a background behind a lit model, a fully shaded background (normals, shadows) may not be needed. Trap colour × step-AO × fog by distance can stay within budget without a normal calculation, which saves 4–6 DE evaluations per pixel.

### Gaps
- iq's orbit-trap article does not give the code that maps trap values to colour (it is on Shadertoy 3tsyzl, which could not be fetched).

---

## 4. Animation: continuous parameters, morphing, rotation inside the loop, camera inside vs. object far away

### Takeaway
The reference shaders animate by rotating space *inside* the iteration loop (`rot` angle + time), by sine-modulating the rotation axis, and by adding cosine "waves" to the position before iterating. All three are continuous and stay compatible with the DE. Changes to integer iteration counts and discrete fold switches cause popping.

### Cited Findings
- Fragmentarium's Mandelbulb builds its rotation as `rotationMatrix3(normalize(RotVector), RotAngle + time*10.0)` and applies it inside the loop. This is a stock example of time-varying rotation within the iteration — [Mandelbulb.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Historical%203D%20Fractals/Mandelbulb.frag)
- Living KIFS adds several animation hooks: `ani.xyz = sin(time*Speed + p·SpeedVary)·Amplitude` added to the in-loop rotation axis, so the rotation axis varies over time and space. There is also a global pre-rotation `TRotSpeed*time`, a translation `TransSpeed*time` (fly-through), an "impulse" `p += dir*(0.8+sin(time*Rate+Offset))*Strength` (a pulse-like push), and a wave displacement `p += cos(p*WaveLength/wz + time*WaveSpeed*10)*WaveAmp*0.5*wz` — [LivingKIFS.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Kali%27s%20Creations/LivingKIFS.frag)
- iq animates the Mandelbulb by morphing through its associated Julia sets (changing c) and by zooming close to the surface — [iq, Mandelbulb](https://iquilezles.org/articles/mandelbulb/)
- Mandelbulber can ramp a transform's offset with iteration index (`tempVC += offset*currentIteration/iterationRange`), so a rotation can grow with depth in the iteration. This is another continuous control knob — [fractal_abox_mod_kali.cpp](https://github.com/buddhi1980/mandelbulber2/blob/master/mandelbulber2/formula/definition/fractal_abox_mod_kali.cpp)
- Translation-based fly-through (camera moving through space) is built into Living KIFS through TransVector and TransSpeed — [LivingKIFS.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Kali%27s%20Creations/LivingKIFS.frag)

### Inferences
- **Mapping to the project's audio features** (all fed through the existing `app/routing.py` envelopes and springs):
  - KICK spring → in-loop rotation angle (±2–8°) or fold-offset impulse (the Living KIFS impulse model).
  - SNARE → palette phase jump or glow flash.
  - Section energy → Scale, or Mandelbulb Power (slow, ±0.3).
  - Beat phase → palette `d`.
  - SUB / BASS bands → Fatness (thickening) or MinRad2.
  - HIGH bands → wave displacement amplitude (small).
  - BPM → global pre-rotation or fly-through speed.
- **Morphing between fractals.** Within the generic template, interpolate fold weights and continuous parameters over 1–4 bars. Between unrelated families (Mandelbulb versus KIFS), the DEs can be blended (`mix(d1, d2, t)`) or combined as a smooth union for the duration of the transition. Both evaluations cost money only during the transition, and the blended field is not a strict bound, so a lower fudge factor is needed then.
- **Camera inside vs. object far away.**
  - *Object far behind the model*: a bounded fractal (bulb, box, KIFS) placed at a distance acts as a "planet" or "sculpture". It is cheap, because most rays miss and exit through a bounding sphere.
  - *Camera inside or fly-through*: infinite or space-filling forms (pseudo-Kleinian, Apollonian, Amazing Surf, Kalibox, Mandelbox interior with negative scale) fill the screen. Every pixel then marches to a hit, so the cost per pixel is higher.
  - The orbit camera (T26) gives parallax in both cases. With the model in front, the background could use its own slower virtual camera (a parallax factor below 1) so that it reads as distant.
- **Per-pixel precision when zooming deep**: float32 limits deep zoom to roughly 1e-4–1e-5 relative scale. A "flying inside" mode should loop its motion (periodic fold structures such as `fract`-based Apollonian or pseudo-Kleinian repeat exactly) rather than zoom without limit.

### Gaps
- No primary source was found that documents seamless "infinite zoom" for 3D DE fractals in real time. The loop-in-periodic-structure idea above is an inference.

---

## 5. Real-time performance: iterations, steps, fudge, half resolution, fps

### Takeaway
Reference real-time settings: 5–18 fractal iterations, 56 (default) to about 300 ray steps, a hit epsilon around 10^-2.3, epsilon growing with distance, and a fudge factor of 1 for exact KIFS and Mandelbox and about 0.35–0.8 for hybrids and generic KIFS. No reliable published fps figures for Apple Silicon or mid-range GPUs were found.

### Cited Findings
- Fragmentarium's default raytracer settings: `Detail` −2.3 (minDist = 10^Detail ≈ 0.005), MaxRaySteps 56 (slider up to 2000), FudgeFactor default 1 (range 0–1), Dither 0.5 (randomises the first step to hide banding), and a distance-dependent epsilon `eps * totalDist^ClarityPower`. The normal is evaluated at a back-stepped hit point (`NormalBackStep`) — [DE-Raytracer.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Include/DE-Raytracer.frag)
- The Living KIFS preset needs FudgeFactor 0.35 and 308 steps — [LivingKIFS.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Kali%27s%20Creations/LivingKIFS.frag)
- The Mandelbox breaks out early when `r2 > 1000`, the pseudo-Kleinian when `ap == p` (a fixed point), and the quaternion Julia when `p2 > Threshold`. Early exit cuts the average iteration cost — [Mandelbox.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Historical%203D%20Fractals/Mandelbox.frag); [PseudoKleinian.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Knighty%20Collection/PseudoKleinian.frag); [QuaternionJulia.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Historical%203D%20Fractals/QuaternionJulia.frag)
- iq reports a GPU real-time Mandelbulb at about 720p with shadows and no antialiasing — [iq, Mandelbulb](https://iquilezles.org/articles/mandelbulb/)
- A Mandelbulb-based WebGL benchmark names the main performance levers: kernel iterations, ray step size, and resolution scale — [volumeshader.dev GPU test (via search summary)](https://www.volumeshader.dev/en/gpu-test)
- Hybrids often need a fudge factor because their analytic DE overshoots — [Syntopia epilogue (via search summary)](https://blog.hvidtfeldts.net/index.php/2012/05/distance-estimated-3d-fractals-part-viii-epilogue/)

### Inferences
- **Budget estimate (rough, not measured).** Half-resolution Retina for a 1440×900-point window is 1440×900 pixels, about 1.3 M pixels. At 64 steps × 10 iterations × about 10 flops for a KIFS, that is about 8 GFLOP per frame, which fits an M1/M2 GPU (≈2.6–3.6 TFLOPS fp32) in roughly 3–5 ms in theory. Real throughput is lower because of divergent early exits. So the 1–3 ms target probably needs:
  - **quarter-resolution** (half width and height of the logical size) with bilateral or depth-aware upsampling;
  - **48–80 steps**, with **8–10 iterations**;
  - a **bounding sphere** early-out for bounded fractals;
  - **no normals or shadows** (colour from traps plus step AO), or normals through a cheap 4-tap tetrahedral gradient;
  - **temporal reuse**: march every other pixel per frame in a checkerboard and reproject, or start the march from the previous frame's distance. This is an inference; check how it interacts with the orbit camera.
- Because the background sits behind an opaque model, rays covered by the model can be skipped with a depth or stencil pre-pass. The renderer already has an MSAA depth buffer (T21) that could be downsampled into a mask.
- Banding: dither the first step (Fragmentarium Dither 0.5) and use fog by distance so that missed rays fade out. Step-limited rays should get the fog or background colour, not black.
- Profile on the target Mac with `tools/snapshot.py`-style offline captures plus GPU timer queries (moderngl `Query(time=True)`). This is an inference; the API exists in moderngl but was not verified here.

### Gaps
- No credible published fps or ms numbers were found for DE fractals on Apple Silicon or mid-range GPUs. The two theses found (Cal Poly VR Mandelbulb, Chalmers user-defined fractals) give no numbers in their abstracts.
- Whether macOS OpenGL 4.1 (deprecated, Metal-backed) gives good dynamic-loop performance for these shaders is unverified and must be measured.

---

## 6. Reference tools and famous shaders for parameter sets

### Takeaway
Fragmentarium (Syntopia, with the forks FragM / 3Dickulus) and Mandelbulber2 are the best machine-readable references. Their GLSL and C++ formula sources and presets give exact loops, slider ranges and tested values. Shadertoy classics (iq's Mandelbulb ltfSWn, Apollonian 4ds3zn, Knighty's pseudo-Kleinian, Kali's Fractal Land XsBXWt) are good visual references but were not fetchable here.

### Cited Findings
- The Fragmentarium examples repository contains, among others:
  - "Historical 3D Fractals": Mandelbox, Mandelbulb, QuaternionJulia;
  - "Kaleidoscopic IFS": Menger, NewMenger, Tetrahedron, Octahedron, Icosahedron;
  - "Kali's Creations": Kalibox, LivingKIFS, RotJulia, Treebroccoli, Xray_skifs;
  - "Knighty Collection": PseudoKleinian, PseudoKleinianMenger, Fold-and-Cut Polyhedra, Mandalabeth, Doyle spirals, hyperbolic tessellations, polychora;
  - "Theory": Mandelbox Dual-Numbers DE.

  All are GLSL with slider ranges and presets — [Fragmentarium repository (GitHub)](https://github.com/Syntopia/Fragmentarium/tree/master/Fragmentarium-Source/Examples)
- Mandelbulber2 implements Amazing Surf (Kali / DarkBeam, from M3D), Kalisets1 Mod (Kalisets1 / KaliDucks), ABoxModKali, Pseudo Kleinian, Mandelbox, Mandelbulb, Menger sponge, Sierpinski3D, Quaternion, Aexion, Benesi, Mandelbulb Kali, Xenodreambuie and others. Each file names its source thread on fractalforums.com — [Mandelbulber2 formula definitions (GitHub)](https://github.com/buddhi1980/mandelbulber2/tree/master/mandelbulber2/formula/definition)
- iq's Mandelbulb has a real-time Shadertoy implementation: https://www.shadertoy.com/view/ltfSWn — [iq, Mandelbulb](https://iquilezles.org/articles/mandelbulb/)
- fractalforums.com is archived (read-only) and the community moved to fractalforums.org. The Kaliset thread dates from July 2012 — [fractalforums Kaliset thread](https://www.fractalforums.com/3d-fractal-generation/kaliset-3d-fractal-used-as-coloring-and-bump-mapping-for-de-systems)

### Inferences
- The Fragmentarium `.frag` presets (Scale, MinRad2, Offset, RotVector and RotAngle values above) can be ported almost directly as named presets in `presets.default.json`. Examples: "Mandelbox +2.39", "Mandelbox −1.49", "Menger brisé 3.1", "Tétra 2", "Pseudo-Kleinian".
- Licence check before copying code: Fragmentarium examples are distributed with the Fragmentarium source, under a licence not verified here. Mandelbulber2 is GPLv3, so do not copy its code into a non-GPL project; the formulas themselves (maths) can be re-implemented. Shadertoy code defaults to CC BY-NC-SA 3.0 unless the author states otherwise.

### Gaps
- Mandelbulb3D's formula list and Subblue's Fractal Lab parameter set were not retrieved.
- The exact source of Knighty's pseudo-Kleinian Shadertoy and Kali's Fractal Land could not be read, so it is unknown how they differ from the Fragmentarium versions.
