# benchmarks/howell/analytic.jl
# Closed-form configuration factors from J. R. Howell, "A Catalog of Radiation
# Heat Transfer Configuration Factors", 3rd ed. (thermalradiation.net),
# Section C: finite area to finite area.
#
# Each function is named for its catalog case and reproduces the governing
# equation exactly as published on that case's page. Variable names follow the
# catalog's own "Definitions" line so the transcription can be checked by eye
# against the source.

# C-1: two infinitely long, directly opposed parallel plates of equal width w,
# separation h.  H = h/w.
c1(w, h) = (H = h / w; sqrt(1 + H^2) - H)

# C-2: two infinitely long parallel plates of different widths b (plate 1) and
# c (plate 2), separated by a, with the plate centrelines joined by the common
# perpendicular.  B = b/a, C = c/a.
c2(a, b, c) = (B = b / a; C = c / a;
               (sqrt((B + C)^2 + 4) - sqrt((C - B)^2 + 4)) / (2B))

# C-3: two infinitely long plates of unequal width with one common edge at 90°.
# A1 has width w, A2 has width h, H = h/w.  F is A1 → A2.
c3(w, h) = (H = h / w; 0.5 * (1 + H - sqrt(1 + H^2)))

# C-4: two infinitely long plates of equal width with a common edge and
# included angle α (radians).
c4(alpha) = 1 - sin(alpha / 2)

# C-11: identical, parallel, directly opposed a×b rectangles separated by c.
# X = a/c, Y = b/c.
function c11(a, b, c)
  X = a / c; Y = b / c
  (2 / (pi * X * Y)) * (
    log(sqrt((1 + X^2) * (1 + Y^2) / (1 + X^2 + Y^2)))
    + X * sqrt(1 + Y^2) * atan(X / sqrt(1 + Y^2))
    + Y * sqrt(1 + X^2) * atan(Y / sqrt(1 + X^2))
    - X * atan(X) - Y * atan(Y)
  )
end

# C-14: two finite rectangles of the same length l with one common edge, at 90°.
# A1 is l×w, A2 is l×h.  H = h/l, W = w/l.  F is A1 → A2.
function c14(l, w, h)
  W = w / l; H = h / l
  (1 / (W * pi)) * (
    W * atan(1 / W) + H * atan(1 / H)
    - sqrt(H^2 + W^2) * atan(1 / sqrt(H^2 + W^2))
    + 0.25 * log(
      ((1 + W^2) * (1 + H^2) / (1 + W^2 + H^2))
      * (W^2 * (1 + W^2 + H^2) / ((1 + W^2) * (W^2 + H^2)))^(W^2)
      * (H^2 * (1 + H^2 + W^2) / ((1 + H^2) * (H^2 + W^2)))^(H^2)
    )
  )
end

# C-41: disk of radius r1 to parallel coaxial disk of radius r2, separation a.
# R1 = r1/a, R2 = r2/a, X = 1 + (1 + R2^2)/R1^2.  F is disk 1 → disk 2.
# C-40 (equal radii) is the r1 == r2 special case of this formula.
function c41(r1, r2, a)
  R1 = r1 / a; R2 = r2 / a
  X = 1 + (1 + R2^2) / R1^2
  0.5 * (X - sqrt(X^2 - 4 * (r2 / r1)^2))
end

c40(r, a) = c41(r, r, a)

# C-79: base of a right circular cylinder to the cylinder's inside lateral
# surface.  H = h/(2r).  F is base → lateral surface.
c79(r, h) = (H = h / (2r); 2H * (sqrt(1 + H^2) - H))

# C-109: interior lateral surface of a right circular cone to its base.
# H = h/r.  F is cone interior → base.
c109(r, h) = 1 / sqrt(1 + (h / r)^2)

# C-125: sphere to a coaxial disk of radius r whose plane lies a distance a
# from the sphere centre.  R = r/a.  F is sphere → disk, and is independent of
# the sphere radius (a convex body's view factor is a solid-angle fraction).
c125(r, a) = 0.5 * (1 - 1 / sqrt(1 + (r / a)^2))

# C-135: concentric spheres, inner radius r1, outer r2.
c135_12(r1, r2) = 1.0                 # inner → outer
c135_21(r1, r2) = (r1 / r2)^2         # outer → inner
c135_22(r1, r2) = 1 - (r1 / r2)^2     # outer → itself

# C-63: concentric cylinders of infinite length, inner diameter D1, outer D2.
c63_12(D1, D2) = 1.0            # inner → outer
c63_21(D1, D2) = D1 / D2        # outer → inner
c63_22(D1, D2) = 1 - D1 / D2    # outer → itself

# C-68: infinitely long parallel cylinders of equal radius r, surface-to-surface
# gap s.  X = 1 + s/(2r).
c68(r, s) = (X = 1 + s / (2r); (sqrt(X^2 - 1) + asin(1 / X) - X) / pi)

# C-69: infinitely long parallel cylinders of different radius, surface-to-surface
# gap s (see the C-69 figure: s is measured between the cylinder surfaces, not
# between the axes).  R = r2/r1, S = s/r1, C = 1 + R + S.  F is cylinder 1 → 2.
function c69(r1, r2, s)
  R = r2 / r1; S = s / r1; C = 1 + R + S
  (pi
   + sqrt(C^2 - (R + 1)^2)
   - sqrt(C^2 - (R - 1)^2)
   + (R - 1) * acos((R / C) - (1 / C))
   - (R + 1) * acos((R / C) + (1 / C))) / (2 * pi)
end
