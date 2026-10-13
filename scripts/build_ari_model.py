"""Author Ari's editable full-body Blender prototype and a self-contained GLB.

Run with Blender --background --python scripts/build_ari_model.py.
The reference art remains the likeness target; this is an original authored rig,
not a claimed automatic reconstruction of the illustration.
"""

import itertools
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "characters/ari/model"
OUT.mkdir(parents=True, exist_ok=True)
PI = math.pi
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 24
scene.cycles.use_denoising = True
scene.render.resolution_x = 760
scene.render.resolution_y = 1100
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.fps = 30
scene.frame_end = 600
scene.world.color = (0.25, 0.25, 0.25)
scene.view_settings.view_transform = "AgX"


def material(name, color, roughness=0.65, metallic=0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    return mat


skin = material("Warm porcelain", (0.76, 0.49, 0.39), 0.54)
skin.node_tree.nodes.get("Principled BSDF").inputs["Subsurface Weight"].default_value = 0.08
hair = material("Jet black hair", (0.004, 0.005, 0.009), 0.48)
hair_light = material("Hair ribbon highlights", (0.008, 0.010, 0.016), 0.48)
for mat in (hair, hair_light):
    mat.node_tree.nodes.get("Principled BSDF").inputs["Specular IOR Level"].default_value = 0.16
ivory = material("Ivory cardigan", (0.79, 0.75, 0.65), 0.92)
ivory_seam = material("Cardigan ribbing", (0.65, 0.60, 0.51), 0.94)
charcoal = material("Charcoal knit", (0.033, 0.036, 0.044), 0.94)
trousers = material("Black tailored trousers", (0.014, 0.016, 0.022), 0.89)
shoes = material("Black leather flats", (0.012, 0.014, 0.019), 0.35)
gold = material("Small gold details", (0.66, 0.43, 0.15), 0.27, 0.7)
white = material("Eyes", (0.91, 0.87, 0.83), 0.38)
iris = material("Slate gray iris", (0.15, 0.18, 0.21), 0.3)
iris_outer = material("Iris outer ring", (0.032, 0.043, 0.057), 0.4)
pupil = material("Pupil and lashes", (0.009, 0.007, 0.012), 0.53)
lip = material("Soft rose lips", (0.53, 0.22, 0.20), 0.46)
mouth_dark = material("Mouth interior", (0.07, 0.016, 0.026), 0.95)
teeth = material("Teeth", (0.84, 0.78, 0.69), 0.58)

meshes = []


def mesh(name, vertices, faces, mat, weights=None):
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices, [], faces)
    data.update()
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    for polygon in data.polygons:
        polygon.use_smooth = True
    if weights:
        groups = {}
        for i, point in enumerate(vertices):
            for bone, weight in weights(point).items():
                if weight > 0:
                    group = groups.setdefault(
                        bone, obj.vertex_groups.get(bone) or obj.vertex_groups.new(name=bone)
                    )
                    group.add([i], weight, "REPLACE")
    meshes.append(obj)
    return obj


def fixed(bone):
    return lambda _: {bone: 1}


def mix_z(z, low, high, a, b):
    weight = max(0, min(1, (z - low) / (high - low)))
    return {a: 1 - weight, b: weight}


def torso_weights(p):
    if p[2] < 1.12:
        return mix_z(p[2], 0.99, 1.12, "Hips", "Spine")
    return mix_z(p[2], 1.12, 1.34, "Spine", "Chest")


def rings(name, sections, mat, weights, segments=40, open_front=0):
    vertices, faces = [], []
    for z, xcenter, ycenter, rx, ry in sections:
        for j in range(segments + 1):
            angle = open_front + (2 * PI - 2 * open_front) * j / segments
            vertices.append((xcenter + rx * math.sin(angle), ycenter - ry * math.cos(angle), z))
    for row in range(len(sections) - 1):
        for j in range(segments):
            a = row * (segments + 1) + j
            faces.append((a, a + 1, a + segments + 2, a + segments + 1))
    if not open_front:
        faces.extend(
            [
                tuple(reversed(range(segments + 1))),
                tuple(range((len(sections) - 1) * (segments + 1), len(vertices))),
            ]
        )
    return mesh(name, vertices, faces, mat, weights)


def tube(name, points, radii, mat, bone, sides=8):
    vertices, faces = [], []
    for i, p in enumerate(points):
        p = Vector(p)
        tangent = Vector(points[min(i + 1, len(points) - 1)]) - Vector(points[max(0, i - 1)])
        tangent.normalize()
        cross = tangent.cross(Vector((0, 1, 0)))
        if cross.length < 0.01:
            cross = tangent.cross(Vector((1, 0, 0)))
        cross.normalize()
        other = tangent.cross(cross).normalized()
        radius = radii[i] if isinstance(radii, list) else radii
        for j in range(sides):
            v = p + radius * (
                math.cos(j * 2 * PI / sides) * cross + math.sin(j * 2 * PI / sides) * other
            )
            vertices.append(tuple(v))
    for i in range(len(points) - 1):
        for j in range(sides):
            a = i * sides + j
            b = i * sides + (j + 1) % sides
            faces.append((a, b, b + sides, a + sides))
    faces.extend(
        [tuple(reversed(range(sides))), tuple(range(len(vertices) - sides, len(vertices)))]
    )
    return mesh(name, vertices, faces, mat, fixed(bone))


def ellipsoid(name, center, scale, mat, bone, segments=32, rows=20):
    vertices, faces = [], []
    for i in range(rows + 1):
        v = PI * i / rows
        for j in range(segments):
            u = 2 * PI * j / segments
            vertices.append(
                (
                    center[0] + scale[0] * math.sin(v) * math.cos(u),
                    center[1] + scale[1] * math.sin(v) * math.sin(u),
                    center[2] + scale[2] * math.cos(v),
                )
            )
    for i in range(rows):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            faces.append((a, b, b + segments, a + segments))
    return mesh(name, vertices, faces, mat, fixed(bone))


def shape(obj, name, transform):
    if obj.data.shape_keys is None:
        obj.shape_key_add(name="Basis", from_mix=False).value = 0
    key = obj.shape_key_add(name=name, from_mix=False)
    key.value = 0
    for point in key.data:
        point.co = transform(point.co.copy())
    return key


def bezier(points, steps=24):
    a, b, c, d = [Vector(v) for v in points]
    return [
        tuple((1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t * t * c + t**3 * d)
        for t in [i / steps for i in range(steps + 1)]
    ]


def lock(name, controls, width, mat=hair, bone="Head"):
    points = bezier(controls, 28)
    vertices, faces = [], []
    for i, point in enumerate(points):
        t = i / (len(points) - 1)
        w = width * (0.38 + 0.62 * math.sin(PI * (t * 0.8 + 0.1))) * max(0.025, (1 - t) ** 0.45)
        for j in range(7):
            u = (j - 3) / 3
            vertices.append((point[0] + w * u, point[1] - w * 0.25 * (1 - u * u), point[2]))
    for i in range(len(points) - 1):
        for j in range(6):
            a = i * 7 + j
            faces.append((a, a + 1, a + 8, a + 7))
    obj = mesh(name, vertices, faces, mat, fixed(bone))
    return obj


# Rest skeleton: natural standing pose; local +Z is up, the face looks toward -Y.
arm = bpy.data.armatures.new("Ari humanoid skeleton")
rig = bpy.data.objects.new("AriRig", arm)
scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bones = {}


def bone(name, start, end, parent=None):
    b = arm.edit_bones.new(name)
    b.head, b.tail = start, end
    if parent:
        b.parent = bones[parent]
    bones[name] = b


bone("Root", (0, 0, 0), (0, 0, 0.15))
bone("Hips", (0, 0, 0.95), (0, 0, 1.06), "Root")
bone("Spine", (0, 0, 1.06), (0, 0, 1.24), "Hips")
bone("Chest", (0, 0, 1.24), (0, 0, 1.41), "Spine")
bone("Neck", (0, 0, 1.41), (0, 0, 1.50), "Chest")
bone("Head", (0, 0, 1.50), (0, 0, 1.70), "Neck")
for side, sign in (("L", 1), ("R", -1)):
    bone("Shoulder." + side, (0, 0, 1.405), (sign * 0.172, 0, 1.405), "Chest")
    bone("UpperArm." + side, (sign * 0.172, 0, 1.405), (sign * 0.222, 0, 1.125), "Shoulder." + side)
    bone(
        "LowerArm." + side,
        (sign * 0.222, 0, 1.125),
        (sign * 0.266, -0.015, 0.91),
        "UpperArm." + side,
    )
    bone(
        "Hand." + side,
        (sign * 0.266, -0.015, 0.91),
        (sign * 0.273, -0.025, 0.81),
        "LowerArm." + side,
    )
    bone("UpperLeg." + side, (sign * 0.092, 0, 0.96), (sign * 0.093, -0.009, 0.52), "Hips")
    bone(
        "LowerLeg." + side,
        (sign * 0.093, -0.009, 0.52),
        (sign * 0.094, 0, 0.105),
        "UpperLeg." + side,
    )
    bone(
        "Foot." + side, (sign * 0.094, 0, 0.105), (sign * 0.094, -0.125, 0.055), "LowerLeg." + side
    )
    bone("Hair." + side, (sign * 0.07, 0.025, 1.61), (sign * 0.12, 0.02, 1.19), "Head")
bpy.ops.object.mode_set(mode="OBJECT")
rig.select_set(False)
rig.show_in_front = True

# Torso, cardigan, and ribbed collar. Profiles are editable cross-sections.
rings(
    "Knit torso",
    [
        (0.95, 0, 0, 0.121, 0.076),
        (1.035, 0, 0, 0.110, 0.063),
        (1.12, 0, 0, 0.098, 0.057),
        (1.22, 0, -0.012, 0.122, 0.075),
        (1.29, 0, -0.015, 0.137, 0.082),
        (1.36, 0, 0, 0.145, 0.066),
        (1.405, 0, 0, 0.162, 0.045),
        (1.43, 0, 0, 0.064, 0.039),
    ],
    charcoal,
    torso_weights,
)
rings(
    "Ribbed high collar",
    [(1.40, 0, 0, 0.049, 0.043), (1.455, 0, 0, 0.047, 0.041), (1.47, 0, 0, 0.044, 0.038)],
    charcoal,
    fixed("Neck"),
)
rings("Neck skin", [(1.447, 0, 0, 0.039, 0.033), (1.515, 0, 0, 0.037, 0.032)], skin, fixed("Neck"))
rings(
    "Open ivory cardigan",
    [
        (0.86, 0, 0.014, 0.158, 0.088),
        (0.91, 0, 0.014, 0.159, 0.087),
        (1.04, 0, 0.013, 0.139, 0.082),
        (1.15, 0, 0.01, 0.131, 0.083),
        (1.29, 0, 0.004, 0.151, 0.088),
        (1.37, 0, 0.006, 0.173, 0.067),
        (1.414, 0, 0.006, 0.168, 0.049),
        (1.435, 0, 0.008, 0.068, 0.04),
    ],
    ivory,
    torso_weights,
    open_front=0.64,
)
for sign in (-1, 1):
    tube(
        "Cardigan opening seam",
        [
            (sign * 0.097, -0.058, 0.86),
            (sign * 0.085, -0.065, 1.04),
            (sign * 0.077, -0.074, 1.15),
            (sign * 0.090, -0.082, 1.29),
            (sign * 0.080, -0.055, 1.37),
            (sign * 0.043, -0.037, 1.435),
        ],
        0.0035,
        ivory_seam,
        "Chest",
    )
for index in range(17):
    x = (index - 8) * 0.012
    tube(
        "Knit rib",
        [
            (x, -0.074, 1.04),
            (x * 0.92, -0.064, 1.12),
            (x * 1.12, -0.099, 1.25),
            (x * 1.14, -0.09, 1.31),
            (x * 0.7, -0.056, 1.40),
        ],
        0.00065,
        charcoal,
        "Chest",
        sides=4,
    )

# Belt, trousers and legs retain real geometry for side/back views.
rings(
    "Trouser hips",
    [
        (0.82, 0, 0.003, 0.111, 0.071),
        (0.90, 0, 0.003, 0.137, 0.085),
        (0.98, 0, 0.003, 0.131, 0.078),
        (1.025, 0, 0.001, 0.11, 0.065),
    ],
    trousers,
    fixed("Hips"),
)
rings(
    "Waist band",
    [(1.002, 0, 0, 0.114, 0.068), (1.027, 0, 0, 0.111, 0.066)],
    trousers,
    fixed("Hips"),
)
ellipsoid("Waist button", (0, -0.069, 1.014), (0.006, 0.002, 0.006), shoes, "Hips", 16, 8)
for side, sign in (("L", 1), ("R", -1)):

    def arm_weights(p, side=side):
        return mix_z(p[2], 1.09, 1.16, "LowerArm." + side, "UpperArm." + side)

    def leg_weights(p, side=side):
        return mix_z(p[2], 0.46, 0.58, "LowerLeg." + side, "UpperLeg." + side)

    rings(
        "Cardigan sleeve." + side,
        [
            (1.40, sign * 0.17, 0, 0.057, 0.063),
            (1.34, sign * 0.187, 0, 0.063, 0.066),
            (1.24, sign * 0.204, 0, 0.060, 0.061),
            (1.15, sign * 0.22, 0, 0.054, 0.055),
            (1.12, sign * 0.225, -0.002, 0.060, 0.055),
            (1.09, sign * 0.230, -0.004, 0.050, 0.049),
            (1.06, sign * 0.237, -0.006, 0.056, 0.049),
            (1.02, sign * 0.245, -0.010, 0.038, 0.036),
            (0.992, sign * 0.25, -0.012, 0.034, 0.032),
        ],
        ivory,
        arm_weights,
        segments=32,
    )
    rings(
        "Cardigan cuff." + side,
        [(0.990, sign * 0.25, -0.012, 0.035, 0.032), (1.035, sign * 0.243, -0.009, 0.039, 0.035)],
        ivory,
        fixed("LowerArm." + side),
        segments=32,
    )
    rings(
        "Forearm." + side,
        [
            (0.91, sign * 0.266, -0.015, 0.020, 0.019),
            (0.95, sign * 0.259, -0.014, 0.024, 0.022),
            (1.01, sign * 0.249, -0.010, 0.029, 0.026),
        ],
        skin,
        fixed("LowerArm." + side),
        segments=24,
    )
    # Palms face forward in this rest pose: the thumbs must sit on the lateral side.
    ellipsoid(
        "Palm." + side,
        (sign * 0.273, -0.018, 0.865),
        (0.028, 0.014, 0.052),
        skin,
        "Hand." + side,
        24,
        16,
    )
    for index, length in enumerate((0.041, 0.055, 0.061, 0.052)):
        x = sign * (0.253 + index * 0.012)
        tube(
            "Finger." + side,
            [
                (x, -0.022, 0.836),
                (x + sign * 0.002, -0.027, 0.81),
                (x + sign * 0.004, -0.025, 0.836 - length),
                (x + sign * 0.004, -0.021, 0.830 - length),
            ],
            [0.006, 0.0057, 0.0048, 0.0035],
            skin,
            "Hand." + side,
        )
    tube(
        "Thumb." + side,
        [
            (sign * 0.294, -0.026, 0.883),
            (sign * 0.308, -0.033, 0.866),
            (sign * 0.316, -0.037, 0.845),
            (sign * 0.316, -0.037, 0.830),
        ],
        [0.009, 0.008, 0.006, 0.004],
        skin,
        "Hand." + side,
    )
    rings(
        "Trouser leg." + side,
        [
            (0.095, sign * 0.094, 0, 0.068, 0.06),
            (0.14, sign * 0.094, 0, 0.07, 0.061),
            (0.30, sign * 0.093, 0, 0.070, 0.063),
            (0.48, sign * 0.093, -0.008, 0.064, 0.062),
            (0.57, sign * 0.091, -0.006, 0.068, 0.066),
            (0.73, sign * 0.085, 0.001, 0.083, 0.076),
            (0.88, sign * 0.078, 0.004, 0.081, 0.079),
        ],
        trousers,
        leg_weights,
        segments=32,
    )
    ellipsoid(
        "Ankle." + side,
        (sign * 0.094, -0.009, 0.096),
        (0.033, 0.04, 0.06),
        skin,
        "Foot." + side,
        24,
        12,
    )
    ellipsoid(
        "Flat shoe." + side,
        (sign * 0.094, -0.046, 0.044),
        (0.048, 0.11, 0.037),
        shoes,
        "Foot." + side,
        32,
        16,
    )
    ellipsoid(
        "Sole." + side,
        (sign * 0.094, -0.046, 0.022),
        (0.049, 0.11, 0.013),
        shoes,
        "Foot." + side,
        32,
        10,
    )

# Continuous head surface with a small sculpted bridge and nose tip.
profiles = [
    (1.48, 0.016, 0.038),
    (1.49, 0.036, 0.052),
    (1.515, 0.061, 0.071),
    (1.545, 0.079, 0.081),
    (1.58, 0.086, 0.087),
    (1.615, 0.091, 0.092),
    (1.655, 0.09, 0.089),
    (1.69, 0.079, 0.080),
    (1.72, 0.056, 0.062),
    (1.74, 0.016, 0.024),
]
sections = []
for a, b in itertools.pairwise(profiles):
    for i in range(5):
        t = i / 5
        sections.append(tuple(a[k] * (1 - t) + b[k] * t for k in range(3)))
sections.append(profiles[-1])
verts, faces = [], []
for z, width, depth in sections:
    for j in range(80):
        angle = j * 2 * PI / 80
        x, y = width * math.sin(angle), -depth * math.cos(angle)
        front = max(0, math.cos(angle)) ** 10
        y -= front * (
            0.022 * math.exp(-(((z - 1.563) / 0.012) ** 2) - (x / 0.012) ** 2)
            + 0.010 * math.exp(-(((z - 1.591) / 0.031) ** 2) - (x / 0.010) ** 2)
        )
        verts.append((x, y, z))
for row in range(len(sections) - 1):
    for j in range(80):
        a, b = row * 80 + j, row * 80 + (j + 1) % 80
        faces.append((a, b, b + 80, a + 80))
face = mesh("Face", verts, faces, skin, fixed("Head"))
shape(
    face,
    "JawOpen",
    lambda p: (
        p
        + Vector(
            (0, 0.001, -0.009 * math.exp(-(((p.z - 1.512) / 0.029) ** 2)) * max(0, -p.y / 0.08))
        )
    ),
)
shape(
    face,
    "Happy",
    lambda p: (
        p
        + Vector((0, -0.002, 0.002))
        * math.exp(-(((abs(p.x) - 0.048) / 0.022) ** 2) - ((p.z - 1.56) / 0.025) ** 2)
        * max(0, -p.y / 0.08)
    ),
)
for side, sign in (("L", 1), ("R", -1)):
    ellipsoid(
        "Ear." + side, (sign * 0.086, 0.001, 1.573), (0.014, 0.020, 0.027), skin, "Head", 24, 16
    )
    circle = [
        (sign * (0.094 + 0.006 * math.cos(t)), -0.018, 1.546 + 0.010 * math.sin(t))
        for t in [j * 2 * PI / 32 for j in range(33)]
    ]
    tube("Gold hoop." + side, circle, 0.0018, gold, "Head", 6)

# Almond eye surfaces, shaped lid edges, iris depth and small catchlights.
for side, sign in (("L", 1), ("R", -1)):
    cx, cz = sign * 0.040, 1.602
    vertices = [(cx, -0.093, cz)]
    for i in range(49):
        theta = i * 2 * PI / 48
        x = 0.028 * math.cos(theta)
        z = (0.012 if math.sin(theta) > 0 else 0.008) * math.sin(theta) + sign * x * 0.10
        vertices.append((cx + x, -0.087 + 0.012 * (x / 0.028) ** 2, cz + z))
    obj = mesh(
        "Eye white." + side, vertices, [(0, i, i + 1) for i in range(1, 49)], white, fixed("Head")
    )
    shape(
        obj,
        "Blink" + side,
        lambda p, cz=cz, cx=cx, sign=sign: Vector(
            (
                p.x,
                p.y,
                cz + (p.x - cx) * sign * 0.10 + (p.z - cz - (p.x - cx) * sign * 0.10) * 0.015,
            )
        ),
    )
    shape(obj, "EyeWide", lambda p, cz=cz: Vector((p.x, p.y, cz + (p.z - cz) * 1.28)))
    for name, radius, y, mat in [
        ("Iris rim", 0.0105, -0.094, iris_outer),
        ("Iris", 0.009, -0.095, iris),
        ("Pupil", 0.0047, -0.096, pupil),
        ("Catchlight", 0.0027, -0.097, white),
    ]:
        x = cx + (0.003 if name == "Catchlight" else 0)
        z = cz + (0.004 if name == "Catchlight" else 0)
        eye = ellipsoid(name + "." + side, (x, y, z), (radius, 0.001, radius), mat, "Head", 24, 12)
        shape(eye, "Blink" + side, lambda p, cz=cz: Vector((p.x, p.y, cz + (p.z - cz) * 0.001)))
        shape(eye, "LookLeft", lambda p: p + Vector((-0.005, 0, 0)))
        shape(eye, "LookRight", lambda p: p + Vector((0.005, 0, 0)))
        shape(eye, "LookUp", lambda p: p + Vector((0, 0, 0.003)))
        shape(eye, "LookDown", lambda p: p + Vector((0, 0, -0.003)))
    for upper in (True, False):
        points = []
        for i in range(25):
            u = i / 24 * 2 - 1
            points.append(
                (
                    cx + u * 0.028,
                    -0.088 + 0.012 * u * u,
                    cz
                    + sign * u * 0.0028
                    + (1 if upper else -1)
                    * (0.012 if upper else 0.008)
                    * math.sqrt(max(0, 1 - u * u)),
                )
            )
        lid = tube(
            ("Upper lashes." if upper else "Lower lid.") + side,
            points,
            0.0016 if upper else 0.0007,
            pupil if upper else lip,
            "Head",
            6,
        )
        shape(
            lid,
            "Blink" + side,
            lambda p, cz=cz, cx=cx, sign=sign: Vector(
                (
                    p.x,
                    p.y,
                    cz + (p.x - cx) * sign * 0.10 + (p.z - cz - (p.x - cx) * sign * 0.10) * 0.015,
                )
            ),
        )
        shape(lid, "EyeWide", lambda p, cz=cz: Vector((p.x, p.y, cz + (p.z - cz) * 1.28)))
    brow_points = [
        (cx + u * 0.027, -0.076 + 0.009 * u * u, 1.632 + 0.004 * (1 - u * u) + sign * u * 0.002)
        for u in [i / 20 * 2 - 1 for i in range(21)]
    ]
    brow = tube(
        "Brow." + side,
        brow_points,
        [0.0005 + 0.0013 * math.sin(PI * i / 20) for i in range(21)],
        hair,
        "Head",
        6,
    )
    shape(brow, "Surprised", lambda p: p + Vector((0, 0, 0.011)))
    shape(
        brow,
        "Concerned",
        lambda p, cx=cx, sign=sign: p + Vector((0, 0, 0.002 - 0.007 * (p.x - cx) * sign / 0.027)),
    )
    shape(
        brow,
        "Annoyed",
        lambda p, cx=cx, sign=sign: (
            p + Vector((0, -0.001, -0.003 + 0.006 * (p.x - cx) * sign / 0.027))
        ),
    )
    shape(brow, "Curious", lambda p, sign=sign: p + Vector((0, 0, 0.008 if sign == 1 else -0.001)))


# Lips deform continuously. Speech uses JawOpen independently of emotional corners.
def mouth_transform(p, kind):
    p = p.copy()
    across = min(1, abs(p.x) / 0.024)
    if kind == "JawOpen":
        p.z = 1.535 + (p.z - 1.535) * 5 - 0.002 * (1 - across)
    elif kind == "Happy":
        p.x *= 1.10
        p.z += 0.006 * across**1.5
    elif kind == "Concerned":
        p.z -= 0.0035 * across
    elif kind == "Surprised":
        p.x *= 0.66
        p.z = 1.535 + (p.z - 1.535) * 3
    elif kind == "MouthRound":
        p.x *= 0.62
        p.y -= 0.003
    return p


inside = ellipsoid(
    "Mouth cavity", (0, -0.0855, 1.5345), (0.023, 0.001, 0.0023), mouth_dark, "Head", 32, 12
)
mouth_parts = [inside]
for upper in (True, False):
    points = []
    for i in range(33):
        u = i / 32 * 2 - 1
        z = (0.0012 + 0.0012 * math.exp(-(((abs(u) - 0.25) / 0.2) ** 2))) if upper else -0.0024
        points.append((u * 0.023, -0.087 + 0.002 * u * u, 1.535 + z * math.sin(PI * i / 32)))
    mouth_parts.append(
        tube(
            "Upper lip" if upper else "Lower lip",
            points,
            [0.0003 + 0.0012 * math.sin(PI * i / 32) for i in range(33)],
            lip,
            "Head",
            8,
        )
    )
for obj in mouth_parts:
    for name in ("JawOpen", "Happy", "Concerned", "Surprised", "MouthRound"):
        shape(obj, name, lambda p, name=name: mouth_transform(p, name))

# Layered, curved hair ribbons with fine secondary strands, rather than a solid helmet.
cap_sections = [
    (1.615, 0, 0.008, 0.096, 0.094),
    (1.66, 0, 0.008, 0.096, 0.094),
    (1.70, 0, 0.008, 0.088, 0.090),
    (1.735, 0, 0.008, 0.059, 0.064),
    (1.755, 0, 0.008, 0.006, 0.008),
]
rings("Hair crown", cap_sections, hair, fixed("Head"), 64, open_front=0.97)
rings(
    "Closed crown",
    [
        (1.68, 0, 0.008, 0.093, 0.090),
        (1.71, 0, 0.008, 0.082, 0.083),
        (1.738, 0, 0.008, 0.053, 0.060),
        (1.755, 0, 0.008, 0.002, 0.003),
    ],
    hair,
    fixed("Head"),
    64,
)
for j in range(19):
    angle = 0.8 + (2 * PI - 1.6) * j / 18
    x, y = 0.086 * math.sin(angle), -0.086 * math.cos(angle) + 0.012
    endx = 0.115 * math.sin(angle)
    lock(
        "Long hair lock",
        [
            (x * 0.42, y * 0.42, 1.742),
            (x * 1.10, y * 1.10, 1.64),
            (endx * 1.45, y + 0.025, 1.36),
            (endx + 0.014 * math.sin(j * 2), y + 0.016, 1.03 + 0.065 * math.sin(j * 1.7)),
        ],
        0.024,
        hair if j % 3 else hair_light,
        "Hair.L" if x > 0 else "Hair.R",
    )
for sign in (-1, 1):
    for i in range(5):
        lock(
            "Face framing lock",
            [
                (sign * (0.054 + i * 0.007), -0.019, 1.716 - i * 0.007),
                (sign * (0.108 + i * 0.004), -0.055, 1.59),
                (sign * (0.07 + i * 0.018), -0.094, 1.37),
                (sign * (0.10 + i * 0.014), -0.087, 1.235 + i * 0.018),
            ],
            0.011,
            hair if i % 2 else hair_light,
            "Hair.L" if sign > 0 else "Hair.R",
        )
for i in range(8):
    t = i / 7
    lock(
        "Swept fringe",
        [
            (0.038 - t * 0.012, -0.032, 1.747 - t * 0.006),
            (0.01 - t * 0.05, -0.104, 1.728 - t * 0.015),
            (-0.004 - t * 0.067, -0.112, 1.639 - t * 0.014),
            (-0.026 - t * 0.060, -0.095 + t * 0.021, 1.617 - t * 0.022),
        ],
        0.020,
        hair if i % 3 else hair_light,
    )
for i in range(5):
    t = i / 4
    lock(
        "Right fringe",
        [
            (0.03 + t * 0.008, -0.011, 1.748 - t * 0.006),
            (0.08 + t * 0.012, -0.051, 1.708),
            (0.055 + t * 0.025, -0.097, 1.653),
            (0.055 + t * 0.032, -0.078 + t * 0.027, 1.617 - t * 0.041),
        ],
        0.018,
        hair if i % 2 else hair_light,
    )

# Round garment cross-sections before skinning, while retaining the authored control cage in code.
for obj in meshes:
    if obj.name.startswith(
        (
            "Knit torso",
            "Open ivory",
            "Cardigan sleeve",
            "Trouser leg",
            "Trouser hips",
            "Hair crown",
            "Closed crown",
        )
    ):
        bpy.context.view_layer.objects.active = obj
        modifier = obj.modifiers.new("Soft garment surface", "SUBSURF")
        modifier.levels = 1
        bpy.ops.object.modifier_apply(modifier=modifier.name)

# Merge compatible static parts to keep runtime draw calls bounded; preserve all shape keys.
groups = {}
for obj in meshes:
    if obj.data.shape_keys is None:
        bone_names = tuple(sorted(g.name for g in obj.vertex_groups))
        groups.setdefault((obj.data.materials[0].name, bone_names), []).append(obj)
for objects in groups.values():
    if len(objects) < 2:
        continue
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
meshes = [obj for obj in scene.objects if obj.type == "MESH"]
for obj in meshes:
    modifier = obj.modifiers.new("Ari skeleton", "ARMATURE")
    modifier.object = rig
    obj.parent = rig

# A short, editable demonstration action; runtime uses the same bones and morph names.
for b in rig.pose.bones:
    b.rotation_mode = "XYZ"
for frame, yaw, tilt in [
    (1, 0, 0),
    (90, -0.24, 0.025),
    (180, 0.24, -0.035),
    (270, 0, 0),
    (420, 0, 0.025),
    (600, 0, 0),
]:
    head = rig.pose.bones["Head"]
    head.rotation_euler = (0.015, yaw, tilt)
    head.keyframe_insert("rotation_euler", frame=frame)
for frame, shoulder, elbow in [
    (1, 0, 0),
    (270, 0, 0),
    (310, 0.9, 1.6),
    (350, 0.96, 1.72),
    (390, 0.9, 1.55),
    (440, 0, 0),
    (600, 0, 0),
]:
    rig.pose.bones["UpperArm.R"].rotation_euler.z = shoulder
    rig.pose.bones["UpperArm.R"].keyframe_insert("rotation_euler", frame=frame)
    rig.pose.bones["LowerArm.R"].rotation_euler.x = -0.18 if elbow else 0
    rig.pose.bones["LowerArm.R"].rotation_euler.z = elbow
    rig.pose.bones["LowerArm.R"].keyframe_insert("rotation_euler", frame=frame)
if rig.animation_data:
    rig.animation_data.action.name = "Ari_LookAndGreet_20s"
scene.frame_set(1)

# Reference planes live only in the editable Blender document, never the GLB.
reference = bpy.data.objects.new("REFERENCE — approved Ari full body", None)
reference.empty_display_type = "IMAGE"
reference.data = bpy.data.images.load(str(ROOT / "characters/ari/design/full-body-neutral-v1.png"))
reference.empty_display_size = 1.8
reference.location = (1.2, 0.12, 0.90)
reference.rotation_euler.x = PI / 2
reference.hide_render = True
scene.collection.objects.link(reference)


def point_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


bpy.ops.object.camera_add(location=(0, -3.8, 1.0))
camera = bpy.context.object
camera.name = "Portrait and full-body camera"
camera.data.type = "ORTHO"
camera.data.ortho_scale = 1.98
point_at(camera, (0, 0, 0.9))
scene.camera = camera
for name, location, energy, size, color in [
    ("Soft key", (-2, -3, 4), 380, 4, (1, 0.88, 0.77)),
    ("Cool fill", (2, -1, 2), 160, 3, (0.77, 0.86, 1)),
    ("Hair rim", (1, 2, 3), 340, 3, (1, 0.90, 0.79)),
]:
    bpy.ops.object.light_add(type="AREA", location=location)
    lamp = bpy.context.object
    lamp.name = name
    lamp.data.energy, lamp.data.shape, lamp.data.size, lamp.data.color = energy, "DISK", size, color
    point_at(lamp, (0, 0, 1.0))

bpy.ops.object.select_all(action="DESELECT")
rig.select_set(True)
for obj in meshes:
    obj.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.export_scene.gltf(
    filepath=str(OUT / "ari-prototype.glb"),
    export_format="GLB",
    use_selection=True,
    export_animations=True,
    export_morph=True,
    export_skins=True,
    export_yup=True,
)
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "ari-prototype.blend"))
triangles = sum(sum(len(p.vertices) - 2 for p in obj.data.polygons) for obj in meshes)
manifest = {
    "status": "prototype — likeness review required",
    "source": "scripts/build_ari_model.py",
    "height_m": 1.8,
    "triangles": triangles,
    "mesh_objects": len(meshes),
    "bones": len(rig.data.bones),
    "shape_keys": sorted(
        {
            k.name
            for obj in meshes
            if obj.data.shape_keys
            for k in obj.data.shape_keys.key_blocks
            if k.name != "Basis"
        }
    ),
    "reference": "../design/full-body-neutral-v1.png",
    "animations": ["Ari_LookAndGreet_20s"],
    "license": "Original geometry and rig: project MIT license. Reference artwork generated with built-in image_gen.",
}
(OUT / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
scene.render.filepath = str(OUT / "full-body-preview.png")
bpy.ops.render.render(write_still=True)
camera.location = (0, -3.8, 1.55)
point_at(camera, (0, 0, 1.55))
camera.data.ortho_scale = 0.53
scene.render.resolution_x = 900
scene.render.resolution_y = 900
scene.render.filepath = str(OUT / "portrait-preview.png")
bpy.ops.render.render(write_still=True)
print("ARI_MODEL_REPORT", json.dumps(manifest))
