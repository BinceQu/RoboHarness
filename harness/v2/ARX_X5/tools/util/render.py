"""Static URDF mesh rendering on Mesa's CPU rasterizer; no simulator or scene."""

from pathlib import Path
import os
import math
import xml.etree.ElementTree as ET
import ctypes
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

from .model import URDF, ACTION_NAMES, LOCKED_INDICES, forward, joints, origin, numbers

ASSETS = Path(__file__).resolve().parents[2] / "robot_config"


def font(size=20, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(ASSETS / "fonts" / name), size)


def camera_basis(direction, up=(0, 0, 1)):
    z = np.asarray(direction, dtype=float)
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up, dtype=float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.array([x, y, z])


def software_context():
    """Select Mesa's software device explicitly, before initializing EGL."""
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"
    # A 1320x650 body illustration does not need a host-sized worker pool.
    os.environ.setdefault("LP_NUM_THREADS", "2")
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = (
        "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
    )
    egl = ctypes.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    query_devices = ctypes.CFUNCTYPE(
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
    )(egl.eglGetProcAddress(b"eglQueryDevicesEXT"))
    query_string = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)(
        egl.eglGetProcAddress(b"eglQueryDeviceStringEXT")
    )
    devices, count = (ctypes.c_void_p * 64)(), ctypes.c_int()
    if not query_devices(64, devices, ctypes.byref(count)):
        raise RuntimeError("Could not enumerate EGL devices")
    for index in range(count.value):
        extensions = query_string(devices[index], 0x3055) or b""
        if b"EGL_MESA_device_software" in extensions:
            import moderngl

            # glcontext otherwise chooses hardware device 0, even when
            # LIBGL_ALWAYS_SOFTWARE is set. Never initialize that device.
            os.environ["GLCONTEXT_DEVICE_INDEX"] = str(index)
            return moderngl.create_standalone_context(backend="egl", device_index=index)
    raise RuntimeError("Mesa EGL software device is unavailable")


class RobotRenderer:
    def __init__(self, mesh_root, urdf=URDF):
        # Setting this before importing ModernGL is intentional: this renderer
        # must never allocate a physical GPU, including on evaluator hosts.
        import moderngl

        self.gl = moderngl
        self.ctx = software_context()
        self._resources = []
        self.meshes = []
        try:
            self._initialize(mesh_root, urdf)
        except BaseException:
            self.close()
            raise

    def _own(self, resource):
        self._resources.append(resource)
        return resource

    def _initialize(self, mesh_root, urdf):
        self.renderer = self.ctx.info["GL_RENDERER"]
        if not any(x in self.renderer.lower() for x in ("llvmpipe", "softpipe")):
            raise RuntimeError(f"A software renderer is required; got {self.renderer}")
        self.urdf = urdf
        self.program = self._own(self.ctx.program(
            vertex_shader="""
            #version 330
            in vec3 pos; in vec3 normal; in vec3 color;
            uniform mat4 transform; uniform mat3 rotate;
            out vec3 n; out vec3 c;
            void main() { gl_Position = transform * vec4(pos, 1);
                          n = rotate * normal; c = color; }
        """,
            fragment_shader="""
            #version 330
            in vec3 n; in vec3 c; out vec4 frag; uniform vec3 tint;
            void main() {
                vec3 norm = normalize(n);
                float diffuse = max(dot(norm, normalize(vec3(-0.5, 0.7, 1.))), 0.);
                float fill = max(dot(norm, normalize(vec3(0.8, -0.2, 0.6))), 0.);
                frag = vec4(tint * (0.60 + 0.32 * diffuse + 0.08 * fill) * (0.9 + 0.1 * c), 1.);
            }
        """,
        ))
        mesh_root = Path(mesh_root)
        for link in ET.parse(urdf).getroot().findall("link"):
            for visual in link.findall("visual"):
                mesh = visual.find("geometry/mesh")
                if mesh is None:
                    continue
                loaded = trimesh.load(
                    mesh_root / mesh.get("filename"), force="mesh", process=False
                )
                pose_node = visual.find("origin")
                transform = (
                    origin(
                        numbers(pose_node.get("xyz", "0 0 0")),
                        numbers(pose_node.get("rpy", "0 0 0")),
                    )
                    if pose_node is not None
                    else np.eye(4)
                )
                xyz = np.asarray(loaded.vertices) * numbers(mesh.get("scale", "1 1 1"))
                xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
                normals = np.asarray(loaded.vertex_normals) @ transform[:3, :3].T
                visual_colors = (
                    loaded.visual.to_color()
                    if hasattr(loaded.visual, "to_color")
                    else loaded.visual
                )
                colors = np.asarray(visual_colors.vertex_colors)
                if colors.ndim != 2 or len(colors) != len(xyz):
                    colors = np.full((len(xyz), 3), 220.0, dtype=float)
                colors = colors[:, :3] / 255.0
                # UV-atlas colors come from the public robot asset, not labels
                # invented to suggest a physical joint or material.
                buffer = self._own(self.ctx.buffer(
                    np.column_stack((xyz, normals, colors)).astype("f4").tobytes()
                ))
                indices = self._own(self.ctx.buffer(
                    np.asarray(loaded.faces, dtype="i4").tobytes()
                ))
                vao = self._own(self.ctx.vertex_array(
                    self.program,
                    [(buffer, "3f 3f 3f", "pos", "normal", "color")],
                    indices,
                ))
                self.meshes.append((link.get("name"), xyz, vao))

    def close(self):
        if self.ctx is None:
            return
        try:
            # ModernGL's default gc_mode=None does not release GL objects.
            # Delete them on their owning thread while its context is current.
            while self._resources:
                self._resources.pop().release()
        finally:
            self.meshes.clear()
            self.program = None
            try:
                # glcontext.release() only calls eglDestroyContext. A current
                # EGL context remains alive until explicitly unbound, including
                # when this short-lived request thread exits.
                self.ctx.__exit__(None, None, None)
            finally:
                self.ctx.release()
                self.ctx = None

    def scene(self, positions, arm=None):
        frames, axes = forward(positions, self.urdf)
        selected = [
            (link, xyz, vao)
            for link, xyz, vao in self.meshes
            if arm is None or link.startswith(arm + "_")
        ]
        return frames, axes, selected

    def fit(self, scenes, basis, aspect):
        cloud = (
            np.concatenate(
                [
                    xyz @ frames[link][:3, :3].T + frames[link][:3, 3]
                    for frames, _, meshes in scenes
                    for link, xyz, _ in meshes
                ]
            )
            @ basis.T
        )
        low, high = cloud.min(axis=0), cloud.max(axis=0)
        center = (low + high) / 2
        half_y = max((high[1] - low[1]) / 2, (high[0] - low[0]) / 2 / aspect) * 1.20
        half_z = max(1.0, (high[2] - low[2]) * 2)
        matrix = np.eye(4)
        matrix[:3, :3] = (
            np.diag([1 / (half_y * aspect), 1 / half_y, -1 / half_z]) @ basis
        )
        matrix[:3, 3] = -center * np.array(
            [1 / (half_y * aspect), 1 / half_y, -1 / half_z]
        )
        return matrix

    def render(
        self,
        scene,
        direction=(2, -3, 1.3),
        size=(440, 580),
        fit=None,
        selected=None,
        up=(0, 0, 1),
    ):
        frames, axes, meshes = scene
        basis = camera_basis(direction, up)
        matrix = fit if fit is not None else self.fit([scene], basis, size[0] / size[1])
        fbo = self.ctx.simple_framebuffer(size, components=3)
        try:
            return self._render_framebuffer(
                fbo, frames, axes, meshes, basis, matrix, size, selected
            )
        finally:
            # Framebuffer.release() does not own/delete its attachments.
            attachments = (*fbo.color_attachments, fbo.depth_attachment)
            fbo.release()
            for attachment in attachments:
                if attachment is not None:
                    attachment.release()

    def _render_framebuffer(self, fbo, frames, axes, meshes, basis, matrix, size, selected):
        fbo.use()
        self.ctx.viewport = (0, 0, *size)
        self.ctx.enable(self.gl.DEPTH_TEST)
        fbo.clear(0.976, 0.984, 0.996, 1.0, depth=1.0)
        for link, _, vao in meshes:
            transform = matrix @ frames[link]
            self.program["transform"].write(transform.astype("f4").T.tobytes())
            self.program["rotate"].write(
                (basis @ frames[link][:3, :3]).astype("f4").T.tobytes()
            )
            highlighted = (
                selected is not None and link == joints(self.urdf)[selected]["child"]
            )
            tint = (0.96, 0.39, 0.08) if highlighted else (0.23, 0.36, 0.49)
            if selected is None:
                tint = (
                    (0.30, 0.56, 0.74)
                    if link.startswith("left_")
                    else (0.91, 0.48, 0.20)
                    if link.startswith("right_")
                    else (0.55, 0.61, 0.68)
                )
            self.program["tint"].value = tint
            vao.render()
        image = Image.frombytes(
            "RGB", size, fbo.read(components=3, alignment=1)
        ).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if selected:
            point, axis = axes[selected]
            self.axis(image, matrix, point, axis)
        return image

    @staticmethod
    def axis(image, matrix, point, axis):
        width, height = image.size
        points = np.array([point - axis * 0.055, point + axis * 0.085])
        projected = points @ matrix[:3, :3].T + matrix[:3, 3]
        pixels = np.column_stack(
            ((projected[:, 0] + 1) * width / 2, (1 - projected[:, 1]) * height / 2)
        )
        start, end = pixels
        draw = ImageDraw.Draw(image)
        color = "#009eb9"
        draw.line([tuple(start), tuple(end)], fill=color, width=4)
        delta = end - start
        length = np.linalg.norm(delta)
        if length > 8:
            along = delta / length
            side = np.array([-along[1], along[0]])
            draw.polygon(
                [
                    tuple(end),
                    tuple(end - along * 12 + side * 5),
                    tuple(end - along * 12 - side * 5),
                ],
                fill=color,
            )
        midpoint = (start * 0.085 + end * 0.055) / 0.14
        draw.ellipse(
            (*(midpoint - 5), *(midpoint + 5)), fill="white", outline=color, width=2
        )
        draw.text(tuple(end + [6, -15]), "+axis", font=font(15), fill=color)

    def triptych(self, name):
        joint = joints(self.urdf)[name]
        finger = joint["type"] == "prismatic"
        values = (0.0, 0.022, 0.044) if finger else (-math.pi / 2, 0.0, math.pi / 2)
        arm = name.split("_")[0] if name.startswith(("left_", "right_")) else None
        scenes = [self.scene({name: value}, arm) for value in values]
        direction = (2.4, -3.2 if arm != "left" else 3.2, 1.6)
        size = (440, 570)
        fit = self.fit(scenes, camera_basis(direction), size[0] / size[1])
        canvas = Image.new("RGB", (1320, 735), "white")
        draw = ImageDraw.Draw(canvas)
        index = ACTION_NAMES.index(name)
        number = int(name.rsplit("joint", 1)[1])
        slot = f"arm[{number - 1}]" if number <= 6 else "coupled gripper finger"
        draw.text((22, 12), f"{name} · {slot}", font=font(29, True), fill="#183454")
        draw.text(
            (22, 52),
            "Other joints = 0. Same camera/scale. Orange = selected link; cyan = positive axis. Colors are annotations.",
            font=font(18),
            fill="#51637b",
        )
        for column, (value, scene) in enumerate(zip(values, scenes)):
            x = column * size[0]
            label = (
                f"{value:.3f} m"
                if finger
                else f"{math.degrees(value):+.0f}°  ({value:+.4f} rad)"
            )
            draw.text((x + 18, 86), label, font=font(23, True), fill="#176ab5")
            canvas.paste(self.render(scene, direction, size, fit, name), (x, 126))
            invalid = (
                value < joint["limits"]["lower"] - 1e-8
                or value > joint["limits"]["upper"] + 1e-8
            )
            locked = index in LOCKED_INDICES and abs(value) > 1e-8
            if invalid or locked:
                warning = (
                    "LOCKED: illustration only"
                    if locked
                    else "Outside URDF limit: illustration only"
                )
                draw.text((x + 12, 667), warning, font=font(17, True), fill="#bd292e")
            if column:
                draw.line((x, 80, x, 699), fill="#bacadb", width=2)
        low, high = (joint["limits"][k] for k in ("lower", "upper"))
        note = (
            f"Geometry: {low:g}–{high:g} m; controller opening = (finger_m + 0.01) / 0.054. Both fingers move together."
            if finger
            else f"URDF limits: {math.degrees(low):+.1f}° to {math.degrees(high):+.1f}°. Positive rotation follows the right-hand rule."
        )
        if index in LOCKED_INDICES:
            note += " Task01 locks this joint at 0."
        draw.text((18, 707), note, font=font(16), fill="#51637b")
        return canvas

    def three_views(self, positions, caption="Measured robot pose", sequence=None):
        scene = self.scene(positions)
        views = [
            ("Front · looking along +Y", (0, -1, 0), (0, 0, 1)),
            ("Side · looking along −X", (1, 0, 0), (0, 0, 1)),
            ("Top · looking along −Z", (0, 0, 1), (1, 0, 0)),
        ]
        canvas = Image.new("RGB", (1320, 650), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 12), caption, font=font(26, True), fill="#183454")
        subtitle = "Static model + joint positions only; no simulator scene."
        if sequence is not None:
            subtitle += f" Observation sequence: {sequence}"
        draw.text((20, 50), subtitle, font=font(17), fill="#51637b")
        for column, (label, direction, up) in enumerate(views):
            x = 440 * column
            draw.text((x + 15, 87), label, font=font(18, True), fill="#176ab5")
            canvas.paste(self.render(scene, direction, (440, 520), up=up), (x, 120))
        return canvas


class BodyViewRenderer:
    """Reuse one software context on one owning thread across socket calls."""

    def __init__(self, mesh_root):
        self.mesh_root = Path(mesh_root)
        self._renderer = None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="x5-body-view")

    def save(self, positions, path, *, caption, sequence):
        self._worker.submit(self._save, dict(positions), Path(path), caption, sequence).result()

    def _save(self, positions, path, caption, sequence):
        if self._renderer is None:
            self._renderer = RobotRenderer(self.mesh_root)
        image = self._renderer.three_views(positions, caption=caption, sequence=sequence)
        try:
            image.save(path)
        finally:
            image.close()

    def _close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def close(self):
        worker, self._worker = self._worker, None
        if worker is not None:
            try:
                worker.submit(self._close).result()
            finally:
                worker.shutdown(wait=True)

