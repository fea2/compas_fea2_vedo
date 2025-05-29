from vedo import Plotter, Points, Mesh, TetMesh, Cone, Glyph, Arrows, Grid, Text2D, precision, Sphere, Axes, Polygon
from compas_fea2.model import Model, Part
from compas_fea2.problem import Step

# from compas_fea2.results.fields import FieldResults  # Assuming this is your field results type
import numpy as np
from typing import List, Tuple, Optional, Any

# Type alias for field results
# FieldResultsType = TypeAlias("compas_fea2.results.fields._FieldResults")


class FEA2Viewer:
    def __init__(self, shape: Tuple = (1, 1), *args: Any, **kwargs: Any) -> None:
        self.plotter = Plotter(shape=shape, title="Model Viewer", axes=14, bg="white", size=(1200, 800))

        self.point_color = "red"
        self.point_size = 5
        self.mesh_color = "lightblue"
        self.mesh_alpha = 1
        self.cmap = "jet"
        self.grid = None  # Store the grid actor
        self.camera_position = None

        if kwargs.get("show_grid", False):
            self.add_grid()

    def add_grid(self, size: Tuple[float, float] = (5000, 5000), resolution: Tuple[int, int] = (10, 10), color: str = "gray", alpha: float = 0.3):
        if self.grid is None:
            self.grid = Grid(s=size, res=resolution, lw=0.5)
            self.grid.c(color)
            self.grid.alpha(alpha)
            self.grid.z(0)
            self.plotter.add(self.grid)
        else:
            self.plotter.add(self.grid)

    def remove_grid(self):
        if self.grid is not None:
            self.plotter.remove(self.grid)

    def add_objects(self, *args: Any, **kwargs: Any) -> None:
        self.plotter.add(args + tuple(kwargs.values()))

    def add_cmap_to_mesh(
        self,
        mesh: Mesh,
        values: Optional[List[float]] = None,
        title: str = "title",
        cmap: str = "jet",
        on: str = "points",
    ) -> Mesh:
        if values is not None:
            mesh.cmap(cmap, values, on=on)
            mesh.add_scalarbar(title=title, font_size=24)
        return mesh

    def add_isolines_to_mesh(self, mesh: Mesh, n: int = 10) -> Mesh:
        surface = mesh.tomesh()
        isolines = surface.isolines(n=n)
        isolines.c("black").lw(4)
        return isolines

    def add_isosurfaces_to_mesh(self, mesh: Mesh, n: int, scalars: list) -> Mesh:
        iso_values = np.linspace(min(scalars), max(scalars), n)
        isosurfaces = []
        for value in iso_values:
            isosurfaces.append(mesh.isosurface(value=value))
        return isosurfaces

    def show(self, camera_position: Optional[Tuple] = None) -> None:
        if camera_position:
            self.plotter.camera.SetPosition(*camera_position)  # Set camera position
        self.plotter.show(viewup="z", interactive=True).close()


class ModelViewer(FEA2Viewer):
    def __init__(self, model: Model, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self._parts = [PartViewer(part) for part in self.model.parts]

    @property
    def parts(self) -> List["PartViewer"]:
        """Get parts from the model."""
        return self._parts

    def add_part(self, part: Part) -> None:
        """Add parts to the plotter.

        Parameters
        ----------
        part : Part
            The part to add to the plotter.
        """
        part_viewer = PartViewer(part)
        self.plotter.add(part_viewer.elements)

    def add_bcs(self) -> None:
        """Add boundary conditions to the plotter."""
        cone_height = 50
        cone = Cone(r=cone_height / 2, height=cone_height).scale(1).rotate_y(90)
        for bc, nodes in self.model.bcs.items():
            pts = np.c_[[n.xyz for n in nodes]]
            vecs = np.tile([0, 0, 1], (len(nodes), 1))
            offsets = vecs * (cone_height / 2)
            shifted_pts = pts - offsets
            glyph = Glyph(shifted_pts, cone, vecs, c="red", alpha=0.8)
            glyph.lighting("ambient")
            self.plotter.add(glyph)

    def add_node_field_results(self, field, draw_vectors: float = None, draw_cmap: str = None, draw_isolines: int = None, draw_isosurfaces: int = None) -> None:
        """Add field results to the plotter.

        Parameters
        ----------
        field : Any
            The field results to add.
        draw_vectors : bool
            Whether to draw vectors.
        draw_cmap : bool
            Whether to draw color map.
        draw_isolines : int
            The number of isolines to draw.
        draw_isosurfaces : int
            The number of isosurfaces to draw.
        """
        for part in self.parts:
            part.add_node_field_results(field, draw_vectors, draw_cmap, draw_isolines, draw_isosurfaces)

    def add_stess_field_results(self, field, draw_vectors: bool, draw_cmap: bool, draw_isolines: int) -> None:
        """Add stress field results to the plotter.

        Parameters
        ----------
        field : Any
            The stress field results to add.
        draw_vectors : bool
            Whether to draw vectors.
        draw_cmap : bool
            Whether to draw color map.
        draw_isolines : int
            The number of isolines to draw.
        """
        raise NotImplementedError()

    def add_deformed_shape(self, step: Step, sf: float) -> None:
        """Add deformed shape to the plotter."""
        for part in self.parts:
            part.add_deformed_shape(step, sf)

    def add_mode_shapes(self, shapes: Any, sf: float) -> None:
        """Add mode shapes to the plotter.

        Parameters
        ----------
        shapes : Any
            The mode shapes to add.
        sf : float
            The scale factor for the mode shapes.
        """
        for part in self.parts:
            part.add_mode_shapes(shapes, sf)

    def add_interfaces(self, interfaces, color="yellow", alpha=0.5):
        """Visualize 3D interfaces as polygons in the model.

        Parameters
        ----------
        interfaces : List[Interface]
            A list of detected interface objects.
        color : str, optional
            The color for the interface polygons, by default "yellow".
        alpha : float, optional
            Transparency of the polygons, by default 0.5.
        """
        for interface in interfaces:
            interface = interface[0]
            # Ensure the interface has enough points
            if len(interface.points) < 3:
                continue  # Skip invalid polygons

            # Create a 3D polygon as a vedo Mesh
            poly = Mesh([interface.points, [[i for i in range(len(interface.points))]]])  # Define face connectivity
            poly.c(color).alpha(alpha)  # Set color and transparency

            # Add the polygon to the plotter
            self.plotter.add(poly)

        print(f"Added {len(interfaces)} 3D interface polygons to the visualization.")

    def show(
        self,
        show_bcs: bool = True,
        show_parts: bool = True,
        camera_position: Optional[Tuple] = None,
        section: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
    ) -> None:
        """Show the visualization with toggle buttons.

        Parameters
        ----------
        show_bcs : bool
            Whether to show boundary conditions.
        show_parts : bool
            Whether to show parts.
        section : Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], float]]
            Parameters for creating a section through the model.
        """
        if show_bcs:
            self.add_bcs()

        for part in self.parts:
            self.plotter.add(part._isosurfaces)
            # self.plotter.add(self.cut_mesh(part.elements, None))
        if show_parts:
            for part in self.parts:
                self.plotter.add(part.elements)
                self.plotter.add(part.isolines)
                self.plotter.add(part.deformed)
                self.plotter.add(part.field_vectors)
                for i, shape in enumerate(part.shapes, start=1):
                    self.plotter.at(i).add(shape)
        super().show(camera_position=camera_position)


class PartViewer(FEA2Viewer):
    def __init__(self, part: Part, *args: Any, **kwargs: Any) -> None:
        """
        Visualize a `Part` of the FEA model with toggle buttons.

        Parameters
        ----------
        part : Part
            The `Part` object to visualize.
        """
        super().__init__(*args, **kwargs)
        self.part = part
        self._vertices = part.points_sorted
        self._points = Points(self.vertices, r=self.point_size, c=self.point_color).legend("Nodes")
        self._elements_connectivity = part.elements_connectivity
        self._elements = TetMesh([self.vertices, self.elements_connectivity]).alpha(self.mesh_alpha).c(self.mesh_color)
        self._isolines = None
        self._isosurfaces = None
        self._field_vectors = None
        self._deformed = None
        self._shapes = []

    @property
    def vertices(self) -> List[List[float]]:
        """Get vertices from nodes."""
        return self._vertices

    @property
    def points(self) -> Points:
        """Create points for nodes and store them in actors."""
        return self._points

    @property
    def elements_connectivity(self) -> List[List[int]]:
        """Get connectivity from elements."""
        return self._elements_connectivity

    @property
    def elements(self) -> TetMesh:
        """Create a mesh for 2D elements and store it in actors."""
        return self._elements

    @property
    def isolines(self) -> Optional[Mesh]:
        """Get isolines from mesh."""
        return self._isolines

    @property
    def deformed(self) -> Optional[TetMesh]:
        """Get deformed shape from mesh."""
        return self._deformed

    @property
    def field_vectors(self) -> Optional[Arrows]:
        """Get field vectors from mesh."""
        return self._field_vectors

    @property
    def shapes(self) -> List[TetMesh]:
        """Get mode shapes from mesh."""
        return self._shapes

    # def add_elements(self) -> None:
    #     """Add elements to the plotter.

    #     Parameters
    #     ----------
    #     """
    #     from compas_fea2.model.elements import TetrahedronElement, _Element1D, _Element2D
    #     from vedo import TetMesh, Mesh, Lines  # Replace with your actual visualization classes if different

    #     all_nodes = self.part.nodes_sorted
    #     for cls, elements_group in self.part.elements_connectivity_grouped.items():
    #         connectivity = list(all_nodes.index(n) for e in elements_group for n in e.nodes)
    #         if issubclass(cls, TetrahedronElement):
    #             tet_mesh = TetMesh(self.vertices, connectivity)
    #             self._elements.append(tet_mesh.alpha(self.mesh_alpha).c(self.mesh_color))

    #         elif issubclass(cls, _Element2D):
    #             mesh = Mesh([self.vertices, connectivity])
    #             self._elements.append(mesh.alpha(self.mesh_alpha).c(self.mesh_color))

    #         elif issubclass(cls, _Element1D):
    #             line_el = Lines([self.vertices, connectivity])
    #             self._elements.append(line_el.lw(self.line_width).c(self.line_color))

    #         else:
    #             print(f"Element type '{cls}' not supported for visualization.")

    def add_node_field_results(self, field: Any, draw_vectors: float = None, draw_cmap: str = None, draw_isolines: int = None, draw_isosurfaces: int = None) -> None:
        """Add field results to the plotter.

        Parameters
        ----------
        field : Any
            The field results to add.
        draw_vectors : float
            The scale factor for vectors. If None, vectors are not drawn.
        draw_cmap : str
            The color map to draw. If None, color map is not drawn.
        draw_isolines : int
            The number of isolines to draw. If None, isolines are not drawn.
        draw_isosurfaces : int
            The number of isosurfaces to draw. If None, isosurfaces are not drawn
        """
        locations = []
        vectors = []
        scalars = []
        for loc in sorted(self.part.nodes, key=lambda n: n.part_key):
            vec = field.get_result_at(loc).vector.scaled(draw_vectors)
            loc = loc.xyz
            locations.append(loc)
            vectors.append(loc + vec)
            scalars.append(vec.length)

        if draw_vectors:
            arrows = Arrows(
                start_pts=locations,
                end_pts=vectors,
                c="blue",
                alpha=0.1,
                res=4,
                s=1,
            )
            self._field_vectors = arrows

        if (draw_isolines or draw_isosurfaces) and not draw_cmap:
            draw_cmap = "viridis"

        if draw_cmap:
            mesh = self.add_cmap_to_mesh(self.elements, values=scalars, title=field.field_name, cmap=draw_cmap)

            if draw_isolines:
                self._isolines = self.add_isolines_to_mesh(mesh, n=draw_isolines)
            if draw_isosurfaces:
                self._isosurfaces = self.add_isosurfaces_to_mesh(mesh, n=draw_isosurfaces, scalars=scalars)

    def add_deformed_shape(self, step: Step, sf: float) -> None:
        """Add deformed shape to the plotter.

        Parameters
        ----------
        step : Step
            The step to get the deformed shape from.
        sf : float
            The scale factor for the deformed shape.
        """
        new_pts = []
        for node in sorted(self.part.nodes, key=lambda n: n.part_key):
            vec = node.displacement(step).vector.scaled(sf)
            new_pts.append(node.xyz + vec)
        self._deformed = TetMesh([new_pts, self.elements_connectivity])

    def add_mode_shapes(self, shapes, sf):
        """Add mode shapes to the plotter.

        Parameters
        ----------
        shapes : Any
            The mode shapes to add.
        sf : float
            The scale factor for the mode shapes.
        """
        for shape in shapes:
            locations = []
            vectors = []
            scalars = []

            displacements = list(zip(shape.locations, shape.vectors))
            displacements = sorted(displacements, key=lambda x: x[0].key)

            for disp in displacements:
                vec = disp[1].scaled(sf)
                loc = disp[0].xyz
                locations.append(loc + vec)
                vectors.append(loc + vec)
                scalars.append(vec.length)
            mesh = TetMesh([locations, self.elements_connectivity])
            self._shapes.append(mesh)
            self.add_cmap_to_mesh(mesh=mesh, values=scalars, title=shape.field_name, cmap=self.cmap)
