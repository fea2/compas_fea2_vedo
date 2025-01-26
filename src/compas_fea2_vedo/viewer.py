from vedo import Plotter, Points, Mesh, TetMesh, Cone, Glyph, Arrows
from compas_fea2.model import Model, DeformablePart
from compas_fea2.problem import Step
from compas_fea2.results.fields import FieldResults
from itertools import chain
import numpy as np
from typing import List, Any, Optional


class FEA2Viewer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.plotter = Plotter(shape=(2, 2), title="Model Viewer with Toggle Menu", axes=8, bg="white", size=(1200, 800))

        # Visualization parameters
        self.point_color = "red"
        self.point_size = 5
        self.mesh_color = "lightblue"
        self.mesh_alpha = 0.8
        self.cmap = "jet"

    def add_objects(self, *args: Any, **kwargs: Any) -> None:
        """Add actors to the plotter."""
        self.plotter.add(args + tuple(kwargs.values()))

    def add_cmap_to_mesh(self, mesh: Mesh, values: Optional[List[float]] = None, title: str = "title", cmap: str = "jet", on: str = "points") -> Mesh:
        """Add color map to the mesh.

        Parameters
        ----------
        mesh : Mesh
            The mesh to which the color map will be added.
        values : Optional[List[float]]
            The values for the color map.
        title : str
            The title of the scalar bar.
        cmap : str
            The color map to use.
        on : str
            Apply the color map on points or cells.

        Returns
        -------
        Mesh
            The mesh with the color map applied.
        """
        if values:
            mesh.cmap(cmap, values, on=on)
            mesh.add_scalarbar(title=title)
        return mesh

    def add_isolines_to_mesh(self, mesh: Mesh, n: int = 10) -> Mesh:
        """Add isolines to the mesh.

        Parameters
        ----------
        mesh : Mesh
            The mesh to which the isolines will be added.
        n : int
            The number of isolines.

        Returns
        -------
        Mesh
            The mesh with isolines added.
        """
        surface = mesh.tomesh()
        isolines = surface.isolines(n=n)
        isolines.c("black").lw(4)
        return isolines

    def show(self) -> None:
        """Show the visualization with toggle buttons."""
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

    @property
    def elements(self) -> List[Any]:
        """Get elements from the model."""
        return list(chain.from_iterable([part.elements for part in self.parts]))

    def add_part(self, part: DeformablePart) -> None:
        """Add parts to the plotter.

        Parameters
        ----------
        part : DeformablePart
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

    def add_node_field_results(self, field: FieldResults, draw_vectors: bool, draw_cmap: bool, draw_isolines: int) -> None:
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
        """
        for part in self.parts:
            part.add_node_field_results(field, draw_vectors, draw_cmap, draw_isolines)

    def add_stess_field_results(self, field: FieldResults, draw_vectors: bool, draw_cmap: bool, draw_isolines: int) -> None:
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

    def show(self, show_bcs: bool = True, show_parts: bool = True) -> None:
        """Show the visualization with toggle buttons.

        Parameters
        ----------
        show_bcs : bool
            Whether to show boundary conditions.
        show_parts : bool
            Whether to show parts.
        """
        if show_bcs:
            self.add_bcs()
        if show_parts:
            for part in self.parts:
                self.plotter.add(part.elements)
                self.plotter.add(part.isolines)
                self.plotter.add(part.deformed)
                self.plotter.add(part.field_vectors)
                for i, shape in enumerate(part.shapes, start=1):
                    self.plotter.at(i).add(shape)
        self.plotter.show(viewup="z", interactive=True)


class PartViewer(FEA2Viewer):
    def __init__(self, part: DeformablePart, *args: Any, **kwargs: Any) -> None:
        """
        Visualize a `DeformablePart` of the FEA model with toggle buttons.

        Parameters
        ----------
        part : DeformablePart
            The `DeformablePart` object to visualize.
        """
        super().__init__(*args, **kwargs)
        self.part = part
        self._vertices = [node.xyz for node in sorted(self.part.nodes, key=lambda n: n.key)]
        self._points = Points(self.vertices, r=self.point_size, c=self.point_color).legend("Nodes")
        self._elements_faces = [face.nodes_key for element in self.part.elements for face in element.faces]
        self._elements_connectivity = [element.nodes_key for element in self.part.elements]
        self._elements = TetMesh([self.vertices, self.elements_connectivity])
        self._isolines = None
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
    def elements_faces(self) -> List[List[int]]:
        """Get faces from elements."""
        return self._elements_faces

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

    def add_node_field_results(self, field: Any, draw_vectors: float, draw_cmap: str, draw_isolines: int) -> None:
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
        """
        locations = []
        vectors = []
        scalars = []
        for loc in sorted(self.part.nodes, key=lambda n: n.key):
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

        if draw_cmap:
            mesh = self.add_cmap_to_mesh(self.elements, values=scalars, title=field.field_name, cmap=draw_cmap)

            if draw_isolines:
                self._isolines = self.add_isolines_to_mesh(mesh, n=draw_isolines)

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
        for node in sorted(self.part.nodes, key=lambda n: n.key):
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
