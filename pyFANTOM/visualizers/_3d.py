import vtk
import numpy as np
import k3d
import matplotlib.pyplot as plt

def vtk_to_k3d(polydata):
    """Convert VTK PolyData to k3d mesh format"""
    vertices = []
    for i in range(polydata.GetNumberOfPoints()):
        vertices.append(polydata.GetPoint(i))
    vertices = np.array(vertices, dtype=np.float64)
    
    indices = []
    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        n_points = cell.GetNumberOfPoints()
        if n_points == 3:
            indices.extend([cell.GetPointId(0), cell.GetPointId(1), cell.GetPointId(2)])
        elif n_points == 4:
            p = [cell.GetPointId(j) for j in range(4)]
            indices.extend([p[0], p[1], p[2], p[0], p[2], p[3]])
    indices = np.array(indices, dtype=np.uint32)
    
    return vertices, indices

def vtk_to_k3d_extract_edges(polydata):
    vertices = []
    for i in range(polydata.GetNumberOfPoints()):
        vertices.append(polydata.GetPoint(i))
    vertices = np.array(vertices, dtype=np.float64)
    
    indices = []
    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        n_points = cell.GetNumberOfPoints()
        if n_points == 3:
            indices.extend([cell.GetPointId(0), cell.GetPointId(1),
                            cell.GetPointId(1), cell.GetPointId(2),
                            cell.GetPointId(2), cell.GetPointId(0)])
        elif n_points == 4:
            p = [cell.GetPointId(j) for j in range(4)]
            indices.extend([p[0], p[1],
                            p[1], p[2],
                            p[2], p[3],
                            p[3], p[0]])
    indices = np.array(indices, dtype=np.uint32)
    
    return vertices, indices

def create_force_glyphs(nodes, forces, max_length=0.05):
    """Create force vectors using VTK glyphs"""
    points = vtk.vtkPoints()
    vectors = vtk.vtkFloatArray()
    vectors.SetNumberOfComponents(3)
    
    force_magnitudes = np.linalg.norm(forces, axis=1)
    max_force = np.max(force_magnitudes)
    
    if max_force > 0:
        force_nodes = np.where(force_magnitudes > 1e-10)[0]
        
        for idx in force_nodes:
            points.InsertNextPoint(nodes[idx])
            force = forces[idx]
            magnitude = force_magnitudes[idx]
            direction = force / magnitude
            scaled_vector = direction * (max_length * magnitude / max_force)
            vectors.InsertNextTuple(scaled_vector)
    
    point_data = vtk.vtkPolyData()
    point_data.SetPoints(points)
    point_data.GetPointData().SetVectors(vectors)
    
    arrow = vtk.vtkArrowSource()
    arrow.SetShaftRadius(0.03)
    arrow.SetTipRadius(0.05)
    arrow.SetTipLength(0.3)
    
    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(arrow.GetOutputPort())
    glyph.SetInputData(point_data)
    glyph.SetVectorModeToUseVector()
    glyph.SetScaleModeToScaleByVector()
    glyph.SetScaleFactor(1.0)
    glyph.OrientOn()
    glyph.Update()
    
    return glyph.GetOutput()

def create_bc_glyphs(nodes, constraints, size=0.01):
    """Create boundary condition visualization using VTK glyphs"""
    points_x = vtk.vtkPoints()
    points_x_free = vtk.vtkPoints()
    points_y = vtk.vtkPoints()
    points_y_free = vtk.vtkPoints()
    points_z = vtk.vtkPoints()
    points_z_free = vtk.vtkPoints()
    
    # Find nodes with constraints in each direction
    constrained_nodes_x = np.where(constraints[:, 0])[0]
    constrained_nodes_y = np.where(constraints[:, 1])[0]
    constrained_nodes_z = np.where(constraints[:, 2])[0]
    
    all_constrained_nodes = np.where(np.any(constraints, axis=1))[0]
    
    for node in all_constrained_nodes:
        if constraints[node, 0]:
            points_x.InsertNextPoint(nodes[node])
        else:
            points_x_free.InsertNextPoint(nodes[node])
        if constraints[node, 1]:
            points_y.InsertNextPoint(nodes[node])
        else:
            points_y_free.InsertNextPoint(nodes[node])
        if constraints[node, 2]:
            points_z.InsertNextPoint(nodes[node])
        else:
            points_z_free.InsertNextPoint(nodes[node])
    
    # Add points for each direction
    # for node_idx in constrained_nodes_x:
    #     points_x.InsertNextPoint(nodes[node_idx])
    # for node_idx in constrained_nodes_y:
    #     points_y.InsertNextPoint(nodes[node_idx])
    # for node_idx in constrained_nodes_z:
    #     points_z.InsertNextPoint(nodes[node_idx])
    
    def create_direction_arrow(direction, add_cones=False):
        # Create base cylinder
        cylinder = vtk.vtkCylinderSource()
        cylinder.SetHeight(0.8 * size)
        cylinder.SetRadius(0.05 * size)
        cylinder.SetResolution(8)
        
        # Create cones for both ends
        cone1 = vtk.vtkConeSource()
        cone1.SetHeight(0.2 * size)
        cone1.SetRadius(0.1 * size)
        cone1.SetResolution(8)
        
        cone2 = vtk.vtkConeSource()
        cone2.SetHeight(0.2 * size)
        cone2.SetRadius(0.1 * size)
        cone2.SetResolution(8)
        
        if direction == 'x':
            # X direction
            transform = vtk.vtkTransform()
            transform.RotateZ(90)
            
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetInputConnection(cylinder.GetOutputPort())
            transformFilter.SetTransform(transform)
            cylinder_output = transformFilter
            
            if add_cones:
                cone1.SetDirection(1, 0, 0)
                cone2.SetDirection(-1, 0, 0)
                cone1.SetCenter(0.5 * size, 0, 0)
                cone2.SetCenter(-0.5 * size, 0, 0)
            
        elif direction == 'y':
            # Y direction - default orientation
            cylinder_output = cylinder
            if add_cones:
                cone1.SetDirection(0, 1, 0)
                cone2.SetDirection(0, -1, 0)
                cone1.SetCenter(0, 0.5 * size, 0)
                cone2.SetCenter(0, -0.5 * size, 0)
            
        else:  # z direction
            # Z direction
            transform = vtk.vtkTransform()
            transform.RotateX(90)
            
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetInputConnection(cylinder.GetOutputPort())
            transformFilter.SetTransform(transform)
            cylinder_output = transformFilter
            if add_cones:
                cone1.SetDirection(0, 0, 1)
                cone2.SetDirection(0, 0, -1)
                cone1.SetCenter(0, 0, 0.5 * size)
                cone2.SetCenter(0, 0, -0.5 * size)
        
        # Combine parts
        append = vtk.vtkAppendPolyData()
        append.AddInputConnection(cylinder_output.GetOutputPort())
        if add_cones:
            append.AddInputConnection(cone1.GetOutputPort())
            append.AddInputConnection(cone2.GetOutputPort())
        append.Update()
        
        return append.GetOutput()
    
    def create_direction_glyph(points, direction, add_cones=False):
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        arrow = create_direction_arrow(direction, add_cones)
        
        glyph = vtk.vtkGlyph3D()
        glyph.SetSourceData(arrow)
        glyph.SetInputData(polydata)
        glyph.ScalingOff()
        glyph.Update()
        
        return glyph.GetOutput()
    
    # Create and combine glyphs for all directions
    append = vtk.vtkAppendPolyData()
    
    if points_x.GetNumberOfPoints() > 0:
        x_glyphs = create_direction_glyph(points_x, 'x', add_cones=False)
        append.AddInputData(x_glyphs)
    
    if points_x_free.GetNumberOfPoints() > 0:
        x_glyphs_free = create_direction_glyph(points_x_free, 'x', add_cones=True)
        append.AddInputData(x_glyphs_free)
    
    if points_y.GetNumberOfPoints() > 0:
        y_glyphs = create_direction_glyph(points_y, 'y', add_cones=False)
        append.AddInputData(y_glyphs)
    
    if points_y_free.GetNumberOfPoints() > 0:
        y_glyphs_free = create_direction_glyph(points_y_free, 'y', add_cones=True)
        append.AddInputData(y_glyphs_free)
    
    if points_z.GetNumberOfPoints() > 0:
        z_glyphs = create_direction_glyph(points_z, 'z', add_cones=False)
        append.AddInputData(z_glyphs)
    
    if points_z_free.GetNumberOfPoints() > 0:
        z_glyphs_free = create_direction_glyph(points_z_free, 'z', add_cones=True)
        append.AddInputData(z_glyphs_free)
    
    append.Update()
    return append.GetOutput()

def plot_problem_3D(nodes: np.ndarray,
                    elements: np.ndarray,
                    f: np.ndarray,
                    c: np.ndarray,
                    rho=None,
                    face_color=0x888888,
                    force_color=0xff8300,
                    constraint_color=0xff6347):
    """Plot FEA problem with forces and boundary conditions"""
    # Extract mesh data
    
    approx_elem_size = np.mean(np.linalg.norm(nodes[elements[:, 1]] - nodes[elements[:, 0]], axis=1))
    
    if rho is not None:
        if rho.ndim > 1:
            elements_ = []
            for i in range(rho.shape[1]):
                elements_.append(elements[rho[:,i]>0.5])
            elements = elements_
        else:
            elements = elements[rho>0.5]
    else:
        rho = np.empty([elements.shape[0]])
    
    # Create base mesh geometry
    points = vtk.vtkPoints()
    for node in nodes:
        points.InsertNextPoint(node)
    
    # Create k3d plot
    plot = k3d.plot()
    
    if rho.ndim > 1:
        if isinstance(face_color, list):
            fc = face_color
        else:
            # If face_color is not a list, then in multi-material case we use viridis colormap
            fc = plt.cm.viridis(np.linspace(0, 1, rho.shape[1]))
            
            # turn rgb to hex string
            fc = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b, a in fc]
            
            # convert to k3d color format
            fc = [int(color[1:], 16) for color in fc]
        
        geoms = []    
        
        for j in range(rho.shape[1]):
            ugrid = vtk.vtkUnstructuredGrid()
            ugrid.SetPoints(points)
            
            # Add elements to grid
            element_size = elements[j].shape[1]
            for element in elements[j]:
                if element_size == 4:
                    cell = vtk.vtkTetra()
                elif element_size == 8:
                    cell = vtk.vtkHexahedron()
                else:
                    raise ValueError("Unsupported element type")
                    
                for i in range(element_size):
                    cell.GetPointIds().SetId(i, element[i])
                ugrid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())

            # Extract surface
            surface_filter = vtk.vtkDataSetSurfaceFilter()
            surface_filter.SetInputData(ugrid)
            surface_filter.Update()
            
            geom = k3d.factory.vtk_poly_data(surface_filter.GetOutput(), color=fc[j])
            
            geoms.append(geom)
            # Add main mesh
            plot += geom
    else:
        ugrid = vtk.vtkUnstructuredGrid()
        ugrid.SetPoints(points)
        
        # Add elements to grid
        element_size = elements.shape[1]
        for element in elements:
            if element_size == 4:
                cell = vtk.vtkTetra()
            elif element_size == 8:
                cell = vtk.vtkHexahedron()
            else:
                raise ValueError("Unsupported element type")
                
            for i in range(element_size):
                cell.GetPointIds().SetId(i, element[i])
            ugrid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())

        # Extract surface
        surface_filter = vtk.vtkDataSetSurfaceFilter()
        surface_filter.SetInputData(ugrid)
        surface_filter.Update()
        
        # Add main mesh
        plot += k3d.factory.vtk_poly_data(surface_filter.GetOutput(), color=face_color)
    # Add force vectors
    if f is not None:
        force_glyphs = create_force_glyphs(nodes, f, approx_elem_size*4)
        vertices, indices = vtk_to_k3d(force_glyphs)
        plot += k3d.mesh(vertices, indices, color=force_color)
    
    # Add boundary conditions
    if c is not None:
        bc_glyphs = create_bc_glyphs(nodes, c, approx_elem_size)
        vertices, indices = vtk_to_k3d(bc_glyphs)
        plot += k3d.mesh(vertices, indices, color=constraint_color)
    
    plot.grid_visible = False
    
    from ipywidgets import FloatSlider, interact, Dropdown, Text, Checkbox, GridBox, Layout
    
    if rho.ndim > 1:
        chks = {}
        for i in range(rho.shape[1]):
            chks[f"Material {i+1}"] = Checkbox(value=True, description=f"<b style='color:#{fc[i]:06x}'>Material {i+1}</b>", indent=False)
        
        layout = Layout(grid_template_columns="repeat(4, 1fr)")
        GridBox(children=list(chks.values()), layout=layout)
        
        @interact(**chks)
        def _(**kwargs):
            for i, geom in enumerate(geoms):
                geom.visible = kwargs[f"Material {i+1}"]
    
    return plot

def plot_mesh_3D(nodes: np.ndarray, elements: np.ndarray, rho=None, face_color=0x888888):
    """Plot FEA problem with forces and boundary conditions"""
    # Extract mesh data

    if rho is not None:
        elements = elements[rho > 0.5]
    
    # Create base mesh geometry
    points = vtk.vtkPoints()
    for node in nodes:
        points.InsertNextPoint(node)
    
    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(points)
    
    # Add elements to grid
    element_size = elements.shape[1]
    for element in elements:
        if element_size == 4:
            cell = vtk.vtkTetra()
        elif element_size == 8:
            cell = vtk.vtkHexahedron()
        else:
            raise ValueError("Unsupported element type")
            
        for i in range(element_size):
            cell.GetPointIds().SetId(i, element[i])
        ugrid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())

    # Extract surface
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(ugrid)
    surface_filter.Update()
    
    # Create k3d plot
    plot = k3d.plot()
    
    # Add main mesh
    plot += k3d.factory.vtk_poly_data(surface_filter.GetOutput(), color=face_color)
    plot.grid_visible = False
    return plot

def plot_field_3D(
    nodes: np.ndarray,
    elements: np.ndarray,
    field,
    rho=None,
    colormap='viridis',
    min_value=None,
    max_value=None,
    **kwargs
):
    """Plot field values on a 3D mesh surface using VTK and k3d
    """
    if rho is not None:
        elements = elements[rho > 0.5]
        field = field[rho > 0.5]
    
    # Create base mesh geometry
    points = vtk.vtkPoints()
    for node in nodes:
        points.InsertNextPoint(node)
    
    # Create cell array
    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(points)
    
    # Add elements to grid
    element_size = elements.shape[1]
    for i, element in enumerate(elements):
        if element_size == 4:
            cell = vtk.vtkTetra()
        elif element_size == 8:
            cell = vtk.vtkHexahedron()
        else:
            raise ValueError("Unsupported element type")
            
        for j in range(element_size):
            cell.GetPointIds().SetId(j, element[j])
        ugrid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())
    
    # Add field data to cells
    cell_data = vtk.vtkFloatArray()
    cell_data.SetName("FieldData")
    for value in field:
        cell_data.InsertNextValue(value)
    ugrid.GetCellData().SetScalars(cell_data)

    # Setup plane and center
    bbox = np.array(ugrid.GetBounds()).reshape(3, 2)
    center = [0,0,bbox[2].max()]
    
    plane = vtk.vtkPlane()
    plane.SetOrigin(*center)
    plane.SetNormal(0, 0, 1)

    def vtk_ExtractSurface(vtk_grid, vtk_o, vtk_n):
        plane.SetOrigin(*vtk_o)
        plane.SetNormal(*vtk_n)

        myExtractGeometry = vtk.vtkExtractGeometry()
        myExtractGeometry.SetInputData(vtk_grid)
        myExtractGeometry.SetImplicitFunction(plane)
        myExtractGeometry.ExtractInsideOn()
        myExtractGeometry.SetExtractBoundaryCells(0)
        myExtractGeometry.Update()

        myExtractSurface = vtk.vtkDataSetSurfaceFilter()
        myExtractSurface.SetInputConnection(myExtractGeometry.GetOutputPort())
        myExtractSurface.Update()

        return myExtractSurface.GetOutput()

    def update_from_cut(grid, vtk_o, vtk_n, plt_vtk):
        poly_data = vtk_ExtractSurface(grid, vtk_o, vtk_n)

        if poly_data.GetNumberOfCells() > 0:
            vertices, indices = vtk_to_k3d(poly_data)
            surface_data = poly_data.GetCellData().GetScalars()
            attribute = np.array([surface_data.GetValue(i) for i in range(surface_data.GetNumberOfTuples())])
            
            with plt_vtk.hold_sync():
                plt_vtk.vertices = vertices
                plt_vtk.indices = indices
                plt_vtk.attribute = attribute

    # Create k3d plot
    plot = k3d.plot(grid_visible=False)
    
    # Set up color bounds
    if min_value is None:
        min_value = field.min()
    if max_value is None:
        max_value = field.max()

    k3d_colormap = getattr(k3d.colormaps.matplotlib_color_maps, colormap)
    
    # Create initial visualization
    plt_vtk = k3d.vtk_poly_data(
        vtk_ExtractSurface(ugrid, center, [0, 0, 1]),
        color_map=k3d_colormap,
        color_range=[min_value, max_value],
        side='double'
    )
    
    plt_vtk.flat_shading = True
    plot += plt_vtk

    # Create the interactive widget
    from ipywidgets import FloatSlider, interact, Dropdown, Text
    
    offset_slider = FloatSlider(min=bbox[2].min(), max=bbox[2].max(), step=(bbox[2].max()-bbox[2].min())/100,value=bbox[2].max(),description="Offset From Base",continuous_update=True)
    axis_dropdown = Dropdown(options=['X', 'Y', 'Z'], value='Z', description="Cut Along")
    last_axis_inp = Text(value="Z", description="Last Axis", disabled=True)
    last_axis_inp.layout.visibility = 'hidden'
    @interact(offset=offset_slider, axis=axis_dropdown, last_axis=last_axis_inp)
    def _(offset, axis, last_axis):
        if axis == 'X':
            # update offset slider if changed
            if last_axis != 'X':
                offset_slider.min = bbox[0].min()
                offset_slider.max = bbox[0].max()
                offset_slider.value = bbox[0].max()
                offset = bbox[0].max()
            last_axis_inp.value = 'X'
            update_from_cut(ugrid, [offset, center[1], center[2]], [1, 0, 0], plt_vtk)
        elif axis == 'Y':
            # update offset slider
            if last_axis != 'Y':
                offset_slider.min = bbox[1].min()
                offset_slider.max = bbox[1].max()
                offset_slider.value = bbox[1].max()
                offset = bbox[1].max()
            last_axis_inp.value = 'Y'
            update_from_cut(ugrid, [center[0], offset, center[2]], [0, 1, 0], plt_vtk)
        else:
            # update offset slider
            if last_axis != 'Z':
                offset_slider.min = bbox[2].min()
                offset_slider.max = bbox[2].max()
                offset_slider.value = bbox[2].max()
                offset = bbox[2].max()
            last_axis_inp.value = 'Z'
            update_from_cut(ugrid, [center[0], center[1], offset], [0, 0, 1], plt_vtk)
    update_from_cut(ugrid, [center[0], center[1], bbox[2].max()], [0, 0, 1], plt_vtk)
    return plot

import os
import asyncio
import warnings
import logging
from k3d.colormaps import matplotlib_color_maps

def _compute_depth_attribute(vertices, camera_pos, exponent=0.4):
    cam = np.array(camera_pos[:3], dtype=np.float64)
    tgt = np.array(camera_pos[3:6], dtype=np.float64)
    view_dir = tgt - cam
    view_dir /= np.linalg.norm(view_dir) + 1e-12

    depths = np.dot(vertices - cam, view_dir)
    d_min, d_max = depths.min(), depths.max()
    if d_max == d_min:
        return np.zeros_like(depths, dtype=np.float32)

    attr = ((depths - d_min) / (d_max - d_min)) ** exponent
    return attr.astype(np.float32)


async def capture_solution_screenshots_3D(nodes, elements, f, c, rho=None,
                                          output_dir='screenshots', delay=2.0,
                                          face_color=0x888888, force_color=0xff8300,
                                          constraint_color=0xff6347):
    """
    Export K3D visualization to HTML and capture screenshots with Playwright (async),
    with depth-based coloring (near vs far) baked into each view.
    """
    warnings.filterwarnings('ignore', category=UserWarning, module='traittypes')
    logging.getLogger('k3d.helpers').setLevel(logging.ERROR)

    if hasattr(nodes, 'get'):
        nodes = nodes.get()
    if hasattr(elements, 'get'):
        elements = elements.get()
    if f is not None and hasattr(f, 'get'):
        f = f.get()
    if c is not None and hasattr(c, 'get'):
        c = c.get()
    if rho is not None and hasattr(rho, 'get'):
        rho = rho.get()

    os.makedirs(output_dir, exist_ok=True)

    approx_elem_size = np.mean(
        np.linalg.norm(nodes[elements[:, 1]] - nodes[elements[:, 0]], axis=1)
    )

    if rho is not None:
        if rho.ndim > 1:
            elements_ = []
            for i in range(rho.shape[1]):
                elements_.append(elements[rho[:, i] > 0.5])
            elements = elements_
        else:
            elements = elements[rho > 0.5]

    points = vtk.vtkPoints()
    for node in nodes:
        points.InsertNextPoint(node)

    plot = k3d.plot(
        height=1080,
        camera_auto_fit=False,
        grid_visible=False,
        menu_visibility=False
    )

    surface_meshes = []

    if rho is not None and rho.ndim > 1:
        for j in range(rho.shape[1]):
            if elements[j].size == 0:
                continue

            ugrid = vtk.vtkUnstructuredGrid()
            ugrid.SetPoints(points)
            element_size = elements[j].shape[1]

            for element in elements[j]:
                if element_size == 4:
                    cell = vtk.vtkTetra()
                elif element_size == 8:
                    cell = vtk.vtkHexahedron()
                else:
                    raise ValueError("Unsupported element type")
                for i in range(element_size):
                    cell.GetPointIds().SetId(i, int(element[i]))
                ugrid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())

            surface_filter = vtk.vtkDataSetSurfaceFilter()
            surface_filter.SetInputData(ugrid)
            surface_filter.Update()

            surface = surface_filter.GetOutput()
            vertices, indices = vtk_to_k3d(surface)

            if len(vertices) == 0 or len(indices) == 0:
                continue

            mesh = k3d.mesh(
                vertices,
                indices,
                attribute=np.zeros(vertices.shape[0], dtype=np.float32),
                color_map=matplotlib_color_maps.viridis,
                color_range=[0.0, 1.0]
            )
            surface_meshes.append((mesh, vertices))
            plot += mesh
    else:
        ugrid = vtk.vtkUnstructuredGrid()
        ugrid.SetPoints(points)
        element_size = elements.shape[1]

        for element in elements:
            if element_size == 4:
                cell = vtk.vtkTetra()
            elif element_size == 8:
                cell = vtk.vtkHexahedron()
            else:
                raise ValueError("Unsupported element type")
            for i in range(element_size):
                cell.GetPointIds().SetId(i, int(element[i]))
            ugrid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())

        surface_filter = vtk.vtkDataSetSurfaceFilter()
        surface_filter.SetInputData(ugrid)
        surface_filter.Update()

        surface = surface_filter.GetOutput()
        vertices, indices = vtk_to_k3d(surface)

        if len(vertices) > 0 and len(indices) > 0:
            mesh = k3d.mesh(
                vertices,
                indices,
                attribute=np.zeros(vertices.shape[0], dtype=np.float32),
                color_map=matplotlib_color_maps.viridis,
                color_range=[0.0, 1.0]
            )
            surface_meshes.append((mesh, vertices))
            plot += mesh

    if f is not None:
        force_glyphs = create_force_glyphs(nodes, f, approx_elem_size * 4)
        vertices_f, indices_f = vtk_to_k3d(force_glyphs)
        if len(vertices_f) > 0 and len(indices_f) > 0:
            plot += k3d.mesh(vertices_f, indices_f, color=force_color)

    if c is not None:
        bc_glyphs = create_bc_glyphs(nodes, c, approx_elem_size)
        vertices_c, indices_c = vtk_to_k3d(bc_glyphs)
        if len(vertices_c) > 0 and len(indices_c) > 0:
            plot += k3d.mesh(vertices_c, indices_c, color=constraint_color)

    bbox_min = nodes.min(axis=0)
    bbox_max = nodes.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    size = np.linalg.norm(bbox_max - bbox_min)
    offset = size * 1.0

    camera_presets = {
        'bottom': [center[0], center[1] - offset, center[2], center[0], center[1], center[2], 0, 0, 1],
        'top':    [center[0], center[1] + offset, center[2], center[0], center[1], center[2], 0, 0, 1],
        'left':   [center[0] + offset, center[1], center[2], center[0], center[1], center[2], 0, 1, 0],
        'right':  [center[0] - offset, center[1], center[2], center[0], center[1], center[2], 0, 1, 0],
        'back':   [center[0], center[1], center[2] + offset, center[0], center[1], center[2], 0, 1, 0],
        'front':  [center[0], center[1], center[2] - offset, center[0], center[1], center[2], 0, 1, 0],
    }

    print(f"\n📝 Exporting {len(camera_presets)} HTML files...")
    html_files = {}

    for view_name, camera_pos in camera_presets.items():
        for mesh, vertices in surface_meshes:
            attr = _compute_depth_attribute(vertices, camera_pos)
            mesh.attribute = attr
            mesh.color_range = [0.0, 1.0]

        plot.camera = camera_pos
        html_path = os.path.join(output_dir, f'{view_name}.html')

        html_content = plot.get_snapshot()

        css_hide_panel = """
        <style>
        .k3d-panel, .k3d-control-panel, .k3d-panel-container {
            display: none !important;
        }
        </style>
        """
        html_content = html_content.replace('</head>', css_hide_panel + '</head>')

        with open(html_path, 'w') as f_out:
            f_out.write(html_content)

        html_files[view_name] = html_path
        print(f"✓ {view_name:15s} → {view_name}.html")

    print(f"\n📸 Capturing screenshots from HTML...")
    try:
        from playwright.async_api import async_playwright

        saved_files = {}
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(viewport={'width': 1920, 'height': 1080})

            for view_name, html_path in html_files.items():
                await page.goto(f'file://{os.path.abspath(html_path)}')
                await page.wait_for_timeout(int(delay * 1000) + 1000)

                canvas = await page.query_selector("canvas")
                box = await canvas.bounding_box()
                png_path = os.path.join(output_dir, f'{view_name}.png')
                await page.screenshot(path=png_path, clip=box)

                saved_files[view_name] = png_path
                print(f"✓ {view_name:15s} → {view_name}.png")

            await browser.close()

        print(f"\n✨ Success! {len(saved_files)} high-quality screenshots captured")
        return plot, camera_presets, saved_files

    except ImportError:
        print("\n⚠️  Playwright not installed. Install with:")
        print("    pip install playwright")
        print("    playwright install chromium")
        return plot, camera_presets, html_files
