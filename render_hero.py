"""Offscreen render of one molecule for the README header.

Left panel  : ball-and-stick geometry with the true and predicted dipole vectors.
Right panel : the RHF/6-31G* electron density isosurface, coloured by the
              electrostatic potential rebuilt from the fitted charges.
Run under a virtual display:  xvfb-run -a python3 render_hero.py 101807
"""
import json, re, sys
import numpy as np
import vtk
from vtk.util import numpy_support

BOHR = 0.5291772109
ISOVAL = 0.002
ARROW_SCALE = 0.46          # Angstrom per atomic unit, same as the web viewer
TRUE_RGB = (0.176, 0.831, 0.749)   # #2dd4bf
PRED_RGB = (0.961, 0.647, 0.141)   # #f5a524
JMOL = {1: (1.0, 1.0, 1.0), 6: (0.565, 0.565, 0.565), 7: (0.188, 0.314, 0.973),
        8: (1.0, 0.051, 0.051), 9: (0.565, 0.878, 0.314), 16: (1.0, 1.0, 0.188)}
RCOV = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 16: 1.05}
RBALL = {1: 0.24, 6: 0.35, 7: 0.34, 8: 0.33, 9: 0.32, 16: 0.42}


def read_cube(path):
    L = open(path).read().splitlines()
    h = L[2].split(); nat = abs(int(h[0]))
    origin = np.array([float(x) for x in h[1:4]])
    dims, axes = [], []
    for k in range(3):
        p = L[3 + k].split(); dims.append(int(p[0])); axes.append([float(x) for x in p[1:4]])
    Z, pos = [], []
    for a in range(nat):
        p = L[6 + a].split(); Z.append(int(float(p[0]))); pos.append([float(p[2]), float(p[3]), float(p[4])])
    vals = np.array(' '.join(L[6 + nat:]).split(), dtype=float).reshape(dims)
    return dict(origin=origin, dims=np.array(dims), axes=np.array(axes),
                Z=np.array(Z), pos=np.array(pos), rho=vals)


def esp_at(points_bohr, q, pos_bohr):
    """Potential from the fitted point charges, same formula the browser uses."""
    d = np.linalg.norm(points_bohr[:, None, :] - pos_bohr[None, :, :], axis=2)
    return (np.asarray(q)[None, :] / np.clip(d, 0.6, None)).sum(axis=1)


def sphere(center, radius, rgb, res=56):
    s = vtk.vtkSphereSource(); s.SetCenter(*center); s.SetRadius(radius)
    s.SetThetaResolution(res); s.SetPhiResolution(res)
    m = vtk.vtkPolyDataMapper(); m.SetInputConnection(s.GetOutputPort())
    a = vtk.vtkActor(); a.SetMapper(m)
    p = a.GetProperty(); p.SetColor(*rgb); p.SetSpecular(0.45); p.SetSpecularPower(45); p.SetAmbient(0.18)
    return a


def cylinder(p0, p1, radius, rgb, res=40, opacity=1.0):
    line = vtk.vtkLineSource(); line.SetPoint1(*p0); line.SetPoint2(*p1)
    tube = vtk.vtkTubeFilter(); tube.SetInputConnection(line.GetOutputPort())
    tube.SetRadius(radius); tube.SetNumberOfSides(res); tube.CappingOn()
    m = vtk.vtkPolyDataMapper(); m.SetInputConnection(tube.GetOutputPort())
    a = vtk.vtkActor(); a.SetMapper(m)
    p = a.GetProperty(); p.SetColor(*rgb); p.SetSpecular(0.4); p.SetSpecularPower(40)
    p.SetAmbient(0.2); p.SetOpacity(opacity)
    return a


def cone(tip_at, direction, length, radius, rgb):
    u = np.asarray(direction, float); u = u / np.linalg.norm(u)
    c = vtk.vtkConeSource(); c.SetResolution(64); c.SetHeight(length); c.SetRadius(radius)
    c.SetDirection(*u); c.SetCenter(*(np.asarray(tip_at) - u * length / 2.0))
    m = vtk.vtkPolyDataMapper(); m.SetInputConnection(c.GetOutputPort())
    a = vtk.vtkActor(); a.SetMapper(m)
    p = a.GetProperty(); p.SetColor(*rgb); p.SetSpecular(0.5); p.SetSpecularPower(50); p.SetAmbient(0.2)
    return a


def label(text, x, y, size, rgb, bold=False):
    t = vtk.vtkTextActor(); t.SetInput(text)
    pr = t.GetTextProperty()
    pr.SetFontFamilyToArial(); pr.SetFontSize(size); pr.SetColor(*rgb)
    pr.SetBold(bold); pr.SetJustificationToLeft()
    t.SetDisplayPosition(x, y)
    return t


def ball_stick(Z, pos):
    """Half-coloured bonds plus atom spheres, positions in Angstrom."""
    acts, n = [], len(Z)
    for i in range(n):
        acts.append(sphere(pos[i], RBALL.get(Z[i], 0.33), JMOL.get(Z[i], (0.8, 0.8, 0.8))))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < (RCOV.get(Z[i], 0.75) + RCOV.get(Z[j], 0.75)) * 1.3:
                mid = (pos[i] + pos[j]) / 2.0
                acts.append(cylinder(pos[i], mid, 0.105, JMOL.get(Z[i], (0.8, 0.8, 0.8))))
                acts.append(cylinder(mid, pos[j], 0.105, JMOL.get(Z[j], (0.8, 0.8, 0.8))))
    return acts


def dipole_actors(center, dtrue, dpred):
    """Solid arrow for the reference dipole, dashed sleeve for the prediction."""
    acts = []
    Lt = np.linalg.norm(dtrue) * ARROW_SCALE; ut = np.asarray(dtrue) / np.linalg.norm(dtrue)
    Lp = np.linalg.norm(dpred) * ARROW_SCALE; up = np.asarray(dpred) / np.linalg.norm(dpred)
    head = 0.62
    acts.append(cylinder(center, center + ut * (Lt - head * 0.55), 0.058, TRUE_RGB))
    acts.append(cone(center + ut * Lt, ut, head, 0.175, TRUE_RGB))
    dash, gap, s = 0.30, 0.20, 0.12
    while s + dash < Lp - head:
        acts.append(cylinder(center + up * s, center + up * (s + dash), 0.104, PRED_RGB))
        s += dash + gap
    tip = cone(center + up * Lp, up, head, 0.205, PRED_RGB)
    tip.GetProperty().SetOpacity(0.55)
    acts.append(tip)
    acts.append(sphere(center, 0.105, (0.91, 0.93, 0.96)))
    return acts


def aim_camera(ren, center, mu, zoom=1.3):
    """Look at the molecule from a direction perpendicular to the dipole,
    with the dipole running up the frame."""
    u = np.asarray(mu, float) / np.linalg.norm(mu)
    t = np.array([0.25, 0.85, 0.45])
    if abs(float(np.dot(t, u))) > 0.93:
        t = np.array([1.0, 0.0, 0.0])
    n = np.cross(u, t); n /= np.linalg.norm(n)
    cam = ren.GetActiveCamera()
    cam.SetViewUp(*u); cam.SetFocalPoint(*center); cam.SetPosition(*(center + n * 12.0))
    ren.ResetCamera()
    cam.Zoom(zoom)
    return cam


def main():
    mid = sys.argv[1] if len(sys.argv) > 1 else '101807'
    W, H, SS = 1680, 780, 2
    html = open('index.html').read()
    MOLS = json.loads(re.search(r'const MOLS = (\[.*?\]);', html, re.S).group(1))
    PRED = json.loads(re.search(r'const PRED\s*=\s*(\{.*?\})\s*;', html, re.S).group(1))
    mol = [x for x in MOLS if x['id'] == mid][0]
    cube = read_cube('dens/%s.cube' % mid)
    q = json.load(open('dens/charges.json'))[mid]

    posA = cube['pos'] * BOHR
    center = posA.mean(axis=0)
    dtrue = np.array(mol['dip'], float)
    dpred = np.array(PRED[mid], float)

    img = vtk.vtkImageData()
    img.SetDimensions(*[int(d) for d in cube['dims']])
    img.SetOrigin(*(cube['origin'] * BOHR))
    img.SetSpacing(*[cube['axes'][k][k] * BOHR for k in range(3)])
    flat = cube['rho'].transpose(2, 1, 0).ravel()
    arr = numpy_support.numpy_to_vtk(flat, deep=True); arr.SetName('rho')
    img.GetPointData().SetScalars(arr)

    mc = vtk.vtkFlyingEdges3D(); mc.SetInputData(img); mc.SetValue(0, ISOVAL); mc.ComputeNormalsOff(); mc.Update()
    sm = vtk.vtkWindowedSincPolyDataFilter(); sm.SetInputData(mc.GetOutput())
    sm.SetNumberOfIterations(24); sm.SetPassBand(0.08)
    sm.BoundarySmoothingOff(); sm.FeatureEdgeSmoothingOff(); sm.NormalizeCoordinatesOn(); sm.Update()
    nrm = vtk.vtkPolyDataNormals(); nrm.SetInputData(sm.GetOutput()); nrm.SetFeatureAngle(60); nrm.Update()
    surf = nrm.GetOutput()

    verts = numpy_support.vtk_to_numpy(surf.GetPoints().GetData()) / BOHR
    v = esp_at(verts, q, cube['pos'])
    lo, hi = np.percentile(v, [4, 96]); R = float(max(abs(lo), abs(hi), 0.012))
    sc = numpy_support.numpy_to_vtk(v, deep=True); sc.SetName('esp')
    surf.GetPointData().SetScalars(sc)
    print('isosurface: %d triangles, ESP on surface %.4f..%.4f a.u., colour range +-%.4f'
          % (surf.GetNumberOfPolys(), v.min(), v.max(), R))

    ctf = vtk.vtkColorTransferFunction()
    ctf.AddRGBPoint(-R, 1.00, 0.29, 0.31)
    ctf.AddRGBPoint(-R * 0.35, 1.00, 0.62, 0.55)
    ctf.AddRGBPoint(0.0, 0.97, 0.97, 0.99)
    ctf.AddRGBPoint(R * 0.35, 0.52, 0.72, 1.00)
    ctf.AddRGBPoint(R, 0.22, 0.47, 1.00)
    dmap = vtk.vtkPolyDataMapper(); dmap.SetInputData(surf)
    dmap.SetLookupTable(ctf); dmap.SetScalarRange(-R, R); dmap.ScalarVisibilityOn()
    dact = vtk.vtkActor(); dact.SetMapper(dmap)
    dp = dact.GetProperty()
    dp.SetOpacity(0.62); dp.SetSpecular(0.4); dp.SetSpecularPower(35); dp.SetAmbient(0.28); dp.SetDiffuse(0.85)

    renL = vtk.vtkRenderer(); renL.SetViewport(0.0, 0.0, 0.5, 1.0)
    renR = vtk.vtkRenderer(); renR.SetViewport(0.5, 0.0, 1.0, 1.0)
    kits = []
    for r in (renL, renR):
        r.GradientBackgroundOn(); r.SetBackground(0.027, 0.037, 0.060); r.SetBackground2(0.055, 0.076, 0.117)
        r.SetUseDepthPeeling(True); r.SetMaximumNumberOfPeels(16); r.SetOcclusionRatio(0.02)
        k = vtk.vtkLightKit(); k.SetKeyLightIntensity(1.05); k.AddLightsToRenderer(r); kits.append(k)
    for a in ball_stick(cube['Z'], posA):
        renL.AddActor(a)
    for a in ball_stick(cube['Z'], posA):
        renR.AddActor(a)
    for a in dipole_actors(center, dtrue, dpred):
        renL.AddActor(a)
    for a in dipole_actors(center, dtrue, dpred):
        renR.AddActor(a)
    renR.AddActor(dact)

    aim_camera(renL, center, dtrue, 0.96)
    aim_camera(renR, center, dtrue, 1.20)

    f = SS
    renL.AddActor2D(label('geometry + dipole', 26 * f, 34 * f, 15 * f, (0.62, 0.70, 0.80)))
    renR.AddActor2D(label('electron density, coloured by electrostatic potential', 26 * f, 34 * f, 15 * f, (0.62, 0.70, 0.80)))
    renR.AddActor2D(label('negative', 26 * f, 62 * f, 14 * f, (1.0, 0.45, 0.45)))
    renR.AddActor2D(label('/ positive', 92 * f, 62 * f, 14 * f, (0.42, 0.64, 1.0)))
    renL.AddActor2D(label('true mu (solid)', 26 * f, 84 * f, 14 * f, TRUE_RGB))
    renL.AddActor2D(label('predicted mu (dashed)', 26 * f, 62 * f, 14 * f, PRED_RGB))
    renL.AddActor2D(label(mol['smiles'], 26 * f, H * f - 46 * f, 19 * f, (0.86, 0.91, 0.96), True))

    rw = vtk.vtkRenderWindow(); rw.AddRenderer(renL); rw.AddRenderer(renR)
    rw.SetSize(W * SS, H * SS); rw.SetMultiSamples(0)
    rw.Render()
    w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(rw); w2i.ReadFrontBufferOff(); w2i.Update()
    wr = vtk.vtkPNGWriter(); wr.SetFileName('/tmp/hero_big.png')
    wr.SetInputConnection(w2i.GetOutputPort()); wr.Write()

    from PIL import Image
    im = Image.open('/tmp/hero_big.png').convert('RGB').resize((W, H), Image.LANCZOS)
    import os
    os.makedirs('assets', exist_ok=True)
    out = 'assets/hero_%s.png' % mid
    im.save(out, optimize=True)
    print('wrote %s  %dx%d  %d KB' % (out, W, H, os.path.getsize(out) // 1024))


if __name__ == '__main__':
    main()
