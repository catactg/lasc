import os
import vtk
import csv
import math
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import numpy as np
import sklearn.utils

#-----------------------------------------------------------------------
# SPECIFIC FUNCTIONS
#-----------------------------------------------------------------------

def check_images(ifile_image1, ifile_image2):
    """Check whether the extent, spacing and origin of two images are the
    same."""
    # read images
    print "Checking images", ifile_image1, "and", ifile_image2
    image1 = readmetaimage(ifile_image1)
    image2 = readmetaimage(ifile_image2)

    # compare image extents
    extent1 = image1.GetExtent()
    extent2 = image2.GetExtent()
    if (extent1[0] == extent2[0] and
        extent1[1] == extent2[1] and
        extent1[2] == extent2[2] and
        extent1[3] == extent2[3] and
        extent1[4] == extent2[4] and
        extent1[5] == extent2[5]):
        match = True
        print 'Extents match'
    else:
        match = False
        print ('Greyscale image and binary mask have different dimensions.\n' +
               'Fix this before proceeding with the benchmark.')

    # compare image spacings
    spacing1 = image1.GetSpacing()
    spacing2 = image2.GetSpacing()
    if (spacing1[0] == spacing2[0] and
        spacing1[1] == spacing2[1] and
        spacing1[2] == spacing2[2]):
        match = True
        print 'Spacings match'
    else:
        match = False
        print ('Greyscale image and binary mask have different spacings.\n' +
               'Fix this before proceeding with the benchmark.')

    # compare image origins
    origin1 = image1.GetOrigin()
    origin2 = image2.GetOrigin()
    if (origin1[0] == origin2[0] and
        origin1[1] == origin2[1] and
        origin1[2] == origin2[2]):
        match = True
        print 'Origin match'
    else:
        match = False
        print ('Greyscale image and binary mask have different origins.\n' +
               'Fix this before proceeding with the benchmark.')
    return match

def clip_mitral(surface, ifile_plane):
    """Clip atrium at level of mitral valve."""
    # read plane information from csv-file
    planeinfo = np.genfromtxt(ifile_plane, delimiter=',')
    normal = planeinfo[0].flatten().tolist()
    point = planeinfo[1].flatten().tolist()
    insideout = int(planeinfo[2, 0].flatten())

    # clip atrium at level of mitral valve
    clippedsurface = planeclip(surface, point, normal, insideout)
    clippedsurface = extractlargestregion(clippedsurface)
    return clippedsurface


def clip_vein_endpoint(surface, ifile_sufix, targetdistance):
    """Clip vein the targetdistance away from the body."""
    regionslabels = getregionslabels()

    # extract the body from the surface
    # including all points (alloff=1) to avoid holes after appending
    body = pointthreshold(surface, 'autolabels',
                          regionslabels['body'], regionslabels['laa'], 1)
    body = extractlargestregion(body)

    # initialize appender with the body
    appender = vtk.vtkAppendPolyData()
    appender.AddInput(body)

    for k in range(1,5):
        index = 'pv' + str(k)
        # extract vein
        # excluding some points (alloff=0)
        # to avoid overlapping edges after appending
        vein = pointthreshold(surface, 'autolabels', regionslabels[index],
                               regionslabels[index], 0)

        # load the centreline and the clipoint
        cl = readvtp(os.path.join(ifile_sufix,
                                'clvein' + str(k) + '.vtp'))
        clippointid = int(np.loadtxt(os.path.join(ifile_sufix,
                                'clippointid' + str(k) + '.csv')))

        clippoint0 = cl.GetPoint(clippointid)
        clipnormal = (np.array(cl.GetPoint(clippointid + 1)) -
                      np.array(cl.GetPoint(clippointid - 1)))

        abscissasarray = cl.GetPointData().GetArray('Abscissas')
        startabscissa = abscissasarray.GetValue(clippointid)
        currentabscissa = 0
        currentid = clippointid

        # find clip point
        while ((currentabscissa < targetdistance) and
               (currentabscissa >= 0) and
               (currentid >= 0)):
            currentid -= 1
            currentabscissa = startabscissa - abscissasarray.GetValue(currentid)

        if currentid > 0:
            currentid = currentid + 1
        else:
            # vein ended before target distance
            # then clip 2 mm before end of centreline (5x0.4 mm) from end point
            currentid = 4

        # clip and append
        clippoint1 = cl.GetPoint(currentid)
        clippedvein = planeclip(vein, clippoint1, clipnormal, 0)

        # keep region closest to ostium point
        clippedvein = extractclosestpointregion(clippedvein, clippoint0)

        # clip generates new points to make a flat cut. The values may be
        # interpolated. we want all values to rounded to a certain label value.
        clippedvein = roundpointarray(clippedvein, 'autolabels')
        appender.AddInput(clippedvein)

    # collect body + veins
    appender.Update()
    clippedsurface = appender.GetOutput()
    clippedsurface = cleanpolydata(clippedsurface)
    return clippedsurface

def compute_dice(ifile_image, surfacetarget, ofile=''):
    """Generate image from surfacetarget and compute Dice-metric with respect to
    ifile_image."""
    # load the whole image
    refimage = readmetaimage(ifile_image)
    orispacing = refimage.GetSpacing()
    spacing = [orispacing[0], orispacing[1], orispacing[2]]

    bounds = refimage.GetBounds()

    # make reference and target image
    # body label is generated with the whole surface
    indexes = ['laa','pv1','pv2','pv3','pv4']
    targetimage = imagefordice(surfacetarget, spacing, bounds, indexes,
                               'autolabels')

    if ofile:
        print "saving", ofile
        writemetaimage(targetimage, ofile)

    # extract each label and compute metric body and pvs (no laa)
    regionslabels = getregionslabels()

    # initialise metric dictionary
    metric_all = {'body': [0.], 'pvs': [0.]}

    # body metric
    refimagelabel = imagethresholdbetween(refimage,
                                          regionslabels['body'] - 0.5,
                                          regionslabels['body'] + 0.5)
    targetimagelabel = imagethresholdbetween(targetimage,
                                             regionslabels['body'] - 0.5,
                                             regionslabels['body'] + 0.5)

    # compute metric
    metric = dicemetric(refimagelabel, targetimagelabel)
    metric_all['body'] = [metric]

    # pvs metric
    metric = [0., 0., 0., 0.]
    for k in range(0, 4):
        index = 'pv' + str(k + 1)
        refimagelabel = imagethresholdbetween(refimage,
                                              regionslabels[index] - 0.5,
                                              regionslabels[index] + 0.5)
        targetimagelabel = imagethresholdbetween(targetimage,
                                                 regionslabels[index] - 0.5,
                                                 regionslabels[index] + 0.5)
        # compute metric
        metric[k] = dicemetric(refimagelabel, targetimagelabel)
    metric_all['pvs'] =  metric
    return metric_all


def compute_s2s_error(surface, surfacetarget, nsamples, ofile=''):
    """Compute the symmetric surface-to-surface distance (s2s-metric) for
    surface and surfacetarget. Resample the s2s-array to nsamples."""

    # cap the surfaces to improve surface to surface distance on clipped areas
    surfacecap = capsurface(surface,'autolabels')
    edges = extractboundaryedge(surfacecap)
    if edges.GetNumberOfPoints() > 0:
        surfacecap = fillholes(surfacecap)

    surfacetargetcap = capsurface(surfacetarget, 'autolabels')
    edges = extractboundaryedge(surfacetargetcap)
    if edges.GetNumberOfPoints() > 0:
        surfacetargetcap = fillholes(surfacetargetcap)

    # compute distances
    seg2gtsurf = surface2surfacedistance(surfacecap, surfacetargetcap, 'S2S')
    gt2segsurf = surface2surfacedistance(surfacetargetcap, surfacecap, 'S2S')

    if ofile:
        writevtp(seg2gtsurf, ofile + 'seg2gt.vtp')
        writevtp(gt2segsurf, ofile + 'gt2seg.vtp')

    # to have ~ same amount of samples per case
    # extract body and pvs
    # re sample to nsamples per case
    indexes = ['body', 'pvs']
    rfrom = {'body': 36, 'pvs': 76}
    rto = {'body': 36, 'pvs': 79}

    # initialise metric dictionary
    s2s_all = {'body': [0.], 'pvs': [0.]}
    for index in indexes:
        # extracting each region
        # including all points
        seg2gtsurf = pointthreshold(seg2gtsurf, 'autolabels',
                                    rfrom[index], rto[index], 1)
        gt2segsurf = pointthreshold(gt2segsurf, 'autolabels',
                                    rfrom[index], rto[index], 1)
        # turning distance array into numpy
        if (seg2gtsurf.GetPointData().GetArray('S2S') != None):
            seg2gtarray = vtk_to_numpy(seg2gtsurf.GetPointData().
                                       GetArray('S2S'))
        else:
            seg2gtarray = []
        if (gt2segsurf.GetPointData().GetArray('S2S') != None):
            gt2segarray = vtk_to_numpy(gt2segsurf.GetPointData().
                                       GetArray('S2S'))
        else:
            gt2segarray = []

        # resample to nsamples (unless array smaller than that)
        if len(seg2gtarray) > nsamples:
            seg2gtarray = resamplearray(seg2gtarray, nsamples)
        if len(gt2segarray) > nsamples:
            gt2segarray = resamplearray(gt2segarray, nsamples)

        # concatenate error to have symmetric metric
        superarray = np.concatenate([seg2gtarray, gt2segarray])
        s2s_all[index] = superarray
    return s2s_all


def labels2mesh(image, label, radius=1.):
    """Generate mesh from label image using Marching Cubes algorithm."""
    # threshold all values above label
    image = imagethresholdupper(image, label)
    image = imageopenclose(image, 0, 1, radius)

    # after threshold the image is binary
    surface = marchingcubes(image, 1, 1)
    surface = smoothtaubin(surface)
    surfaceout = cleanpolydata(surface)
    return surfaceout


def transfer_gtlabels(surface, target, arrayname):
    """Project labels in array with arrayname from surface to target."""
    # labels
    regionslabels = getregionslabels()
    indexes = ['pv1', 'pv2', 'pv3', 'pv4', 'laa']

    # cleaning
    target = cleanpolydata(target)
    numberofpoints = target.GetNumberOfPoints()

    # create array
    gtlabelsarray = vtk.vtkDoubleArray()
    gtlabelsarray.SetName(arrayname)
    gtlabelsarray.SetNumberOfTuples(numberofpoints)
    target.GetPointData().AddArray(gtlabelsarray)

    # initialize with body label
    gtlabelsarray.FillComponent(0, regionslabels['body'])

    # get labels from surface
    gtlabelsurface = surface.GetPointData().GetArray(arrayname)

    # initiate locator
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()

    # go through each point of target surface
    for i in range(numberofpoints):
        # determine closest point on surface
        point = target.GetPoint(i)
        closestpointid = locator.FindClosestPoint(point)
        # get label of point
        value = gtlabelsurface.GetValue(closestpointid)
        # assign label to target point
        gtlabelsarray.SetValue(i, value)

    # check that there is only one region per pv/laa label
    for index in indexes:
        # for each region, check if there are other regions on the surface
        # with the same label. If so, keep largest region
        # and relabel small regions to body label
        target = filldisconnectedregion(target,
                                        arrayname,
                                        regionslabels[index],
                                        regionslabels['body'])

        # for each region, fill small patches (i.e. body label)
        # with corresponding region label
        target = fillpatch(target,
                           arrayname,
                           regionslabels[index],
                           regionslabels['body'])

    # relabel isolated points in the body with vein label (e.g. close to ostia)
    target = fillpatch(target,
                       arrayname,
                       regionslabels['body'],
                       regionslabels['body'])

    return target

#-----------------------------------------------------------------------
# GENERAL FUNCTIONS
#-----------------------------------------------------------------------

def addvectors(point1, point2):
    """Add two vectors."""
    return [point1[0] + point2[0],
            point1[1] + point2[1],
            point1[2] + point2[2]]

def cellthreshold(polydata, arrayname, start=0, end=1):
    """Extract those cells from polydata whose celldata values are within a
    specified range."""
    threshold = vtk.vtkThreshold()
    threshold.SetInput(polydata)
    threshold.SetInputArrayToProcess(0, 0, 0,
                                     vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,
                                     arrayname)
    threshold.ThresholdBetween(start, end)
    threshold.Update()

    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInput(threshold.GetOutput())
    surfer.Update()
    return surfer.GetOutput()

def cleanpolydata(polydata):
    """Apply VTK mesh cleaning filter to polydata."""
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInput(polydata)
    cleaner.Update()
    return cleaner.GetOutput()

def dicemetric(image_reference, image_target):
    """Compute overlap between two images."""
    # for overlap image: add two images and extract region with value a + b
    image_overlap = imagesum(image_reference, image_target)
    overlap = imagethresholdupper(image_overlap, 2)

    # image scalars to numpy array
    # this allows us to have quick access to pixel data
    reference_array = vtk_to_numpy(image_reference.GetPointData().GetScalars())
    target_array = vtk_to_numpy(image_target.GetPointData().GetScalars())
    overlap_array = vtk_to_numpy(overlap.GetPointData().GetScalars())

    # to compute dice metric, count non-zero pixels
    pix_reference = np.sum(reference_array)
    pix_target = np.sum(target_array)
    pix_overlap = np.sum(overlap_array)

    if (pix_target + pix_reference ) > 0:
        DM = 2.0 * pix_overlap / (pix_target + pix_reference)
    else:
        DM = 0
    print 'Dice metric', DM
    return DM

def mindistancetopolydata(reference, polydata):
    """Compute minimum distance between two polydata."""
    refdist = 1000000

    # initiate point locator
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(reference)
    locator.BuildLocator()

    # go through each point of polydata
    for i in range(polydata.GetNumberOfPoints()):
        point = polydata.GetPoint(i)
        # determine closest point on target.
        closestpointid = locator.FindClosestPoint(point)
        dist = euclideandistance(point,
                                 reference.GetPoint(closestpointid))
        if dist < refdist:
            refdist = dist
    return refdist

def euclideandistance(point1, point2):
    """Compute Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 +
                     (point1[1] - point2[1])**2 +
                     (point1[2] - point2[2])**2)


def extractboundaryedge(polydata):
    """Extract boundary edges of a surface mesh."""
    edge = vtk.vtkFeatureEdges()
    edge.SetInput(polydata)
    edge.FeatureEdgesOff()
    edge.NonManifoldEdgesOff()
    edge.Update()
    return edge.GetOutput()


def extractconnectedregion(polydata, regionid):
    """Run connectivity filter to assign regionsids and return region with
    given regionid."""
    # extract surface
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInput(polydata)
    surfer.Update()

    # clean before connectivity filter
    # to avoid artificial regionIds
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInput(surfer.GetOutput())
    cleaner.Update()

    # extract all regions
    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInput(cleaner.GetOutput())
    connect.SetExtractionModeToAllRegions()
    connect.ColorRegionsOn()
    connect.Update()

    # threshold especified region
    surface = pointthreshold(connect.GetOutput(), 'RegionId',
                             float(regionid), float(regionid))
    return surface


def extractclosestpointregion(polydata, point=[0, 0, 0]):
    """Extract region closest to specified point."""
    # extract surface
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInput(polydata)
    surfer.Update()

    # clean before connectivity filter
    # to avoid artificial regionIds
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInput(surfer.GetOutput())
    cleaner.Update()

    # extract regions closest to point
    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInput(cleaner.GetOutput())
    connect.SetExtractionModeToClosestPointRegion()
    connect.SetClosestPoint(point)
    connect.FullScalarConnectivityOn()
    connect.Update()
    return connect.GetOutput()


def extractlargestregion(polydata):
    """Extract largest of several disconnected regions."""
    # extract surface
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInput(polydata)
    surfer.Update()

    # clean before connectivity filter
    # to avoid artificial regionIds
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInput(surfer.GetOutput())
    cleaner.Update()

    # extract largest region
    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInput(cleaner.GetOutput())
    connect.SetExtractionModeToLargestRegion()
    connect.Update()

    # cleaning phantom points
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInput(connect.GetOutput())
    cleaner.Update()
    return cleaner.GetOutput()


def fillholes(polydata, holesize=1000000):
    """Fill holes in surface. Use holesize to specify the maximum 'radius' of the
    holes to be filled."""
    filler = vtk.vtkFillHolesFilter()
    filler.SetInput(polydata)
    filler.SetHoleSize(holesize)
    filler.Update()
    return filler.GetOutput()

def fillpatch(surface, arrayname, value, patchvalue):
    """Replace value for patchvalue in specified array of surface."""
    # extract main body
    body = pointthreshold(surface, arrayname, patchvalue, patchvalue)
    mainbody = extractlargestregion(body)
    areamainbody = surfacearea(mainbody)
    edgesmainbody = extractboundaryedge(mainbody)

    # extract subpart
    submesh = pointthreshold(surface, arrayname, value, value, 0)
    # assuming the patch label is known
    patches = pointthreshold(surface, arrayname, patchvalue, patchvalue, 0)

    # if there is more than one edge, smaller edges should be patches
    edges = extractboundaryedge(submesh)
    if edges.GetNumberOfPoints() > 0:
        # the edge closest to the body is the ostium
        # hence it should not be a patch
        smallestdistr = findadjoiningregionid(edgesmainbody,edges)
        nedges = getregionsrange(edges)
        # loop again to fill the patch
        for r in range(int(nedges[1]) + 1):
            # taking centroid of edge
            smalledge = extractconnectedregion(edges, r)
            centroid = pointsetcentreofmass(smalledge)

            if (r != smallestdistr):
                patch = extractclosestpointregion(patches, centroid)
                # check patch is smaller than body, based on surface area
                areapatch = surfacearea(patch)
                if areapatch < 0.5 * areamainbody:

                    transferlabels(surface, patch, arrayname, value)
    return surface

def filldisconnectedregion(targetsurface, arrayname, label, rlabel):
    """Find disconnected regions with the same label.
    Replace its label to the second closest label."""
    # extract label region
    subpd = pointthreshold(targetsurface, arrayname, label, label, 1)

    if subpd.GetNumberOfPoints() > 0:
        # find largest regionid
        regionid = findlargestregionid(subpd)
        regionsrange = getregionsrange(subpd)

        # loop through all regions to relabel small regions
        if regionsrange[1] > 0.0:
            for j in range(int(regionsrange[1]) + 1):
                # for other regions, replace value
                if j != regionid:
                    subsubpd = extractconnectedregion(subpd, j)
                    transferlabels(targetsurface, subsubpd, arrayname, rlabel)
    return targetsurface

def findadjoiningregionid(reference, target):
    """Find the regionid in target closest to any reference region."""
    nregions = getregionsrange(target)
    smallestdist = 1000000
    smallestdistr = 0

    # iterate over regions to find the adjoining regions
    if nregions > 0:
        for r in range(int(nregions[1]) + 1):
            # taking centroid of region
            smallregion = extractconnectedregion(target, r)
            # find region closest to reference
            currentdist = mindistancetopolydata(reference, smallregion)
            if currentdist < smallestdist:
                smallestdist = currentdist
                smallestdistr = r
    return smallestdistr

def findlargestregionid(polydata):
    """Get id of largest of several disconnected regions."""
    # extract surface
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInput(polydata)
    surfer.Update()

    # clean before connectivity filter
    # to avoid artificial regionIds
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInput(surfer.GetOutput())
    cleaner.Update()

    # extract all connected regions
    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInput(cleaner.GetOutput())
    connect.SetExtractionModeToAllRegions()
    connect.ColorRegionsOn()
    connect.Update()

    # extract surface
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInput(connect.GetOutput())
    surfer.Update()

    # compute range
    regions =  surfer.GetOutput().GetPointData().GetArray('RegionId')
    regionsrange = regions.GetRange()
    maxpoints = 0
    largestregionid = regionsrange[0]

    # if more than one region, find the largest
    if (regionsrange[1] > 0.0):
        for j in range(int(regionsrange[0]), int(regionsrange[1]) + 1):
            outsurf = pointthreshold(surfer.GetOutput(), 'RegionId', j, j)
            numberofpoints = outsurf.GetNumberOfPoints()

            if (numberofpoints> maxpoints):
                maxpoints = numberofpoints
                largestregionid = j
    return largestregionid


def getregionsrange(polydata):
    """Return range of connected regions."""
    # extract surface
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInput(polydata)
    surfer.Update()

    # clean before connectivity filter
    # to avoid artificial regionIds
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInput(surfer.GetOutput())
    cleaner.Update()

    # extract all connected regions
    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInput(cleaner.GetOutput())
    connect.SetExtractionModeToAllRegions()
    connect.ColorRegionsOn()
    connect.Update()

    # extract surface
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInput(connect.GetOutput())
    surfer.Update()

    # get range
    regions =  surfer.GetOutput().GetPointData().GetArray('RegionId')
    regionsrange = regions.GetRange()
    return regionsrange


def imagefordice(surface, spacing, bounds, indexes, arrayname):
    """Compute image from surface using imagestencil. Values of labels
        on surface are preserved."""
    # labels
    regionslabels = getregionslabels()

    # cap whole surface with flat covers
    surfacecap = capsurface(surface)

    # if we still have edges, run fill holes
    edges = extractboundaryedge(surfacecap)
    if edges.GetNumberOfPoints() > 0:
        surfacecap = fillholes(surfacecap)

    # make image with the whole surface
    wholeimage = mesh2image(surfacecap, spacing, bounds, 1)

    # each label
    for index in indexes:
        vein = pointthreshold(surface, arrayname,
                              regionslabels[index], regionslabels[index], 0)
        # cap vein with flat covers
        veincap = capsurface(vein)
        # if we still have edges, run fill holes
        edges = extractboundaryedge(veincap)
        if edges.GetNumberOfPoints() > 0:
            veincap = fillholes(veincap)

        # generate image using imagestencil
        veinimage = mesh2image(veincap, spacing, bounds, 1)
        # add the images
        wholeimage = imagesum(wholeimage, veinimage)
        # pixels with value 2 correspond to vein region
        wholeimage = imagereplacevalue(wholeimage, 2, regionslabels[index])

    # pixels remaining with value 1 correspond to body
    wholeimage = imagereplacevalue(wholeimage, 1, regionslabels['body'])
    return wholeimage


def getregionslabels():
    """Return dictionary linking regionids to anatomical locations."""
    regionslabels = {'body': 36,
                     'laa': 37,
                     'pv2': 76,
                     'pv1': 77,
                     'pv3': 78,
                     'pv4': 79}
    return regionslabels


def mesh2image(pd, spacing, bounds, value=255):
    """Generates an image in which pixels inside the surface set to value."""
    # start with white image
    dim, origin = bounds_dim_origin(bounds,spacing)
    whiteimage = image_from_value(spacing,dim,origin,value)

    # polygonal data to image stencil
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInput(pd)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputSpacing(spacing)
    pol2stenc.SetOutputWholeExtent(0, dim[0] ,
                                   0, dim[1] ,
                                   0, dim[2] )
    pol2stenc.Update()

    # cut the corresponding white image and set the background
    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInput(whiteimage)
    imgstenc.SetStencil(pol2stenc.GetOutput())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()
    return imgstenc.GetOutput()


def bounds_dim_origin(bounds,spacing):
    """Compute origin and extent based on bounds."""
    # compute dimensions
    dim = [0, 0, 0]
    origin = [0, 0, 0]
    for i in range(0, 3):
        dim[i] = int(math.ceil((bounds[i * 2 + 1] - bounds[i * 2]) /
                               spacing[i]))
    # origin is the lower bound (double)
    for i in range(0, 3):
        origin[i] = bounds[i * 2]
    return dim,origin


def image_from_value(spacing, dim, origin, value=255):
    """Generate a white image with a defined spacing and bounds."""
    # initialise image
    image =  vtk.vtkImageData()
    image.SetSpacing(spacing)
    image.SetDimensions(dim)
    image.SetExtent(0, dim[0],
                    0, dim[1],
                    0, dim[2])
    image.SetOrigin(origin)
    image.SetScalarTypeToUnsignedChar()
    image.SetNumberOfScalarComponents(1)
    image.AllocateScalars()

    imagescalars = image.GetPointData().GetScalars()
    # Fill component is much faster than visiting each element to set value
    imagescalars.FillComponent(0, value)
    image.Update()
    return image


def capsurface(polydata, arrayname=''):
    """Cap holes in surface with a flat cover."""
    # generates a flat cover for a convex hole defined by edges
    fedges = extractboundaryedge(polydata)

    # find each connected edge
    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInput(fedges)
    connect.Update()

    ncontours = connect.GetNumberOfExtractedRegions()

    append = vtk.vtkAppendPolyData()
    append.AddInput(polydata)

    # generate each flat cover
    for i in range(ncontours):
        connect.AddSpecifiedRegion(i)
        connect.SetExtractionModeToSpecifiedRegions()
        connect.Update()
        edges = connect.GetOutput()
        cover = vtk.vtkPolyData()
        generatecover(edges, cover, arrayname)
        # append to original polydata
        append.AddInput(cover)
        connect.DeleteSpecifiedRegion(i)

    append.Update()
    outsurface = cleanpolydata(append.GetOutput())
    return outsurface


def generatecover(edges, cover, arrayname=''):
    """Create caps for capping a surface with holes."""
    # create the building blocks of polydata.
    polys = vtk.vtkCellArray()
    points = vtk.vtkPoints()

    surfilt = vtk.vtkCleanPolyData()
    surfilt.SetInput( edges )
    surfilt.Update()

    points.DeepCopy(surfilt.GetOutput().GetPoints())
    npoints = points.GetNumberOfPoints()

    if arrayname:
        # keep pre existing array
        array = surfilt.GetOutput().GetPointData().GetArray(arrayname)
        arraynp = vtk_to_numpy(array)
        array.InsertNextValue(np.mean(arraynp))

    # add centroid
    centr = np.zeros(3)
    for i in range( npoints ):
        pt = np.zeros(3)
        points.GetPoint(i,pt)
        centr = centr + pt

    centr = centr / npoints
    cntpt = points.InsertNextPoint(centr)

    # add cells
    for i in range(surfilt.GetOutput().GetNumberOfCells()):
        cell = surfilt.GetOutput().GetCell(i)
        polys.InsertNextCell(3)
        polys.InsertCellPoint(cell.GetPointId(0))
        polys.InsertCellPoint(cell.GetPointId(1))
        polys.InsertCellPoint(cntpt)

    # assign the pieces to the polydata
    cover.SetPoints(points)
    cover.SetPolys(polys)
    if arrayname:
        cover.GetPointData().AddArray(array)


def imageopenclose(image, openvalue, closevalue, kernelsize):
    """Performs opening and closing morphological operations
        with a 3D ellipsoidal kernel."""
    openClose = vtk.vtkImageOpenClose3D()
    openClose.SetInput(image)
    openClose.SetOpenValue(openvalue)
    openClose.SetCloseValue(closevalue)
    openClose.SetKernelSize(kernelsize, kernelsize, kernelsize)
    openClose.ReleaseDataFlagOff()
    openClose.GetOutput()
    openClose.GetCloseValue()
    openClose.GetOpenValue()
    return openClose.GetOutput()


def imagereplacevalue(image, const1, const2):
    """Replaces the scalar value in a image with another."""
    kfilter = vtk.vtkImageMathematics()
    kfilter.SetInput1(image)
    kfilter.SetConstantC(const1)
    kfilter.SetConstantK(const2)
    kfilter.SetOperationToReplaceCByK()
    kfilter.Update()
    return kfilter.GetOutput()


def imagesum(image1, image2):
    """Adds two images."""
    sumfilter = vtk.vtkImageMathematics()
    sumfilter.SetInput1(image1)
    sumfilter.SetInput2(image2)
    sumfilter.SetOperationToAdd()
    sumfilter.Update()
    return sumfilter.GetOutput()


def imagethresholdbetween(image, t1, t2, invalue=1.0):
    """Thresholds an image between t1 and t2."""
    tfilter = vtk.vtkImageThreshold()
    tfilter.SetInput(image)
    tfilter.ThresholdBetween(t1, t2)
    tfilter.SetOutValue(0.0)
    tfilter.SetInValue(invalue)
    tfilter.Update()
    return tfilter.GetOutput()

def imagethresholdupper(image, t, invalue=1.0):
    """Thresholds values equal or greater than t."""
    tfilter = vtk.vtkImageThreshold()
    tfilter.SetInput(image)
    tfilter.ThresholdByUpper(t)
    tfilter.SetOutValue(0.0)
    tfilter.SetInValue(invalue)
    tfilter.Update()
    return tfilter.GetOutput()


def marchingcubes(image, startlabel, endlabel):
    """Generates object boundaries from labelled volumes using
        Marching Cubes algorithm."""
    discretecubes = vtk.vtkDiscreteMarchingCubes()
    discretecubes.SetInput(image)
    discretecubes.GenerateValues(endlabel - startlabel + 1,
                                 startlabel, endlabel)
    discretecubes.Update()
    return discretecubes.GetOutput()

def planeclip(surface, point, normal, insideout=1):
    """Clip a surface using the plane perpendicular to normal
        and centred at point."""
    clipplane = vtk.vtkPlane()
    clipplane.SetOrigin(point)
    clipplane.SetNormal(normal)
    clipper = vtk.vtkClipPolyData()
    clipper.SetInput(surface)
    clipper.SetClipFunction(clipplane)

    if insideout:
        clipper.InsideOutOn()
    else:
        clipper.InsideOutOff()
    clipper.Update()
    return clipper.GetOutput()

def pointsetcentreofmass(polydata):
    """Compute the centre of mass of a polydata."""
    centre = [0, 0, 0]
    for i in range(polydata.GetNumberOfPoints()):
        point = [polydata.GetPoints().GetPoint(i)[0],
          polydata.GetPoints().GetPoint(i)[1],
          polydata.GetPoints().GetPoint(i)[2]]
        centre = addvectors(centre, point)
    return dividevector(centre, polydata.GetNumberOfPoints())

def dividevector(point, n):
    """Divide vector by scalar value."""
    nr = float(n)
    return [point[0]/nr, point[1]/nr, point[2]/nr]

def pointthreshold(polydata, arrayname, start=0, end=1, alloff=0):
    """Threshold between start and end values in array. By default,
        threshold excludes points whose neighbours do not satisfy
        the threshold value.  Enabling the flag 'alloff' disables
        this setting to include all points."""
    threshold = vtk.vtkThreshold()
    threshold.SetInput(polydata)
    threshold.SetInputArrayToProcess(0, 0, 0,
                                     vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
                                     arrayname)
    threshold.ThresholdBetween(start, end)
    if (alloff):
        threshold.AllScalarsOff()
    threshold.Update()

    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInput(threshold.GetOutput())
    surfer.Update()
    return surfer.GetOutput()


def readmetaimage(filename):
    """Read a metaimage."""
    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def readvtp(filename, dataarrays=True):
    """Read polydata in XML format."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    if not dataarrays:
        for i in range(reader.GetNumberOfPointArrays()):
            arrayname = reader.GetPointArrayName(i)
            reader.SetPointArrayStatus(arrayname, 0)
        for i in range(reader.GetNumberOfCellArrays()):
            arrayname = reader.GetCellArrayName(i)
            reader.SetPointArrayStatus(arrayname, 0)
        reader.Update()
    return reader.GetOutput()


def resamplearray(x, n):
    """Resample an array to n samples using a bootstrapping technique."""
    y = sklearn.utils.resample(x, n_samples=n)
    return y


def roundpointarray(polydata, name):
    """Round values in point array."""
    # get original array
    array = polydata.GetPointData().GetArray(name)

    # round labels
    for i in range(polydata.GetNumberOfPoints()):
        value = array.GetValue(i)
        array.SetValue(i, round(value))
    return polydata


def smoothtaubin(polydata, iterations=15, angle=120, passband=0.001):
    """Execute volume reserving smoothing."""
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInput(polydata)
    smoother.SetNumberOfIterations(iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(angle)
    smoother.SetPassBand(passband)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return smoother.GetOutput()


def surfacearea(polydata):
    """Compute surface area of polydata."""
    properties = vtk.vtkMassProperties()
    properties.SetInput(polydata)
    properties.Update()
    return properties.GetSurfaceArea()

def surface2surfacedistance(ref, target, arrayname):
    """Compute distance between two surfaces. Output is added as point array."""
    # adapted from vtkvmtkSurfaceDistance
    # initialise
    locator = vtk.vtkCellLocator()
    genericcell = vtk.vtkGenericCell()
    cellid = vtk.mutable(0)
    point = [0., 0., 0.]
    closestpoint = [0., 0., 0.]
    subid = vtk.mutable(0)
    distance2 = vtk.mutable(0)

    # create array
    distarray = vtk.vtkDoubleArray()
    distarray.SetName(arrayname)
    distarray.SetNumberOfTuples(target.GetNumberOfPoints())
    target.GetPointData().AddArray(distarray)

    # build locator
    locator.SetDataSet(ref)
    locator.BuildLocator()

    # compute distance
    for i in range(target.GetNumberOfPoints()):
        point = target.GetPoint(i)
        locator.FindClosestPoint(point, closestpoint, genericcell, cellid,
                                 subid, distance2)
        distance = math.sqrt(distance2)
        # add value to array
        distarray.SetValue(i, distance)

    target.Update()
    return target


def transferlabels(target, reference, arrayname, value):
    """Project array from reference surface to target surface using closest point."""
    # initiate point locator
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(target)
    locator.BuildLocator()

    # get array from target
    array = target.GetPointData().GetArray(arrayname)

    # go through each point of reference target
    for i in range(reference.GetNumberOfPoints()):
        point = reference.GetPoint(i)
        # determine closest point on target.
        closestpointid = locator.FindClosestPoint(point)
        array.SetValue(closestpointid, value)
    return target


def visualise(surface, reference, case, arrayname, mini, maxi):
    """Visualise surface with colormap based on arrayname.
        Reference surface is visualise with alpha = 0.5."""

    #Create a lookup table to map point data to colors
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(255)
    lut.SetValueRange(0, 255)

    # qualitative data from colorbrewer
    lut.SetTableValue(0, 0, 0, 0, 1)  #black
    lut.SetTableValue(mini, 1, 1, 1, 1) # white
    lut.SetTableValue(mini + 1, 77/255., 175/255., 74/255. , 1)  # green
    lut.SetTableValue(maxi - 3, 152/255., 78/255., 163/255., 1) # purple
    lut.SetTableValue(maxi - 2, 255/255., 127/255., 0., 1) # orange
    lut.SetTableValue(maxi - 1, 55/255., 126/255., 184/255., 1) # blue
    lut.SetTableValue(maxi, 166/255., 86/255., 40/255., 1) # brown
    lut.Build()

    # create a text actor
    txt = vtk.vtkTextActor()
    txt.SetInput(case)
    txtprop=txt.GetTextProperty()
    txtprop.SetFontFamilyToArial()
    txtprop.SetFontSize(18)
    txtprop.SetColor(0, 0, 0)
    txt.SetDisplayPosition(20, 30)

    # create a rendering window, renderer, and renderwindowinteractor
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)
    iren.SetRenderWindow(renWin)

    # surface mapper and actor
    surfacemapper = vtk.vtkPolyDataMapper()
    surfacemapper.SetInput(surface)
    surfacemapper.SetScalarModeToUsePointFieldData()
    surfacemapper.SelectColorArray(arrayname)
    surfacemapper.SetLookupTable(lut)
    surfacemapper.SetScalarRange(0, 255)
    surfaceactor = vtk.vtkActor()
    surfaceactor.SetMapper(surfacemapper)

    # refsurface mapper and actor
    refmapper = vtk.vtkPolyDataMapper()
    refmapper.SetInput(reference)
    refmapper.SetScalarModeToUsePointFieldData()
    refmapper.SelectColorArray(arrayname)
    refmapper.SetLookupTable(lut)
    refmapper.SetScalarRange(0, 255)
    refactor = vtk.vtkActor()
    refactor.GetProperty().SetOpacity(0.5)
    refactor.SetMapper(refmapper)

    # assign actors to the renderer
    ren.AddActor(refactor)
    ren.AddActor(surfaceactor)
    ren.AddActor(txt)

    # set the background and size, zoom in and render
    ren.SetBackground(1, 1, 1)
    renWin.SetSize(1280, 960)
    ren.ResetCamera()
    ren.GetActiveCamera().Zoom(1)

    # enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()

def writearray2csv(array, ofile, label=''):
    """Write array to csv."""
    f = open(ofile, 'wb')
    for i in range(len(array)):
        if label:
            line = str(label[i]) + ', ' + str(array[i]) + '\n'
        else:
            line = str(array[i]) + '\n'
        f.write(line)
    f.close()


def writevtk(surface, filename):
    """Write vtkPolyData file."""
    writer = vtk.vtkPolyDataWriter()
    writer.SetInput(surface)
    writer.SetFileTypeToASCII()
    writer.SetFileName(filename)
    writer.Write()


def writevtp(surface, filename):
    """Write vtkPolyData file in XML format."""
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInput(surface)
    writer.SetFileName(filename)
    writer.Write()

def writemetaimage(image, filename):
    """Write image file in mhd format."""
    writer = vtk.vtkMetaImageWriter()
    writer.SetFileName(filename)
    writer.SetInput(image)
    writer.Write()
