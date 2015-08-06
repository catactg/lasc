"""Extract surface mesh from binary mask, run standardisation, and compute
evaluation metrics"""
import os
import sys
from lasc_benchmark_tools import *


path_groundtruth = sys.argv[1]
segmentation_file = sys.argv[2]
visualise_on = int(sys.argv[3])

# input arguments
print path_groundtruth, segmentation_file, visualise_on

segmentation_pardir = os.path.dirname(segmentation_file)

# if dimensions don't match, nothing will be executed
ifile = os.path.join(path_groundtruth, 'gt_std.mhd')
match = check_images(ifile, segmentation_file)

if match:
    # extract the mesh from all values >= 1
    image = readmetaimage(segmentation_file)
    mesh = labels2mesh(image, 1.)
    ofile = os.path.join(segmentation_pardir, 'mesh.vtp')
    writevtp(mesh, ofile)

    # clip with mitral plane
    planefile = os.path.join(path_groundtruth, 'mitralplane.csv')
    meshclip = clip_mitral(mesh, planefile)

    # transfer autolabels to mesh
    ifile = os.path.join(path_groundtruth, 'gt_noclip_mesh.vtp')
    refmesh = readvtp(ifile)
    meshlabels = transfer_gtlabels(refmesh, meshclip, 'autolabels')

    # clip veins
    targetdist = 10
    meshpvs = clip_vein_endpoint(meshlabels, path_groundtruth, targetdist)
    ofile = os.path.join(segmentation_pardir,'std_mesh.vtp')
    writevtp(meshpvs, ofile)

    # compute dice metric
    ifile = os.path.join(path_groundtruth, 'gt_std.mhd')
    ofile = os.path.join(segmentation_pardir, 'std.mhd')
    dm_all = compute_dice(ifile, meshpvs, ofile)

    # separate body / pvs and save
    labels = {'body': [36], 'pvs': [77, 76, 78, 79]}
    indexes = ['body', 'pvs']
    for index in indexes:
        ofile = os.path.join(segmentation_pardir, 'dm_' + index + '.csv')
        writearray2csv(dm_all[index], ofile, labels[index])

    # compute s2s error
    ifile = os.path.join(path_groundtruth, 'gt_std_mesh.vtp')
    ofile = os.path.join(segmentation_pardir, 'std_')
    refmesh = readvtp(ifile)
    nsamples = 1000 # total samples will be <= nsamples x 2 due to symmetry
    s2s_all = compute_s2s_error(refmesh, meshpvs, nsamples,ofile)

    # separate body / pvs and save
    indexes = ['body','pvs']
    for index in indexes:
        ofile = os.path.join(segmentation_pardir, 's2s_' + index + '.csv')
        writearray2csv(s2s_all[index], ofile)

    if visualise_on:
        ifile = os.path.join(path_groundtruth, 'gt_std_mesh.vtp')
        refmesh = readvtp(ifile)
        ifile = os.path.join(segmentation_pardir, 'std_mesh.vtp')
        meshpvs = readvtp(ifile)
        regionslabels = getregionslabels()
        visualise(meshpvs, refmesh, segmentation_file, 'autolabels',
                  regionslabels['body'], regionslabels['pv4'])










