import argparse
import io
import multiprocessing
import os
import pickle
import random
import shutil
import time
from multiprocessing import Process

import matplotlib.pyplot as plt
import meshio
import numpy as np
import torch
from PIL import Image
from PyFoam.Execution.BasicRunner import BasicRunner
from PyFoam.RunDictionary.ParsedParameterFile import ParsedBoundaryDict
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from fluidfoam import readmesh
from fluidfoam import readscalar
from fluidfoam import readvector
from shapely.geometry import Point, Polygon
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from MeshGenerator import generate_mesh, sort_points


def generate_object_mask(sol_dir, x_res, y_res):
    msh = meshio.read(sol_dir + "/mesh.msh")

    meshio.write(sol_dir + "/mesh_legacy.msh", msh, file_format="gmsh22", binary=False)

    eps = 0.0075
    min_samples = 2
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    ngridx = x_res
    ngridy = y_res

    msh.points[:, 1].max()

    xinterpmin = msh.points[:, 0].min()

    xinterpmax = msh.points[:, 0].max()

    yinterpmin = msh.points[:, 1].min()

    yinterpmax = msh.points[:, 1].max()

    xi = np.linspace(xinterpmin, xinterpmax, ngridx)
    yi = np.linspace(yinterpmin, yinterpmax, ngridy)
    xinterp, yinterp = np.meshgrid(xi, yi)

    wall_points = msh.points[msh.cells_dict['quad'][msh.cell_sets_dict['wall']['quad']]]
    wp_corrected = wall_points[(wall_points[:, :, 2] == 0)][:, :2][1:-1:2]
    wp_corrected = [(p[0], p[1]) for p in wp_corrected]
    wp_corrected = torch.tensor(wp_corrected)

    clusters = dbscan.fit_predict(wp_corrected)

    if len(set(clusters)) < 1:
        return None

    p_clusters = []
    for cluster_id in set(clusters):
        cluster_points = torch.cat(
            [wp_corrected[clusters == cluster_id], wp_corrected[clusters == cluster_id][0].view(1, 2)])
        p_clusters.append(cluster_points)

    polygon_list = [Polygon(sort_points([(p[0].item(), p[1].item()) for p in p_clusters[i]])) for i in
                    range(len(p_clusters))]

    interp = torch.tensor(np.stack((xinterp, yinterp), axis=2)).flatten(0, 1)

    object_mask = []
    for p in tqdm(interp):
        mask_value = 0
        for polygon in polygon_list:
            mask_value += polygon.contains(Point(p))

        object_mask.append(mask_value)

    object_mask = torch.tensor(object_mask).view(ngridy, ngridx).flip(0)
    return object_mask, len(set(clusters))


def readU(arg):
    i, dest = arg
    return torch.tensor(readvector(dest, str(i), 'U'))


def readp(arg):
    i, dest = arg
    return torch.tensor(readscalar(dest, str(i), 'p'))


def readPhi(arg):
    i, dest = arg
    return torch.tensor(readscalar(dest, str(i), 'phi'))


plot_height = 5.0


def scatter_plot(arg):
    i, x, y, v, triangles, mesh_points = arg
    fig = plt.figure(figsize=(plot_height * 2.8, plot_height), dpi=100)
    plt.tripcolor(mesh_points[:, 0], mesh_points[:, 1], triangles, v[i], alpha=1.0, shading='flat', antialiased=True,
                  linewidth=0.72, edgecolors='face')
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    plt.close(fig)
    # print(i)
    return Image.open(img_buf)


def prepareCase(src, dest, n_points, velocity, x_res=384, y_res=256):
    if os.path.exists(dest) and os.path.isdir(dest):
        shutil.rmtree(dest)
    shutil.copytree(src, dest)

    while True:
        process = Process(target=generate_mesh, args=(dest + "mesh.msh", n_points))
        try:
            process.start()
            process.join()
        except Exception as e:
            print("Error during mesh generation, retrying:", e)
        finally:
            process.terminate()

        object_mask, n_detected_objects = generate_object_mask(dest, x_res, y_res)

        if process.exitcode == 0 and object_mask is not None and n_points == n_detected_objects:
            print("Mesh generated")
            break
        else:
            print("retry mesh generation")
            print("objects (expected vs detected):", n_points, n_detected_objects)
            time.sleep(3)

    runner = BasicRunner(argv=["gmshToFoam", "-case", dest, dest + "mesh_legacy.msh"], logname="logifle", noLog=True)
    runner.start()

    f = ParsedBoundaryDict(dest + "constant/polyMesh/boundary")
    f['FrontBackPlane']['type'] = 'empty'
    f.writeFile()

    f = ParsedParameterFile(dest + "0/U")
    f['internalField'] = 'uniform (' + str(velocity) + ' 0 0)'
    f.writeFile()

    return object_mask


def get_current_case(parent_directory):
    directories = [d for d in os.listdir(parent_directory)
                   if os.path.isdir(os.path.join(parent_directory, d))]
    if not directories:
        return 0
    existing_numbers = [int(d.split('_')[1]) for d in directories
                        if d.startswith("case_") and d[5:].isdigit()]
    for i in range(1, max(existing_numbers) + 2):
        if i not in existing_numbers:
            return i


def process_case(args, current_case_number, work_dir, target_dir, save_raw=1):
    n_points = random.randint(1, args.n_objects)
    velocity = random.uniform(0.01, 0.06)
    nr_time_steps = 0

    crash_counter = 0
    object_mask = prepareCase(args.empty_case, work_dir, n_points, velocity)

    msh = meshio.read(work_dir + "/mesh.msh")
    triangles = msh.cells_dict['triangle'][(msh.points[msh.cells_dict['triangle']][:, :, -1] == 0)[:, 0]]
    mesh_points = msh.points

    time.sleep(5)

    delta_t_index = 0
    delta_t = [0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001, 0.00005, 0.000025, 0.00001]
    f = ParsedParameterFile(work_dir + "system/controlDict")
    max_time_steps = f['endTime']
    f['deltaT'] = delta_t[delta_t_index]
    f.writeFile()

    try:
        while nr_time_steps < max_time_steps:
            delta_t_index += 1
            if crash_counter > 0:
                f = ParsedParameterFile(work_dir + "system/controlDict")
                f['deltaT'] = delta_t[delta_t_index]
                f.writeFile()

            runner = BasicRunner(argv=["foamRun", "-solver", "incompressibleFluid", "-case", work_dir],
                                 logname="logifle", noLog=True, silent=True)
            run_information = runner.start()
            nr_time_steps = run_information['time']
            crash_counter += 1

    except IndexError:
        print("List out of bound, restarting outer loop")

    solution_dir = target_dir + "/case_" + str(current_case_number) + "/"
    try:
        os.mkdir(solution_dir)
    except OSError as error:
        raise Exception("Creation of the directory %s failed: %s" % (solution_dir, error))

    os.remove(work_dir + "PyFoamState.CurrentTime")
    os.remove(work_dir + "PyFoamState.LastOutputSeen")
    os.remove(work_dir + "PyFoamState.StartedAt")
    os.remove(work_dir + "PyFoamState.TheState")
    for item in os.listdir(work_dir):
        if item.endswith(".foam"):
            os.remove(os.path.join(work_dir, item))

    torch.save(object_mask, solution_dir + "object_mask.th")

    with ParsedParameterFile(work_dir + "0/U") as f:
        initial_velocity = float(str(f['internalField']).split('(')[1].split(" ")[0])

    simulation_description = {"initial_velocity": initial_velocity, "n_objects": n_points}

    with open(solution_dir + "simulation_description.pkl", 'wb') as handle:
        pickle.dump(simulation_description, handle)

    shutil.make_archive(base_name=solution_dir + "mesh",
                        format='bztar',
                        root_dir=work_dir,
                        base_dir="mesh.msh")

    x, y, z = readmesh(work_dir)

    U = [readU((i, work_dir)) for i in range(1, max_time_steps + 1)]
    p = [readp((i, work_dir)) for i in range(1, max_time_steps + 1)]

    U_stacked = torch.stack(U)

    x = torch.tensor(x)
    y = torch.tensor(y)
    v = torch.sqrt(U_stacked[:, 0, :] ** 2 + U_stacked[:, 1, :] ** 2 + U_stacked[:, 2, :] ** 2)

    if save_raw == 1:

        for i in range(len(U)):
            local_U = U[i].view(3, -1)[:2]
            local_p = p[i].view(1, -1)
            torch.save(torch.cat([local_U, local_p], dim=0), solution_dir + ('{:0>8}'.format(str(i))) + "_mesh.th")
        torch.save(x, solution_dir + "x.th")
        torch.save(y, solution_dir + "y.th")

    try:
        shutil.rmtree(work_dir)
    except Exception as e:
        print(f"Failed to remove {work_dir}: {e}")
    pool_obj = multiprocessing.Pool()
    img_list = pool_obj.map(scatter_plot, [(i, x, y, v, triangles, mesh_points) for i in range(U_stacked.shape[0])])

    pool_obj.close()

    for i in range(len(img_list)):
        img_list[i]._min_frame = 0
        img_list[i].n_frames = 1
        img_list[i]._PngImageFile__frame = 0

    img_list[0].save(solution_dir + 'U.gif', format='GIF', append_images=img_list[1:], save_all=True, duration=50,
                     loop=0)

    print("target directory: ", solution_dir)

    time.sleep(5)


def worker(q, args, case_counter, lock, work_dir, target_dir):
    while True:
        task_item = q.get()
        if task_item is None:
            break
        with lock:
            current_case_number = case_counter.value
            case_counter.value += 1
        print(f'Worker at {work_dir} processing case {current_case_number}')
        process_case(args, current_case_number, work_dir, target_dir)
        q.task_done()


def remove_all_folders(root_dir):
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            try:
                shutil.rmtree(item_path)
                print(f"Removed folder: {item_path}")
            except Exception as e:
                print(f"Failed to remove {item_path}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_objects', type=int,
                        help='maximum number of circles/partial circles the case should have (min is 1)')
    parser.add_argument('n_cases', type=int, help='number of cases to be run')
    parser.add_argument('n_workers', type=int, default=multiprocessing.cpu_count(),
                        help='Number of worker processes to run in parallel (default: number of CPU cores)')
    parser.add_argument('empty_case', type=str, help='the empty OpenFOAM case directory')
    parser.add_argument('target_dir', type=str, help='target directory for the OpenFOAM Cases')
    parser.add_argument('working_dir', type=str, help='working directory for OpenFOAM Simulation')
    args = parser.parse_args()

    assert args.n_objects > 0, "At least one object must be specified"
    assert args.n_workers > 0, "At least one worker must be specified"

    target_dir = os.path.join(args.target_dir, '')

    q = multiprocessing.JoinableQueue()
    manager = multiprocessing.Manager()
    case_counter = manager.Value('i', get_current_case(target_dir))
    lock = manager.Lock()

    delta_t = [0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001, 0.00005, 0.000025, 0.00001]
    delta_t_index = 0

    try:
        processes = []
        for i in range(args.n_workers):
            working_dir = os.path.join(args.working_dir, f"worker_{i}", '')
            os.mkdir(working_dir)
            p = multiprocessing.Process(target=worker, args=(q, args, case_counter, lock, working_dir, target_dir))
            p.start()
            processes.append(p)

        for task_item in range(args.n_cases):
            q.put(task_item)

        q.join()

        for _ in range(args.n_workers):
            q.put(None)

        for p in processes:
            p.join()
    finally:
        remove_all_folders(args.working_dir)
        print("All simulation tasks completed.")
