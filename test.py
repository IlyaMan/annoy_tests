# python3.6.8
from annoy import AnnoyIndex
import random
import ujson
import time
from tqdm import trange


def gen_rand(size, dim):
    print(f"GEN_RAND: size={size} dim={dim}")
    temp = 2**32
    vectors = []
    for i in trange(size):
        vectors.append([int(random.random() * temp) for _ in range(dim)])
    return vectors


def save_vectors(array, file_name):
    with open(file_name + ".json", "w") as file:
        file.write(ujson.dumps(array))


def build_annoy(file_name, trees=1, dim=128):
    vectors = ujson.loads(open(file_name + ".json", "r").read())
    index = AnnoyIndex(dim)
    for i in range(len(vectors)):
        index.add_item(i, vectors[i])
    index.build(trees)
    index.save(file_name + ".ann")
    return index


def on_disk_build_annoy(file_name, trees=1, dim=128):
    vectors = ujson.loads(open(file_name + ".json", "r").read())
    index = AnnoyIndex(dim)
    index.on_disk_build(file_name + ".ann")

    for i in range(len(vectors)):
        index.add_item(i, vectors[i])
    index.build(trees)
    return index


def read_annoy(file_name, dim):
    index = AnnoyIndex(dim)
    index.load(file_name + '.ann')
    return index


def test_annoy(file_name, reads=10, trees=1, nns=1000, dim=128, on_disk=False):
    build_time = time.time()
    if on_disk:
        on_disk_build_annoy(file_name, trees, dim)
    else:
        build_annoy(file_name, trees, dim)
    build_time = time.time() - build_time

    read_time = time.time()
    index = read_annoy(file_name, dim)
    read_time = time.time() - read_time
    size = index.get_n_items()

    query_time = time.time()
    for i in range(reads):
        neigbours = index.get_nns_by_item(random.randint(0, size - 1), nns)
    query_time = (time.time() - query_time) / reads

    # index is immutable. Rebuilding with changes instead
    rebuild_time = time.time()
    new_index = AnnoyIndex(dim)

    for i in range(index.get_n_items()):
        new_index.add_item(i, index.get_item_vector(i))

    new_index.add_item(size, [random.random() * 2**32 for i in range(dim)])
    new_index.build(trees)
    new_index.save(file_name + ".ann")
    rebuild_time = time.time() - rebuild_time

    return {"size": size, "build": build_time, "load": read_time,
            "find": query_time, "rebuild": rebuild_time, "on_disk": on_disk}


def gen_vectors(dim=128):
    save_vectors(gen_rand(10_000, dim), "s")
    save_vectors(gen_rand(100_000, dim), "m")
    save_vectors(gen_rand(1_000_000, dim), "l")
    save_vectors(gen_rand(2_000_000, dim), "xl")
    # save_vectors(gen_rand(40_000_000, dim), "xxl")  # be carefull, extra slow


def test(dim=128, trees=1):  # More trees -> more accuracy
    # don't try xxl here if you have less then 22 GB RAM
    for i in ["s", "m", "l", "xl"]:
        print(test_annoy(i, reads=10, trees=1, nns=1000, dim=128, on_disk=False))
    # here xxl is ok but it is extremely slow
    for i in ["s", "m", "l", "xl"]:
        print(test_annoy(i, reads=10, trees=1, nns=1000, dim=128, on_disk=True))


gen_vectors(128)
test(128, 1)
