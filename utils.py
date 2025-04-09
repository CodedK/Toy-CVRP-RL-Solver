
import numpy as np
import re

def parse_vrp_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    coords, demands, depot_id = {}, {}, None
    capacity = 0
    section = None

    for line in lines:
        line = line.strip()
        if 'CAPACITY' in line:
            capacity = int(re.findall(r'\d+', line)[0])
        elif line.startswith('NODE_COORD_SECTION'):
            section = 'NODE'
        elif line.startswith('DEMAND_SECTION'):
            section = 'DEMAND'
        elif line.startswith('DEPOT_SECTION'):
            section = 'DEPOT'
        elif line.startswith('EOF'):
            break
        elif section == 'NODE':
            parts = line.split()
            if len(parts) >= 3:
                coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
        elif section == 'DEMAND':
            parts = line.split()
            if len(parts) >= 2:
                demands[int(parts[0])] = int(parts[1])
        elif section == 'DEPOT' and line.isdigit():
            depot_id = int(line)

    return coords, demands, depot_id, capacity

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))
