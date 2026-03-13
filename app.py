from flask import Flask, render_template, jsonify
import random
import heapq
import json
from collections import deque

app = Flask(__name__)

MAZE_WIDTH = 37
MAZE_HEIGHT = 37


def generate_maze(width=50, height=50):
    """Generate a maze using randomized DFS (recursive backtracker)."""
    # 0 = wall, 1 = path
    maze = [[0] * width for _ in range(height)]

    def neighbors(r, c):
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(dirs)
        result = []
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                result.append((nr, nc, r + dr // 2, c + dc // 2))
        return result

    # Start from (0, 0)
    maze[0][0] = 1
    stack = [(0, 0)]
    visited = {(0, 0)}

    while stack:
        r, c = stack[-1]
        found = False
        for nr, nc, wr, wc in neighbors(r, c):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                maze[nr][nc] = 1
                maze[wr][wc] = 1
                stack.append((nr, nc))
                found = True
                break
        if not found:
            stack.pop()

    # Collect all path cells
    path_cells = [(r, c) for r in range(height) for c in range(width) if maze[r][c] == 1]

    # Pick random start and end that are far enough apart
    # Try to ensure a minimum Manhattan distance for a meaningful maze
    min_distance = (width + height) // 2
    attempts = 0
    while True:
        start = random.choice(path_cells)
        end = random.choice(path_cells)
        dist = abs(start[0] - end[0]) + abs(start[1] - end[1])
        attempts += 1
        if dist >= min_distance:
            break
        # After many attempts, relax the constraint
        if attempts > 200:
            min_distance = max(min_distance // 2, 10)
            attempts = 0

    return maze, start, end


def shortest_path_distances(maze, start):
    """Return shortest path distance from start to every reachable path cell."""
    rows, cols = len(maze), len(maze[0])
    dist = {start: 0}
    queue = deque([start])

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 1 and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                queue.append((nr, nc))

    return dist


def dijkstra_steps(maze, start, end):
    """Return the ordered list of cells Dijkstra visits (step by step)."""
    rows, cols = len(maze), len(maze[0])
    dist = {start: 0}
    prev = {}
    visited = set()
    heap = [(0, start)]
    steps = []

    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        steps.append(node)
        if node == end:
            break
        r, c = node
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 1:
                nd = d + 1
                if (nr, nc) not in dist or nd < dist[(nr, nc)]:
                    dist[(nr, nc)] = nd
                    prev[(nr, nc)] = node
                    heapq.heappush(heap, (nd, (nr, nc)))

    # Reconstruct path
    path = []
    node = end
    while node in prev:
        path.append(node)
        node = prev[node]
    path.append(start)
    path.reverse()

    full_dist = shortest_path_distances(maze, start)
    weights_serial = {
        f"{r},{c}": {'g': distance}
        for (r, c), distance in full_dist.items()
    }

    return steps, path, weights_serial


def astar_steps(maze, start, end):
    """Return the ordered list of cells A* visits (step by step)."""
    rows, cols = len(maze), len(maze[0])

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    prev = {}
    visited = set()
    heap = [(f_score[start], 0, start)]  # (f, tiebreaker, node)
    counter = 1
    steps = []

    while heap:
        f, _, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        steps.append(node)
        g = g_score[node]
        if node == end:
            break
        r, c = node
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 1:
                ng = g_score[node] + 1
                if (nr, nc) not in g_score or ng < g_score[(nr, nc)]:
                    g_score[(nr, nc)] = ng
                    f_score[(nr, nc)] = ng + heuristic((nr, nc), end)
                    prev[(nr, nc)] = node
                    heapq.heappush(heap, (f_score[(nr, nc)], counter, (nr, nc)))
                    counter += 1

    # Reconstruct path
    path = []
    node = end
    while node in prev:
        path.append(node)
        node = prev[node]
    path.append(start)
    path.reverse()

    full_dist = shortest_path_distances(maze, start)
    weights_serial = {
        f"{r},{c}": {
            'g': distance,
            'h': heuristic((r, c), end),
            'f': distance + heuristic((r, c), end)
        }
        for (r, c), distance in full_dist.items()
    }

    return steps, path, weights_serial


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/player-vs-algorithm')
def player_vs_algorithm():
    return render_template('player_vs_algorithm.html')


@app.route('/algorithm-comparison')
def algorithm_comparison():
    return render_template('algorithm_comparison.html')


@app.route('/api/generate-maze')
def api_generate_maze():
    maze, start, end = generate_maze(MAZE_WIDTH, MAZE_HEIGHT)
    return jsonify({
        'maze': maze,
        'start': list(start),
        'end': list(end),
        'rows': len(maze),
        'cols': len(maze[0])
    })


@app.route('/api/solve/<algorithm>')
def api_solve(algorithm):
    maze_json = __import__('flask').request.args.get('maze')
    start_json = __import__('flask').request.args.get('start')
    end_json = __import__('flask').request.args.get('end')

    maze = json.loads(maze_json)
    start = tuple(json.loads(start_json))
    end = tuple(json.loads(end_json))

    if algorithm == 'dijkstra':
        steps, path, weights = dijkstra_steps(maze, start, end)
    elif algorithm == 'astar':
        steps, path, weights = astar_steps(maze, start, end)
    else:
        return jsonify({'error': 'Unknown algorithm'}), 400

    return jsonify({
        'steps': [list(s) for s in steps],
        'path': [list(p) for p in path],
        'weights': weights
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
