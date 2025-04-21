import sys


def ford_fulkerson(residual, source, sink):
    """
    Compute the maximum flow from source to sink using the Ford–Fulkerson method.

    :param residual: List[List[int]]
        A 2D residual capacity matrix of size N×N where residual[u][v] is the
        remaining capacity from node u to node v.
    :param source: int
        Index of the source node in the residual matrix.
    :param sink: int
        Index of the sink node in the residual matrix.
    :return: int
        The total maximum flow value from source to sink.
    """
    n = len(residual)  # Number of nodes in the network
    max_flow = 0  # Accumulator for total flow sent

    def dfs_find_path():
        """

        :return:
        """
        visited = [False] * n
        parent = [-1] * n

        def dfs(u):
            """
            Locate an augmenting path in the residual graph via DFS.

            Returns:
              dfs (function): recursive function to perform the DFS search.
              parent (List[int]): parent[v] gives the node preceding v on the path.
              visited (List[bool]): tracks which nodes have been visited.
            """
            visited[u] = True
            # If we reached sink, path is found
            if u == sink:
                return True
            for v in range(n):
                # Traverse edges with remaining capacity > 0
                if not visited[v] and residual[u][v] > 0:
                    parent[v] = u
                    if dfs(v):
                        return True
            return False

        return dfs, parent, visited

    # Continuously search for augmenting paths and augment flow
    while True:
        dfs, parent, visited = dfs_find_path()
        if not dfs(source):
            # No augmenting path exists meaning the maximum flow has been reached
            break

        # Find bottleneck
        path_flow = float('inf')
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, residual[u][v])
            v = u

        # Augment flow along the path and update residual capacities
        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= path_flow  # reduce forward capacity
            residual[v][u] += path_flow  # increase reverse capacity
            v = u

        max_flow += path_flow

    return max_flow


def parse_input():
    """
    Parses standard input for the soccer elimination problem.

    :return: (teams, scores, matches).
      teams: list of team names in the order read.
      scores: dictionary mapping team name (str) to current score (int).
      matches: list of (team1, team2) tuples for remaining matches.
    """
    # Read number of teams
    n = int(sys.stdin.readline().strip())
    teams = []  # list to hold team names
    scores = {}  # dictionary to map team name to current score

    # Read each team's name and score
    for _ in range(n):
        name = sys.stdin.readline().strip()  # team name
        score = int(sys.stdin.readline().strip())  # current score
        teams.append(name)
        scores[name] = score

    # Read number of remaining matches
    m = int(sys.stdin.readline().strip())
    matches = []  # list to hold tuples of remaining matches

    # Read each match pairing
    for _ in range(m):
        t1 = sys.stdin.readline().strip()  # first team in match
        t2 = sys.stdin.readline().strip()  # second team in match
        matches.append((t1, t2))

    return teams, scores, matches


def build_graph(teams, scores, matches, team):
    """
    Construct the residual graph for testing elimination of 'team'.

    :param teams:   list of all team names
    :param scores:  dictionary mapping each team name to its current score
    :param matches: list of (team1, team2) tuples for remaining games
    :param team:    name of the team to build graph for
    :return: tuple (
      residual_matrix (list of lists) - capacity matrix of size N×N
      match_start_index (int)         - index in the matrix where match nodes begin
      num_rem_matches (int)           - number of match nodes
    ) or (None, reason) if quick elimination applies
    """
    # Determine the highest current score among all teams
    leader_score = max(scores.values())

    # Current score of the team being evaluated
    current_score = scores[team]

    # Count how many remaining games involve 'team'
    num_remaining_matches = sum(1 for x, y in matches if x == team or y == team)

    # Elimination even if winning all remaining games can't reach leader
    if current_score + 3 * num_remaining_matches < leader_score:
        return None, 'cannot_catch_up'

    # Filter out matches involving 'team'
    rem = [(x, y) for x, y in matches if x != team and y != team]
    # If no other matches remain, 'team' automatically stays in contention
    if not rem:
        return [[ ]], 'no_matches'

    # List of other teams (excluding 'team')
    other = [t for t in teams if t != team]
    idx = {t:i for i, t in enumerate(other)}

    num_remaining_matches = len(rem)
    source = 0
    match_start = 1
    team_start = match_start + num_remaining_matches
    sink = team_start + len(other)
    N = sink + 1
    INF = 4000

    # Initialize N×N residual capacity matrix with zeros
    residual = [[0] * N for _ in range(N)]

    # Connect source to each match node with capacity 3
    for i, (x, y) in enumerate(rem):
        m_id = match_start + i
        residual[source][m_id] = 3
        # Connect match node to the two participating teams with 'infinite' capacity
        residual[m_id][team_start + idx[x]] = INF
        residual[m_id][team_start + idx[y]] = INF

    # Maximum points 'team' can still achieve
    max_possible = current_score + 3 * num_remaining_matches
    # Connect each other team node to sink with capacity limiting their points
    for t in other:
        t_id = team_start + idx[t]
        cap = max_possible - scores[t]
        residual[t_id][sink] = max(cap, 0)

    return residual, match_start, num_remaining_matches


def evaluate_team(teams, scores, matches, team):
    """
    Determine if team can still compete for the title.

    :param teams:   list of all team names
    :param scores:  dictionary mapping each team name to its current score
    :param matches: list of (team1, team2) tuples representing remaining games
    :param team:    name of the team to evaluate
    :return: tuple (
      alive (bool)           - True if z remains in contention, False if eliminated
      score (int)            - current score of team z
      reason (str or None)   - elimination reason if eliminated, else None
      flow (int or None)     - max-flow value if still in contention, else None
    )
    """
    # Build the residual flow network for the team
    graph_data = build_graph(teams, scores, matches, team)

    # If graph_data[0] is None, the team cannot catch up to the leader
    if graph_data[0] is None:
        return False, scores[team], 'as it cannot catch up', None

    # Get the residual matrix, starting index for matches, and number of match nodes
    residual, match_start, num_remaining_matches = graph_data

    # Special case: if no matches remain, match_start is returned as [[ ]]
    if match_start == [[ ]]:
        # The team stays in contention by default, with zero extra flow needed
        return True, scores[team], None, 0

    # Define source and sink node indices for the flow graph
    source = 0
    sink = len(residual) - 1

    # Compute maximum flow from source to sink
    flow = ford_fulkerson(residual, source, sink)

    # If we fail to push all 3 * num_remaining_matches points, z is eliminated by flow constraints
    if flow < 3 * num_remaining_matches:
        return False, scores[team], 'due to smaller max flow', None

    # Otherwise, the team remains in contention, and return computed flow
    return True, scores[team], None, flow


def main():
    # Read input data
    teams, scores, matches = parse_input()
    # Instantiate output lists
    teams_in_contention, teams_eliminated = [], []

    # Evaluate each team's elimination status
    for team in teams:
        alive, score, reason, flow = evaluate_team(teams, scores, matches, team)
        if alive:
            teams_in_contention.append((team, score, flow))
        else:
            teams_eliminated.append((team, score, reason))

    # Sort by descending score
    teams_in_contention.sort(key=lambda x: x[1], reverse=True)
    # Sort eliminated by descending score, then reverse alpha for ties
    teams_eliminated.sort(key=lambda x: (x[1], x[0]), reverse=True)

    # Output teams still in contention
    print("Teams still challenging are...")
    for name, score, flow in teams_in_contention:
        print(f"{name} {score} (flow = {flow})")

    # Output eliminated teams
    print("\nTeams that have been eliminated are...")
    for name, score, reason in teams_eliminated:
        print(f"{name} {score} ({reason})")


main()
