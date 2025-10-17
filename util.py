import gymnasium

def get_full_state(env):
    return [1]
    # print(env.unwrapped.agents)
    # return {
    #     "agent_positions": [agent.pos for agent in env.unwrapped.agents],
    #     "agent_directions": [agent.dir for agent in env.unwrapped.agents],
    #     "shelves": [(s.pos, s.levels) for s in env.unwrapped.shelves],
    #     "requests": list(env.unwrapped.requests),
    # }