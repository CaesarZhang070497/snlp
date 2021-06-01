# Adapted from code shared by Lample et al in Deep Learning for Symbolic Mathematics under fair modification for non-commercial uses

from logging import getLogger

from .char_sp import CharSPEnvironment


logger = getLogger()


ENVS = {
    'char_sp': CharSPEnvironment,
}


def build_env(params):
    """
    Build environment.
    """
    env = ENVS[params.env_name](params)

    # tasks
    tasks = [x for x in params.tasks.split(',') if len(x) > 0]
    assert len(tasks) == len(set(tasks)) > 0
    assert all(task in env.TRAINING_TASKS for task in tasks)
    params.tasks = tasks
    logger.info(f'Training tasks: {", ".join(tasks)}')

    return env
