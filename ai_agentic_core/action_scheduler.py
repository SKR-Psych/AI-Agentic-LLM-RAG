import time

class ActionScheduler:
    def __init__(self):
        self.queue = []

    def schedule(self, action: str, delay: int = 0):
        self.queue.append((action, time.time() + delay))

    def get_ready_actions(self):
        now = time.time()
        return [action for action, exec_time in self.queue if exec_time <= now]
