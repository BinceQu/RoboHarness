"""Synthetic evaluator fixture around the migrated, unmodified core scheduler."""
from tools.util.action_world import RoboDojoActionWorld
from tools.util.observation import ObservationAdapter
from tools.util.storage import PlanStore, SnapshotStore
from tools.util.tracking import PointTracker
from tools.util.scheduler import ToolScheduler

class InterfaceRuntime:
    def __init__(self, task_name='', runs_root=None):
        self.adapter = ObservationAdapter()
        self.world = RoboDojoActionWorld(self.adapter)
        self.snapshots = SnapshotStore(runs_root)
        self.plans = PlanStore(runs_root)
        self.tracker = PointTracker(base_rotation=self.world.base_rotation)
        self.world._official_tracked_object_distances = self.tracker
        self.scheduler = ToolScheduler(adapter=self.adapter, world=self.world, snapshots=self.snapshots,
            plans=self.plans, tracker=self.tracker, task_name=task_name)

    def submit(self, name, args):
        return self.scheduler.submit(name,args)

    def wait(self, job, timeout):
        return self.scheduler.wait(job, timeout)

    def update_observation(self, observation):
        self.scheduler.ingest(observation)

    def get_action(self):
        self.adapter.touch()
        action = self.scheduler.next_action()
        return [] if action is None else [action]
