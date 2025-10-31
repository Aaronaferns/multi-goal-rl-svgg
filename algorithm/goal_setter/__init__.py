from abc import ABC, abstractmethod
class GoalGenerator(ABC):
    @abstractmethod
    def create_goals(self, *args, **kwargs):
        """Create and return goals."""
        pass

    @abstractmethod
    def sample_goal(self):
        """Define a method for sampling a single goal."""
        pass
