from rich.progress import ProgressColumn


class OptimizeColumn(ProgressColumn):
    """Renders optimization progress data"""

    def render(self, task: "Task") -> Text:
        """Show data transfer speed."""
        speed = task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        data_speed = filesize.decimal(int(speed))
        return Text(f"{data_speed}/s", style="progress.data.speed")
